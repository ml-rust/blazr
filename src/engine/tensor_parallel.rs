//! Tensor parallelism support for multi-GPU inference.
//!
//! Wraps boostr's existing `ColumnParallelLinear`/`RowParallelLinear` (Megatron-LM style)
//! and provides blazr-specific validation, setup, and lifecycle management.
//!
//! # Architecture
//!
//! When `tensor_parallel_size > 1`:
//! 1. Each GPU gets a shard of the model (Q/K/V heads split across GPUs)
//! 2. After attention and MLP, an all-reduce synchronizes outputs
//! 3. Embedding and LM head use vocab-parallel sharding
//!
//! boostr provides: `ColumnParallelLinear`, `RowParallelLinear`, NCCL communicator.
//! blazr owns: configuration, validation, and wiring into the executor.

use crate::config::InferenceConfig;

/// Tensor parallelism runtime state.
///
/// Holds the communicator and rank/world_size for the duration of inference.
/// Created once at model load time if `tensor_parallel_size > 1`.
pub struct TensorParallelState {
    /// Which rank this process is (0-indexed)
    pub rank: usize,
    /// Total number of participating GPUs
    pub world_size: usize,
    /// NCCL communicator used for all-reduce during forward passes.
    /// Kept alive here so it is not dropped while the executor exists.
    #[cfg(feature = "cuda")]
    pub comm: std::sync::Arc<dyn boostr::runtime::Communicator>,
}

impl TensorParallelState {
    /// Create TP state for a given rank and world size (non-CUDA / test path).
    #[cfg(not(feature = "cuda"))]
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    /// Create TP state with an NCCL communicator (CUDA path).
    #[cfg(feature = "cuda")]
    pub fn with_comm(
        rank: usize,
        world_size: usize,
        comm: std::sync::Arc<dyn boostr::runtime::Communicator>,
    ) -> Self {
        Self {
            rank,
            world_size,
            comm,
        }
    }

    /// Whether this is the root rank (rank 0)
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }

    /// Compute the shard range for a dimension of size `total`.
    /// Returns `(start, end)` for this rank's portion.
    pub fn shard_range(&self, total: usize) -> (usize, usize) {
        let per_rank = total / self.world_size;
        let remainder = total % self.world_size;
        let start = self.rank * per_rank + self.rank.min(remainder);
        let end = start + per_rank + if self.rank < remainder { 1 } else { 0 };
        (start, end)
    }
}

/// Validate that the tensor parallelism configuration is feasible.
///
/// Checks:
/// - CUDA feature is enabled when tp_size > 1
/// - num_kv_heads is divisible by tp_size (required for head sharding)
/// - num_attention_heads is divisible by tp_size
pub fn validate_tp_config(
    config: &InferenceConfig,
    num_heads: usize,
    num_kv_heads: usize,
) -> Result<(), String> {
    let tp = config.tensor_parallel_size;
    if tp <= 1 {
        return Ok(());
    }

    if cfg!(not(feature = "cuda")) {
        return Err("Tensor parallelism requires the 'cuda' feature".to_string());
    }

    if !num_heads.is_multiple_of(tp) {
        return Err(format!(
            "num_attention_heads ({}) must be divisible by tensor_parallel_size ({})",
            num_heads, tp
        ));
    }
    if !num_kv_heads.is_multiple_of(tp) {
        return Err(format!(
            "num_kv_heads ({}) must be divisible by tensor_parallel_size ({})",
            num_kv_heads, tp
        ));
    }
    Ok(())
}

/// Log tensor parallelism configuration at model load time.
pub fn log_tp_config(tp_size: usize) {
    if tp_size > 1 {
        tracing::info!(
            tensor_parallel_size = tp_size,
            "Tensor parallelism enabled: model sharded across {} GPUs",
            tp_size
        );
        tracing::info!(
            "  Using boostr ColumnParallelLinear (Q/K/V/gate/up) + RowParallelLinear (O/down)"
        );
    }
}

/// Initialize tensor parallelism for model loading.
///
/// Creates a NCCL communicator, logs configuration, and returns a
/// `TensorParallelState` that keeps the communicator alive for the
/// duration of the executor. Call this before loading model weights.
#[cfg(feature = "cuda")]
pub fn init_tp(
    tp_size: usize,
    rank: usize,
) -> Result<
    (
        TensorParallelState,
        std::sync::Arc<dyn boostr::runtime::Communicator>,
    ),
    String,
> {
    if tp_size <= 1 {
        return Err("init_tp called with tp_size <= 1".to_string());
    }
    log_tp_config(tp_size);
    let comm = create_nccl_communicator(rank, tp_size)?;
    let state = TensorParallelState::with_comm(rank, tp_size, std::sync::Arc::clone(&comm));
    Ok((state, comm))
}

/// Create a NCCL communicator for tensor parallelism.
///
/// Returns a communicator that can be passed to `LoadedModel::load_tp()`.
/// For single-process multi-GPU, rank is the GPU index.
#[cfg(feature = "cuda")]
pub fn create_nccl_communicator(
    rank: usize,
    world_size: usize,
) -> Result<std::sync::Arc<dyn boostr::runtime::Communicator>, String> {
    use boostr::runtime::cuda::NcclCommunicator;

    let comm = NcclCommunicator::new(rank, world_size).map_err(|e| {
        format!(
            "Failed to create NCCL communicator (rank={}, world_size={}): {}",
            rank, world_size, e
        )
    })?;

    Ok(std::sync::Arc::new(comm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_shard_range_even() {
        let state = TensorParallelState::new(0, 4);
        assert_eq!(state.shard_range(32), (0, 8));
        let state = TensorParallelState::new(3, 4);
        assert_eq!(state.shard_range(32), (24, 32));
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_shard_range_uneven() {
        let state = TensorParallelState::new(0, 3);
        assert_eq!(state.shard_range(10), (0, 4)); // 3+1 remainder
        let state = TensorParallelState::new(2, 3);
        assert_eq!(state.shard_range(10), (7, 10)); // 3
    }

    #[test]
    fn test_validate_single_gpu_always_ok() {
        let config = InferenceConfig {
            tensor_parallel_size: 1,
            ..Default::default()
        };
        assert!(validate_tp_config(&config, 32, 8).is_ok());
        assert!(validate_tp_config(&config, 7, 3).is_ok());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_validate_divisible() {
        let config = InferenceConfig {
            tensor_parallel_size: 4,
            ..Default::default()
        };
        assert!(validate_tp_config(&config, 32, 8).is_ok());
        assert!(validate_tp_config(&config, 32, 6).is_err());
    }
}
