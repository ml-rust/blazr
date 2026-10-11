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

/// Query and key/value head counts that tensor parallelism shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpHeads {
    pub num_heads: usize,
    pub num_kv_heads: usize,
}

/// Check that a tensor parallel degree can run.
///
/// `serve` calls it at start with `heads` unset, before any model loads.
/// The loader calls it again with the model's heads, before weights load.
///
/// Checks:
/// - `tp_size` is at least 1.
/// - Above 1, the build has the `cuda` and `nccl` features.
/// - The query and key/value head counts divide by `tp_size`.
pub fn validate_tp_config(tp_size: usize, heads: Option<TpHeads>) -> Result<(), String> {
    if tp_size == 0 {
        return Err(
            "tensor_parallel_size is 0. Set inference.tensor_parallel_size or BLAZR_TP_SIZE \
             to 1 or more"
                .to_string(),
        );
    }
    if tp_size == 1 {
        return Ok(());
    }
    if cfg!(not(all(feature = "cuda", feature = "nccl"))) {
        return Err(format!(
            "tensor_parallel_size {tp_size} needs the cuda and nccl features. Rebuild with \
             --features cuda,nccl, or set the size to 1"
        ));
    }
    let Some(heads) = heads else {
        return Ok(());
    };
    if !heads.num_heads.is_multiple_of(tp_size) {
        return Err(format!(
            "num_attention_heads ({}) does not divide by tensor_parallel_size ({tp_size}). \
             Pick a size that divides it",
            heads.num_heads
        ));
    }
    if !heads.num_kv_heads.is_multiple_of(tp_size) {
        return Err(format!(
            "num_kv_heads ({}) does not divide by tensor_parallel_size ({tp_size}). Pick a \
             size that divides it",
            heads.num_kv_heads
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
#[cfg(all(feature = "cuda", feature = "nccl"))]
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
#[cfg(all(feature = "cuda", feature = "nccl"))]
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

    fn heads(num_heads: usize, num_kv_heads: usize) -> Option<TpHeads> {
        Some(TpHeads {
            num_heads,
            num_kv_heads,
        })
    }

    #[test]
    fn test_validate_single_gpu_always_ok() {
        assert!(validate_tp_config(1, heads(32, 8)).is_ok());
        assert!(validate_tp_config(1, heads(7, 3)).is_ok());
        assert!(validate_tp_config(1, None).is_ok());
    }

    #[test]
    fn test_validate_zero_is_an_error() {
        assert!(validate_tp_config(0, None).is_err());
    }

    #[test]
    #[cfg(not(all(feature = "cuda", feature = "nccl")))]
    fn test_validate_needs_cuda_and_nccl() {
        let err = validate_tp_config(2, None).unwrap_err();
        assert!(err.contains("cuda,nccl"), "{err}");
    }

    #[test]
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fn test_validate_divisible() {
        assert!(validate_tp_config(4, None).is_ok());
        assert!(validate_tp_config(4, heads(32, 8)).is_ok());
        assert!(validate_tp_config(4, heads(32, 6)).is_err());
    }
}
