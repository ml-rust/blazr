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
    /// The follower rank threads that run in lockstep with rank 0.
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub group: Option<super::tp_group::TpGroup>,
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
            #[cfg(feature = "nccl")]
            group: None,
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

/// The shape of a fresh contiguous KV cache, as rank 0 built it. Each
/// follower rank builds a matching cache for its own head shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheSpec {
    pub initial_capacity: usize,
    pub max_seq_len: usize,
    pub dtype: boostr::DType,
}

impl<R> super::Executor<R>
where
    R: boostr::Runtime<DType = boostr::DType>,
    R::Client: boostr::model::ModelClient<R> + boostr::IndexingOps<R>,
{
    /// A forward pass over a contiguous KV cache.
    ///
    /// A tensor-parallel executor runs the step on every rank of its group.
    /// Pass `fresh_cache` when `kv_cache` was just built for a new sequence.
    pub(crate) fn forward_contiguous(
        &self,
        input: &boostr::Tensor<R>,
        kv_cache: &mut boostr::inference::LayeredKvCache<R>,
        position: usize,
        fresh_cache: Option<KvCacheSpec>,
    ) -> anyhow::Result<boostr::Tensor<R>> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(group) = self.tp_state.as_ref().and_then(|s| s.group.as_ref()) {
            return group.forward_with_kv_cache(
                &self.model,
                input,
                kv_cache,
                position,
                fresh_cache,
            );
        }
        let _ = fresh_cache;
        self.model
            .forward_with_kv_cache(input, kv_cache, position)
            .map_err(|e| anyhow::anyhow!("Forward pass failed: {e}"))
    }

    /// The lock a tensor-parallel generation holds from prefill to the last
    /// token. `None` on an executor with no rank group.
    pub(crate) async fn tp_generation_guard(&self) -> Option<tokio::sync::OwnedMutexGuard<()>> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(group) = self.tp_state.as_ref().and_then(|s| s.group.as_ref()) {
            return Some(group.generation_lock().lock_owned().await);
        }
        None
    }

    /// [`Self::tp_generation_guard`] for a caller that cannot wait, such as
    /// warmup. An error when a generation holds the lock.
    pub(crate) fn tp_generation_try_guard(
        &self,
    ) -> anyhow::Result<Option<tokio::sync::OwnedMutexGuard<()>>> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(group) = self.tp_state.as_ref().and_then(|s| s.group.as_ref()) {
            return group
                .generation_lock()
                .try_lock_owned()
                .map(Some)
                .map_err(|_| {
                    anyhow::anyhow!(
                        "a tensor-parallel generation is running. Warm up before serving"
                    )
                });
        }
        Ok(None)
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

/// Environment variable a multi-process launcher sets to give each process
/// its rank. blazr refuses it: every rank runs as a thread of one process.
pub const TP_RANK_ENV: &str = "BLAZR_TP_RANK";

/// Refuse a multi-process tensor-parallel launch.
///
/// A multi-process group needs one NCCL unique id shared by every process,
/// for example over nexar. blazr does not exchange that id. It runs all
/// ranks as threads of one process instead.
pub fn refuse_multi_process(rank_var: Option<&str>) -> Result<(), String> {
    match rank_var {
        None => Ok(()),
        Some(value) => Err(format!(
            "{TP_RANK_ENV}={value} asks for one process per rank. blazr runs every tensor-parallel \
             rank as a thread of one process and shares no NCCL id across processes. Unset \
             {TP_RANK_ENV} and start one blazr process"
        )),
    }
}

/// One NCCL communicator per GPU `base_device..base_device + world_size`.
///
/// All ranks live in this process. `Comm::from_devices` creates one NCCL
/// unique id and initializes every rank with it. Each communicator runs on
/// its device's default compute stream, so collectives order with the
/// model's kernels.
#[cfg(all(feature = "cuda", feature = "nccl"))]
pub fn create_nccl_group(
    base_device: usize,
    world_size: usize,
) -> Result<Vec<std::sync::Arc<dyn boostr::runtime::Communicator>>, String> {
    use boostr::runtime::cuda::NcclCommunicator;
    use boostr::{CudaDevice, CudaRuntime, Runtime};

    let available =
        CudaDevice::count().map_err(|e| format!("could not count CUDA devices: {e}"))?;
    if base_device + world_size > available {
        return Err(format!(
            "tensor_parallel_size {world_size} from GPU {base_device} needs GPUs {base_device}..{} \
             and this host has {available}. Lower the size",
            base_device + world_size - 1
        ));
    }
    let streams = (base_device..base_device + world_size)
        .map(|index| {
            let client = CudaRuntime::default_client(&CudaDevice::new(index));
            std::sync::Arc::clone(client.stream_arc())
        })
        .collect();
    let comms = NcclCommunicator::from_streams(streams).map_err(|e| {
        format!("NCCL init for {world_size} ranks from GPU {base_device} failed: {e}")
    })?;
    Ok(comms
        .into_iter()
        .map(|comm| std::sync::Arc::new(comm) as std::sync::Arc<dyn boostr::runtime::Communicator>)
        .collect())
}

/// Start a single-process tensor-parallel group for one model.
///
/// Creates the NCCL group and starts the follower rank threads, which load
/// their shards at once. The caller loads rank 0 with the returned
/// communicator, then calls [`TpGroup::wait_loaded`] on `state.group`.
#[cfg(all(feature = "cuda", feature = "nccl"))]
pub fn init_tp(
    tp_size: usize,
    base_device: usize,
    model_path: &std::path::Path,
    dtype: Option<crate::config::DtypeRequest>,
) -> Result<
    (
        TensorParallelState,
        std::sync::Arc<dyn boostr::runtime::Communicator>,
    ),
    String,
> {
    if tp_size <= 1 {
        return Err(format!(
            "a tensor-parallel group needs 2 or more ranks, got {tp_size}"
        ));
    }
    refuse_multi_process(std::env::var(TP_RANK_ENV).ok().as_deref())?;
    log_tp_config(tp_size);
    let comms = create_nccl_group(base_device, tp_size)?;
    let group = super::tp_group::TpGroup::spawn(&comms, base_device, model_path, dtype)
        .map_err(|e| format!("{e:#}"))?;
    let comm = std::sync::Arc::clone(&comms[0]);
    let state = TensorParallelState {
        rank: 0,
        world_size: tp_size,
        comm: std::sync::Arc::clone(&comm),
        group: Some(group),
    };
    Ok((state, comm))
}

/// Turn off every inference path a tensor-parallel group cannot drive.
///
/// The follower ranks mirror only the contiguous KV cache forward. Paged
/// attention, speculative decoding and the prefix caches run on rank 0
/// alone, so they stay off.
pub fn restrict_to_lockstep(inference: &mut crate::config::InferenceConfig) {
    if inference.paged_attention
        || inference.speculative.is_some()
        || inference.prefix_cache
        || inference.gpu_prefix_cache
    {
        tracing::info!(
            "tensor parallelism decodes with a contiguous KV cache. Paged attention, \
             speculative decoding and the prefix caches are off"
        );
    }
    inference.paged_attention = false;
    inference.speculative = None;
    inference.prefix_cache = false;
    inference.gpu_prefix_cache = false;
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
    fn multi_process_rank_variable_is_refused() {
        assert!(refuse_multi_process(None).is_ok());
        let err = refuse_multi_process(Some("1")).unwrap_err();
        assert!(err.contains("BLAZR_TP_RANK=1"), "{err}");
        assert!(err.contains("Unset BLAZR_TP_RANK"), "{err}");
    }

    #[test]
    fn lockstep_turns_off_rank0_only_paths() {
        let mut inference = crate::config::InferenceConfig {
            paged_attention: true,
            prefix_cache: true,
            gpu_prefix_cache: true,
            ..Default::default()
        };
        restrict_to_lockstep(&mut inference);
        assert!(!inference.paged_attention);
        assert!(!inference.prefix_cache);
        assert!(!inference.gpu_prefix_cache);
        assert!(inference.speculative.is_none());
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

    /// One NCCL rank per visible GPU initializes, and a group larger than
    /// the host is refused before NCCL runs. Skips with no CUDA device.
    #[test]
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    fn nccl_group_spans_the_visible_gpus_only() {
        let available = match boostr::CudaDevice::count() {
            Ok(n) if n > 0 => n,
            _ => {
                println!("skip: no CUDA device");
                return;
            }
        };
        let comms = create_nccl_group(0, available).expect("NCCL group over every GPU");
        assert_eq!(comms.len(), available);
        for (rank, comm) in comms.iter().enumerate() {
            assert_eq!(comm.rank(), rank);
            assert_eq!(comm.world_size(), available);
        }
        let err = create_nccl_group(0, available + 1)
            .err()
            .expect("more ranks than GPUs");
        assert!(err.contains("Lower the size"), "{err}");
    }
}
