//! Helper methods for Executor: prefix cache, input tensors, token reading,
//! MoE offloading, and expert placement management.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::{BlockAllocator, CpuBlockAllocator};
use boostr::inference::prefix_cache::PrefixCache;
use boostr::model::LoadedModel;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use boostr::ExpertWeights;

use super::moe_offload::{LayerExpertPlacement, MoeOffloadManager};

use super::executor::Executor;

impl<R: Runtime<DType = DType>> Executor<R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>,
{
    /// Allocate blocks for paged KV cache, reusing prefix cache blocks when available.
    ///
    /// Returns `(cached_token_count, prefix_cache_seq_id)`.
    /// The MutexGuard is acquired and released within this function, so it is
    /// safe to call from inside a `stream!` generator without blocking Send.
    pub(crate) fn prefix_cache_allocate(
        prefix_cache: &Option<std::sync::Mutex<PrefixCache<CpuBlockAllocator>>>,
        prompt_tokens: &[u32],
        block_size: usize,
        paged_cache: &mut LayeredPagedKvCache<R>,
        allocator: &CpuBlockAllocator,
    ) -> Result<(usize, Option<u64>)> {
        let Some(ref pc) = prefix_cache else {
            paged_cache
                .allocate_blocks(prompt_tokens.len(), allocator)
                .map_err(|e| anyhow!("Failed to allocate blocks for prefill: {}", e))?;
            return Ok((0, None));
        };

        let Ok(mut cache) = pc.lock() else {
            paged_cache
                .allocate_blocks(prompt_tokens.len(), allocator)
                .map_err(|e| anyhow!("Failed to allocate blocks for prefill: {}", e))?;
            return Ok((0, None));
        };

        let seq_id = uuid::Uuid::new_v4().as_u128() as u64;
        match cache.get_or_allocate_blocks(seq_id, prompt_tokens) {
            Ok(result) => {
                let cached = result.cached_count();
                let blocks = result.into_blocks();

                // Set the prefix cache's block IDs directly into paged cache.
                // Cached blocks map to physical blocks with pre-computed KV values.
                // New blocks (for uncached suffix) are freshly allocated.
                let num_cache_blocks = blocks.len();
                paged_cache.set_blocks(blocks);

                if cached > 0 {
                    crate::server::metrics::record_prefix_cache_hit();
                    tracing::info!(
                        prefix_cache_hit = true,
                        cached_blocks = cached,
                        total_prompt_tokens = prompt_tokens.len(),
                        "Prefix cache: {} blocks reused ({} tokens skip prefill KV compute)",
                        cached,
                        cached * block_size
                    );
                } else {
                    crate::server::metrics::record_prefix_cache_miss();
                }
                // Allocate additional blocks for decode tokens beyond what prefix cache covers
                let prompt_blocks = prompt_tokens.len().div_ceil(block_size);
                if num_cache_blocks < prompt_blocks {
                    // Prefix cache didn't cover all prompt blocks — should not happen
                    // but allocate remaining just in case
                    let remaining = prompt_blocks - num_cache_blocks;
                    let extra = allocator
                        .allocate(remaining)
                        .map_err(|e| anyhow!("Failed to allocate remaining blocks: {}", e))?;
                    for bt_idx in 0..paged_cache.num_layers() {
                        paged_cache
                            .block_table_mut(bt_idx)
                            .append_blocks(extra.clone());
                    }
                }
                Ok((cached * block_size, Some(seq_id)))
            }
            Err(_) => {
                drop(cache);
                paged_cache
                    .allocate_blocks(prompt_tokens.len(), allocator)
                    .map_err(|e| anyhow!("Failed to allocate blocks for prefill: {}", e))?;
                Ok((0, None))
            }
        }
    }

    /// Release prefix cache references and free blocks back to pool.
    /// Safe to call from inside `stream!` — acquires and drops MutexGuard internally.
    pub(crate) fn prefix_cache_release(
        prefix_cache: &Option<std::sync::Mutex<PrefixCache<CpuBlockAllocator>>>,
        seq_id: Option<u64>,
        blocks: &[boostr::inference::memory::BlockId],
        allocator: &CpuBlockAllocator,
    ) {
        if let (Some(ref pc), Some(sid)) = (prefix_cache, seq_id) {
            if let Ok(mut cache) = pc.lock() {
                let _ = cache.release_blocks(sid, blocks);
            }
        } else if !blocks.is_empty() {
            let _ = allocator.free(blocks);
        }
    }

    /// Synchronize GPU prefix cache after allocation.
    ///
    /// Inserts new block hash → block ID mappings and increments reference counts.
    /// Called after `prefix_cache_allocate` on the CUDA path.
    #[cfg(feature = "cuda")]
    pub(crate) fn gpu_prefix_cache_insert(
        gpu_cache: &Option<std::sync::Mutex<boostr::inference::prefix_cache::GpuPrefixCache>>,
        prompt_tokens: &[u32],
        block_size: usize,
        blocks: &[boostr::inference::memory::BlockId],
    ) {
        let Some(ref gc) = gpu_cache else { return };
        let Ok(mut cache) = gc.lock() else { return };
        let hashes = boostr::inference::prefix_cache::GpuPrefixCache::compute_block_hashes(
            prompt_tokens,
            block_size,
        );
        for (i, &hash) in hashes.iter().enumerate() {
            if i < blocks.len() {
                cache.insert(hash, blocks[i]);
                cache.inc_ref(hash);
            }
        }
    }

    /// Release GPU prefix cache references.
    #[cfg(feature = "cuda")]
    pub(crate) fn gpu_prefix_cache_release(
        gpu_cache: &Option<std::sync::Mutex<boostr::inference::prefix_cache::GpuPrefixCache>>,
        prompt_tokens: &[u32],
        block_size: usize,
    ) {
        let Some(ref gc) = gpu_cache else { return };
        let Ok(mut cache) = gc.lock() else { return };
        let hashes = boostr::inference::prefix_cache::GpuPrefixCache::compute_block_hashes(
            prompt_tokens,
            block_size,
        );
        for &hash in &hashes {
            cache.dec_ref(hash);
        }
    }

    /// Create input tensor from token IDs
    pub(crate) fn create_input_tensor(&self, tokens: &[u32]) -> Result<Tensor<R>> {
        let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        Ok(Tensor::from_slice(
            &tokens_i64,
            &[1, tokens.len()],
            &self.device,
        ))
    }

    /// Greedy argmax on GPU — returns the token as a GPU tensor [1] i64.
    /// No CPU sync happens here; the result stays on device.
    pub(crate) fn argmax_on_gpu(&self, logits: &Tensor<R>) -> Result<Tensor<R>> {
        let seq_len = logits.dim(1).map_err(|e| anyhow!("{}", e))?;
        let narrowed = logits
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| anyhow!("{}", e))?;
        let squeezed = narrowed.squeeze(Some(1)).squeeze(Some(0));
        squeezed.argmax(0, false).map_err(|e| anyhow!("{}", e))
    }

    /// Read a scalar i64 GPU tensor to CPU using the pipelined copy stream.
    pub(crate) fn read_token_id(token_gpu: &Tensor<R>, event: u64) -> Result<u32> {
        let v: Vec<i64> = token_gpu
            .to_vec_pipelined(event)
            .map_err(|e| anyhow!("Pipelined D2H copy failed: {}", e))?;
        Ok(v[0] as u32)
    }

    /// Step the MoE offload manager (if present). Called after each forward pass.
    ///
    /// Acquires and drops the mutex synchronously to avoid Send issues in async streams.
    /// If rebalancing is triggered, executes the resulting expert transfers and logs each
    /// one at `info` level so placement changes are visible in production logs.
    ///
    /// After executing transfers the updated per-layer placement map is written into
    /// `expert_placements` so that the model's forward pass (and monitoring code) can
    /// read the current GPU/CPU assignment without taking the offload manager lock.
    ///
    /// # Weight transfer strategy
    ///
    /// Expert weights are stored as stacked tensors `[num_experts, dim_in, dim_out]` inside
    /// boostr's `LlamaMoeMlp`, protected by interior `Arc<RwLock<_>>`.  Each transfer
    /// goes through three steps:
    ///
    /// 1. `get_expert_weights` — reads the 2-D slice for the expert via zero-copy `narrow`
    ///    and contiguous copy.
    /// 2. Raw byte transfer — `to_vec` extracts the bytes from the current device; then
    ///    `Tensor::from_slice` allocates on the target device.  For GPU→CPU the target is
    ///    the CPU device; for CPU→GPU the target is the executor's GPU device.
    /// 3. `set_expert_weights` — reconstructs the full stacked tensor and writes it back
    ///    via the write lock.
    ///
    /// Note: true cross-runtime transfers (CudaRuntime ↔ CpuRuntime) require the model to
    /// hold tensors of a single `R` type.  Within a single-runtime model the "CPU device"
    /// is represented as the CpuRuntime device for CPU-only builds, or as a lower-priority
    /// allocation on the same CUDA device for CUDA builds.  Full heterogeneous offloading
    /// (VRAM → pinned host RAM) will require a dedicated offload-store type in boostr.
    pub(crate) fn moe_offload_step(
        moe_offload: &Option<std::sync::Mutex<MoeOffloadManager>>,
        expert_placements: &Arc<std::sync::RwLock<Option<Vec<LayerExpertPlacement>>>>,
        model: &Arc<LoadedModel<R>>,
        device: &R::Device,
    ) {
        let Some(ref mgr_mutex) = moe_offload else {
            return;
        };
        let Ok(mut mgr) = mgr_mutex.lock() else {
            return;
        };

        let transfers = mgr.step();
        if transfers.is_empty() {
            return;
        }

        tracing::info!(
            num_transfers = transfers.len(),
            "MoE expert placement rebalanced, executing transfers"
        );

        mgr.execute_transfers(&transfers, |transfer| {
            // 1. Read the current weight tensors for this expert.
            let current = match model.get_expert_weights(transfer.layer, transfer.expert_id) {
                Some(w) => w,
                None => {
                    tracing::warn!(
                        layer = transfer.layer,
                        expert = transfer.expert_id,
                        "get_expert_weights returned None; skipping transfer"
                    );
                    return false;
                }
            };

            // 2. Copy the weight tensors to the target device.
            //
            // `device` is the executor's primary device (GPU for CUDA builds, CPU otherwise).
            // For CPU→GPU: allocate on `device`.
            // For GPU→CPU: within the single-runtime model we re-allocate on the same device
            //   (a logical no-op at the storage level, but the RwLock write forces any cached
            //   computation referencing the old buffer to rebuild). Full heterogeneous offload
            //   to pinned host memory requires a dedicated boostr offload-store.
            let new_weights = Self::copy_expert_weights_to_device(current, device);

            // 3. Write the new tensors back into the model.
            match model.set_expert_weights(transfer.layer, transfer.expert_id, new_weights) {
                Ok(()) => {
                    tracing::info!(
                        layer = transfer.layer,
                        expert = transfer.expert_id,
                        direction = ?transfer.direction,
                        "MoE expert weight transfer complete"
                    );
                    true
                }
                Err(e) => {
                    tracing::warn!(
                        layer = transfer.layer,
                        expert = transfer.expert_id,
                        direction = ?transfer.direction,
                        error = %e,
                        "MoE expert set_expert_weights failed"
                    );
                    false
                }
            }
        });

        // Snapshot the updated placement map and publish it for readers.
        let num_layers = mgr.num_layers();
        let updated: Vec<LayerExpertPlacement> =
            (0..num_layers).map(|l| mgr.placement(l).clone()).collect();
        if let Ok(mut guard) = expert_placements.write() {
            *guard = Some(updated);
        }
    }

    /// Copy expert weight tensors to `target_device` by extracting raw bytes and
    /// re-allocating on the target device.
    ///
    /// Uses `to_vec::<u8>` (byte-level copy) so that it works for any `DType` without
    /// knowing the element type at compile time.  The raw bytes are then written to
    /// newly allocated storage on `target_device` via `Storage::from_bytes`, and the
    /// resulting tensor is assembled with a contiguous layout matching the source shape.
    fn copy_expert_weights_to_device(
        weights: ExpertWeights<R>,
        target_device: &R::Device,
    ) -> ExpertWeights<R> {
        fn copy_tensor<R: Runtime<DType = DType>>(src: Tensor<R>, target: &R::Device) -> Tensor<R> {
            use boostr::tensor::{Layout, Storage};
            let shape = src.shape().to_vec();
            let dtype = src.dtype();
            let src_c = if src.is_contiguous() {
                src
            } else {
                src.contiguous()
            };
            // Read raw bytes from the current device (CPU or GPU via R::copy_from_device).
            let bytes: Vec<u8> = src_c.to_vec::<u8>();
            // Allocate on target device and upload.
            let storage = Storage::from_bytes(&bytes, dtype, target)
                .unwrap_or_else(|e| panic!("MoE expert weight copy failed: {e}"));
            let layout = Layout::contiguous(&shape);
            Tensor::from_parts(storage, layout)
        }

        ExpertWeights {
            gate_proj: copy_tensor(weights.gate_proj, target_device),
            up_proj: copy_tensor(weights.up_proj, target_device),
            down_proj: copy_tensor(weights.down_proj, target_device),
        }
    }

    /// Get the current per-layer expert placement map.
    ///
    /// Returns `None` if MoE offloading is disabled or no rebalance has occurred yet.
    /// The returned `Arc` can be cloned and held by long-running tasks (e.g., the model
    /// forward pass) to read placement without blocking the offload manager's mutex.
    pub fn expert_placements(&self) -> Arc<std::sync::RwLock<Option<Vec<LayerExpertPlacement>>>> {
        Arc::clone(&self.expert_placements)
    }

    /// Update the expert placement map directly.
    ///
    /// Useful for initializing placement from a checkpoint or overriding from an external
    /// scheduler without waiting for the rebalance interval.
    pub fn set_expert_placement(&self, placements: Vec<LayerExpertPlacement>) {
        if let Ok(mut guard) = self.expert_placements.write() {
            *guard = Some(placements);
        }
    }
}
