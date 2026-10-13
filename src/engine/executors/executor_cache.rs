//! Helper methods for Executor: prefix cache, input tensors, and token
//! reading.

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::{BlockAllocator, CpuBlockAllocator};
use boostr::inference::prefix_cache::PrefixCache;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

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
        + ModelClient<R>
        + boostr::quant::traits::DequantOps<R>,
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
                    crate::telemetry::record::record_prefix_cache_hit();
                    tracing::info!(
                        prefix_cache_hit = true,
                        cached_blocks = cached,
                        total_prompt_tokens = prompt_tokens.len(),
                        "Prefix cache: {} blocks reused ({} tokens skip prefill KV compute)",
                        cached,
                        cached * block_size
                    );
                } else {
                    crate::telemetry::record::record_prefix_cache_miss();
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
        )?)
    }

    /// Greedy argmax on GPU — returns the token as a GPU tensor [1] i64.
    /// No CPU sync happens here; the result stays on device.
    pub(crate) fn argmax_on_gpu(&self, logits: &Tensor<R>) -> Result<Tensor<R>> {
        let seq_len = logits.dim(1).map_err(|e| anyhow!("{}", e))?;
        let narrowed = logits
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| anyhow!("{}", e))?;
        let squeezed = narrowed.squeeze(Some(1)).squeeze(Some(0));
        let token = squeezed.argmax(0, false).map_err(|e| anyhow!("{}", e))?;
        if token.dtype() == DType::I64 {
            return Ok(token);
        }
        // WebGPU has no 64-bit integers, so its argmax returns I32. Every
        // token reader takes I64. The cast moves one element.
        R::default_client(&self.device)
            .cast(&token, DType::I64)
            .map_err(|e| anyhow!("argmax token cast to I64: {e}"))
    }

    /// Read a scalar i64 GPU tensor to CPU using the pipelined copy stream.
    pub(crate) fn read_token_id(token_gpu: &Tensor<R>, event: u64) -> Result<u32> {
        let v: Vec<i64> = token_gpu
            .to_vec_pipelined(event)
            .map_err(|e| anyhow!("Pipelined D2H copy failed: {}", e))?;
        Ok(v[0] as u32)
    }
}
