//! Helper methods for Executor: prefix cache, input tensors, and token
//! reading.

use anyhow::{anyhow, Result};

use boostr::inference::LayeredKvCache;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use super::executor::Executor;
use super::paged_pool::PoolLease;
use crate::engine::tensor_parallel::KvCacheSpec;

/// Lease the prompt blocks of `prompt_tokens`, reusing prefix cache blocks.
///
/// Returns the count of leading prompt tokens whose KV the reused blocks
/// already hold. The MutexGuard drops inside this function, so a `stream!`
/// generator can call it and stay `Send`.
///
/// # Errors
///
/// An error when the pool holds too few free blocks.
pub(crate) fn prefix_cache_allocate(
    lease: &mut PoolLease<'_>,
    prompt_tokens: &[u32],
) -> Result<usize> {
    let cached = match lease.prefix_cache() {
        Some(pc) => match pc.lock() {
            Ok(mut cache) => {
                let seq_id = uuid::Uuid::new_v4().as_u128() as u64;
                cache
                    .get_or_allocate_blocks(seq_id, prompt_tokens)
                    .ok()
                    .map(|result| (seq_id, result.cached_count(), result.into_blocks()))
            }
            Err(_) => None,
        },
        None => None,
    };

    let mut cached_blocks = 0;
    if let Some((seq_id, count, blocks)) = cached {
        lease.hold_prefix_blocks(seq_id, blocks);
        cached_blocks = count;
        if count > 0 {
            crate::telemetry::record::record_prefix_cache_hit();
            tracing::info!(
                prefix_cache_hit = true,
                cached_blocks = count,
                total_prompt_tokens = prompt_tokens.len(),
                "Prefix cache: {count} blocks reused",
            );
        } else {
            crate::telemetry::record::record_prefix_cache_miss();
        }
    }
    // The prefix cache covers every prompt block. Without it, or when it
    // refuses, this allocates them all.
    lease.cover(prompt_tokens.len())?;
    Ok((cached_blocks * lease.block_size()).min(prompt_tokens.len()))
}

/// Synchronize GPU prefix cache after allocation.
///
/// Inserts new block hash → block ID mappings and increments reference counts.
/// Called after [`prefix_cache_allocate`] on the CUDA path.
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
        self.argmax_ids(&squeezed, 0)
    }

    /// Argmax of `logits` over `axis`, as I64 ids on the device.
    ///
    /// Axis 2 of `[1, n, vocab]` logits gives the `[1, n]` greedy ids of
    /// every row. Ties break to the lowest id.
    pub(crate) fn argmax_ids(&self, logits: &Tensor<R>, axis: usize) -> Result<Tensor<R>> {
        let ids = logits.argmax(axis, false).map_err(|e| anyhow!("{}", e))?;
        if ids.dtype() == DType::I64 {
            return Ok(ids);
        }
        // WebGPU has no 64-bit integers, so its argmax returns I32. Every
        // token reader takes I64. The cast moves one element per row.
        R::default_client(&self.device)
            .cast(&ids, DType::I64)
            .map_err(|e| anyhow!("argmax token cast to I64: {e}"))
    }

    /// A fresh contiguous KV cache for one sequence, and its shape for the
    /// tensor-parallel followers.
    ///
    /// The capacity is `prompt_len + max_tokens`, at most `max_seq_len`.
    pub(crate) fn contiguous_kv_cache(
        &self,
        prompt_len: usize,
        max_tokens: usize,
    ) -> Result<(LayeredKvCache<R>, KvCacheSpec)> {
        let max_seq_len = self.config.max_seq_len();
        let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
        let head_dim = self.model.head_dim().unwrap_or(64);
        let initial_capacity = (prompt_len + max_tokens).min(max_seq_len);
        let dtype = self.config.kv_cache_dtype()?;
        let cache = LayeredKvCache::new_positional(
            self.model.num_layers(),
            1,
            num_kv_heads,
            initial_capacity,
            max_seq_len,
            head_dim,
            dtype,
            &self.device,
        )
        .map_err(|e| anyhow!("Failed to create KV cache: {}", e))?;
        let spec = KvCacheSpec {
            initial_capacity,
            max_seq_len,
            dtype,
        };
        Ok((cache, spec))
    }

    /// Read a scalar i64 GPU tensor to CPU using the pipelined copy stream.
    pub(crate) fn read_token_id(token_gpu: &Tensor<R>, event: u64) -> Result<u32> {
        let v: Vec<i64> = token_gpu
            .to_vec_pipelined(event)
            .map_err(|e| anyhow!("Pipelined D2H copy failed: {}", e))?;
        v.first().map(|&id| id as u32).ok_or_else(|| {
            anyhow!(
                "token read returned no element from a tensor of shape {:?}; the sampler \
                 must write a [1] I64 token",
                token_gpu.shape()
            )
        })
    }
}
