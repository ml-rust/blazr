//! Construction of a [`BatchEngine`]: the refusals of models the paged batch
//! cannot serve, and the size of the KV pool it allocates.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::BlockAllocator;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, TypeConversionOps, UnaryOps,
};

use super::batch_engine::BatchEngine;
use super::request_scheduler::RequestScheduler;
use super::seq_sampler::BatchSeqs;
use crate::engine::executor::Executor;

impl<R: Runtime<DType = DType>> BatchEngine<R>
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
        + RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    /// Create a new batch engine
    pub fn new(executor: Arc<Executor<R>>, scheduler: Arc<RequestScheduler>) -> Result<Self> {
        let model = executor.model()?;
        if model.needs_gdn_state() {
            return Err(crate::engine::qwen35_unsupported(
                "continuous batching",
                None,
                Some("set max_batch_size = 1, or use CUDA lanes with paged_attention off"),
            ));
        }
        if model.supports_hybrid_rows() {
            return Err(anyhow!(
                "model type '{}' keeps a contiguous KV cache per sequence and decodes in hybrid \
                 lanes, not in the paged batch; serve it through HybridLaneEngine",
                model.model_type()
            ));
        }
        let config = executor.config();
        if !config.inference.paged_attention {
            return Err(anyhow!(
                "the continuous batch runs over the paged KV pool, and \
                 inference.paged_attention is off; turn it on, or set max_batch_size = 1"
            ));
        }
        let paged_cache = Some(LayeredPagedKvCache::new(
            model.num_layers(),
            Self::kv_pool_blocks(&executor)?,
            config.inference.block_size,
            model.num_kv_heads().unwrap_or(8),
            model.head_dim().unwrap_or(64),
            config.kv_cache_dtype()?,
            executor.device(),
        )?);

        let seqs = BatchSeqs::new(executor.vocab_size(), executor.device())?;
        Ok(Self {
            executor,
            scheduler,
            paged_cache,
            prefill_chunk_size: None,
            seqs,
            #[cfg(test)]
            after_chunk: None,
        })
    }

    /// Blocks in the KV pool an engine over `executor` allocates.
    ///
    /// The scheduler's block allocator must hold exactly this many blocks,
    /// so no block id it hands out falls outside the engine's cache.
    ///
    /// # Errors
    ///
    /// An error when the executor's block allocator lock is poisoned.
    pub fn kv_pool_blocks(executor: &Executor<R>) -> Result<usize> {
        let config = executor.config();
        if config.inference.kv_pool_blocks > 0 {
            return Ok(config.inference.kv_pool_blocks);
        }
        match executor.shared_allocator() {
            Some(shared) => Ok(shared
                .lock()
                .map_err(|e| anyhow!("Block allocator lock poisoned: {e}"))?
                .total_blocks()),
            None => Ok(1024),
        }
    }
}
