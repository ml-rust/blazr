//! Model warmup for pre-loading CUDA PTX modules and JIT compilation.

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::{BlockAllocator, CpuBlockAllocator};
use boostr::inference::{LayeredKvCache, LayeredSsmState};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::parse_dtype;

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
    /// Warm up the model by running a dummy forward pass
    ///
    /// This pre-loads all CUDA PTX modules and triggers JIT compilation,
    /// eliminating the ~90ms first-run overhead from TTFT.
    /// Call this after model loading, before the first generation.
    pub fn warmup(&self) -> Result<()> {
        tracing::debug!("Warming up model kernels...");
        let start = std::time::Instant::now();

        // Create a single-token input
        let warmup_input = Tensor::from_slice(&[1i64], &[1, 1], self.device());

        if self.model().needs_ssm_state() {
            self.warmup_mamba(&warmup_input)?;
        } else if self.config().inference.paged_attention {
            self.warmup_paged(&warmup_input)?;
        } else {
            self.warmup_contiguous(&warmup_input)?;
        }

        // Warmup argmax + pipelined D2H copy
        let vocab_size = self.model().vocab_size();
        let dummy_logits = Tensor::zeros(&[1, 1, vocab_size], DType::F32, self.device());
        let token_gpu = self.argmax_on_gpu(&dummy_logits)?;
        let event = token_gpu
            .record_event()
            .map_err(|e| anyhow!("Warmup record event failed: {}", e))?;
        let _ = Self::read_token_id(&token_gpu, event)?;

        tracing::debug!("Model warmup complete in {:?}", start.elapsed());
        Ok(())
    }

    fn warmup_mamba(&self, warmup_input: &Tensor<R>) -> Result<()> {
        let mamba_config = self
            .model()
            .mamba_config()
            .ok_or_else(|| anyhow!("Mamba2 model missing mamba config"))?;
        let num_layers = self.model().num_layers();
        let warmup_dtype = parse_dtype(self.config().dtype())?;

        let mut ssm_state =
            LayeredSsmState::new(num_layers, 1, mamba_config, warmup_dtype, self.device());

        let _ = self
            .model()
            .forward_with_ssm_state(warmup_input, &mut ssm_state)
            .map_err(|e| anyhow!("Warmup forward pass failed: {}", e))?;
        Ok(())
    }

    fn warmup_paged(&self, warmup_input: &Tensor<R>) -> Result<()> {
        let num_layers = self.model().num_layers();
        let num_kv_heads = self.model().num_kv_heads().unwrap_or(8);
        let head_dim = self.model().head_dim().unwrap_or(64);
        let block_size = self.config().inference.block_size;
        let kv_dtype = parse_dtype(self.config().dtype())?;

        let num_blocks = 4;
        // Use shared allocator if available, otherwise create a small one for warmup
        let allocator = if let Some(shared) = self.shared_allocator() {
            shared
                .lock()
                .map_err(|e| anyhow::anyhow!("Block allocator lock poisoned: {e}"))?
                .clone()
        } else {
            CpuBlockAllocator::new(num_blocks, block_size)
        };

        let mut paged_cache = LayeredPagedKvCache::new(
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            kv_dtype,
            self.device(),
        );

        // Prefill warmup
        paged_cache
            .allocate_blocks(1, &allocator)
            .map_err(|e| anyhow!("Warmup paged block alloc failed: {}", e))?;
        let slot_vec = paged_cache
            .compute_slot_mapping(0, 1)
            .map_err(|e| anyhow!("Warmup slot mapping failed: {}", e))?;
        let slot_mapping = Tensor::from_slice(&slot_vec, &[1], self.device());
        let bt_vec = paged_cache.block_table_device_format(0);
        let block_table = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], self.device());
        paged_cache.set_seq_len(1);

        let _ = self
            .model()
            .forward_with_paged_kv_cache(
                warmup_input,
                &paged_cache,
                &slot_mapping,
                &block_table,
                1,
                0,
            )
            .map_err(|e| anyhow!("Warmup paged prefill failed: {}", e))?;

        // Decode warmup
        paged_cache
            .allocate_blocks(1, &allocator)
            .map_err(|e| anyhow!("Warmup paged decode block alloc failed: {}", e))?;
        let slot_vec = paged_cache
            .compute_slot_mapping(1, 1)
            .map_err(|e| anyhow!("Warmup decode slot mapping failed: {}", e))?;
        let slot_mapping = Tensor::from_slice(&slot_vec, &[1], self.device());
        let bt_vec = paged_cache.block_table_device_format(0);
        let block_table = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], self.device());
        paged_cache.set_seq_len(2);

        let _ = self
            .model()
            .forward_with_paged_kv_cache(
                warmup_input,
                &paged_cache,
                &slot_mapping,
                &block_table,
                2,
                1,
            )
            .map_err(|e| anyhow!("Warmup paged decode failed: {}", e))?;

        // Free warmup blocks back to shared pool
        let blocks_to_free: Vec<_> = paged_cache.block_table(0).blocks.clone();
        if !blocks_to_free.is_empty() {
            let _ = allocator.free(&blocks_to_free);
        }
        Ok(())
    }

    fn warmup_contiguous(&self, warmup_input: &Tensor<R>) -> Result<()> {
        let num_layers = self.model().num_layers();
        let num_kv_heads = self.model().num_kv_heads().unwrap_or(8);
        let head_dim = self.model().head_dim().unwrap_or(64);
        let kv_dtype = parse_dtype(self.config().dtype())?;

        let mut kv_cache = LayeredKvCache::new_positional(
            num_layers,
            1,
            num_kv_heads,
            16,
            16,
            head_dim,
            kv_dtype,
            self.device(),
        )
        .map_err(|e| anyhow!("Failed to create warmup KV cache: {}", e))?;

        let _ = self
            .model()
            .forward_with_kv_cache(warmup_input, &mut kv_cache, 0)
            .map_err(|e| anyhow!("Warmup prefill forward pass failed: {}", e))?;

        let _ = self
            .model()
            .forward_with_kv_cache(warmup_input, &mut kv_cache, 1)
            .map_err(|e| anyhow!("Warmup decode forward pass failed: {}", e))?;
        Ok(())
    }
}
