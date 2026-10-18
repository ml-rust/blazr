//! Prefill for the continuous batching engine.
//!
//! A prefill runs one sequence's prompt through the paged KV cache, whole or
//! in chunks, then samples and emits its first token through
//! [`emit_token`], as every later decode step does.

use anyhow::{anyhow, Result};

use boostr::inference::scheduler::{ScheduledBatch, SequenceId};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use super::batch_emit::emit_token;
use super::batch_engine::BatchEngine;
use super::kv_recompute::{block_ids, block_slots};
use super::request_cancel::{end_cancelled, CancelPhase};
use crate::config::GenerationConfig;

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
    /// Chunked prefill: process prompt in chunks, interleaving decode steps.
    ///
    /// Returns whether the sequence ended on its first token.
    pub(super) async fn process_prefill_chunked(
        &mut self,
        seq_id: SequenceId,
        batch: &ScheduledBatch,
        chunk_size: usize,
    ) -> Result<bool> {
        let prompt_tokens = self
            .scheduler
            .get_prompt_tokens(seq_id)
            .ok_or_else(|| anyhow!("sequence {} not found", seq_id))?;

        // Skip cached tokens from prefix cache
        let cached_count = batch.cached_token_counts.get(&seq_id).copied().unwrap_or(0);

        if prompt_tokens.len() - cached_count <= chunk_size {
            // Small enough (after skipping cache) to do in one shot
            return self.process_prefill(seq_id, batch).await;
        }

        let gen_config = self
            .scheduler
            .get_gen_config(seq_id)
            .ok_or_else(|| anyhow!("gen config for seq {} not found", seq_id))?;

        let block_size = self
            .paged_cache
            .as_ref()
            .ok_or_else(|| anyhow!("paged cache not initialized"))?
            .block_size();
        let blocks = batch
            .block_tables
            .get(&seq_id)
            .ok_or_else(|| anyhow!("block table for seq {} not found", seq_id))?;

        let device = self.executor.device().clone();

        // Process prompt in chunks, starting after cached tokens
        let total = prompt_tokens.len();
        let mut processed = cached_count;
        let mut logits = None;

        while processed < total {
            // A sequence cancelled between chunks stops here and frees its blocks.
            if !self.scheduler.is_live(seq_id) {
                end_cancelled(
                    &self.scheduler,
                    &mut self.seqs,
                    seq_id,
                    CancelPhase::Prefill,
                )?;
                return Ok(true);
            }
            let end = (processed + chunk_size).min(total);
            let chunk = &prompt_tokens[processed..end];

            let slot_mapping_vec = block_slots(blocks, block_size, processed..end);
            let slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[chunk.len()], &device)?;

            let bt_vec = block_ids(blocks);
            let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], &device)?;

            let input = self.executor.create_input_tensor(chunk)?;

            let paged_cache = self
                .paged_cache
                .as_ref()
                .ok_or_else(|| anyhow!("paged cache not initialized"))?;

            self.seqs.count_forward();
            logits = Some(
                self.executor
                    .model()
                    .forward_with_paged_kv_cache(
                        &input,
                        paged_cache,
                        &slot_mapping,
                        &block_table_tensor,
                        end,       // seq_len_k = total tokens up to this chunk
                        processed, // start_pos for this chunk
                        None,
                        None,
                    )
                    .map_err(|e| {
                        anyhow!("chunked prefill failed at offset {}: {}", processed, e)
                    })?,
            );

            processed = end;
            #[cfg(test)]
            if let Some(hook) = self.after_chunk.as_mut() {
                hook(processed);
            }

            // Between chunks, run pending decode steps for other sequences
            if processed < total && !batch.decode_sequences.is_empty() {
                self.process_decode(&batch.decode_sequences, batch).await?;
            }

            tracing::debug!(
                "Chunked prefill: {}/{} tokens for seq {}",
                processed,
                total,
                seq_id
            );
        }

        // Sample first token from final chunk's logits
        let logits = logits.ok_or_else(|| anyhow!("no logits produced during prefill"))?;
        self.first_token(seq_id, &prompt_tokens, gen_config, &logits)
            .await
    }

    /// Sample `seq_id`'s first token from its prefill logits and stream it.
    ///
    /// The sequence's sampler state starts here, from its prompt and its
    /// request's config. Returns whether the sequence ended.
    async fn first_token(
        &mut self,
        seq_id: SequenceId,
        prompt_tokens: &[u32],
        gen_config: GenerationConfig,
        logits: &Tensor<R>,
    ) -> Result<bool> {
        self.seqs
            .begin(&self.executor, seq_id, prompt_tokens, gen_config)?;
        emit_token(
            &self.executor,
            &self.scheduler,
            &mut self.seqs,
            seq_id,
            logits,
            CancelPhase::Prefill,
        )
        .await
    }

    /// Process a single prefill sequence.
    ///
    /// Returns whether the sequence ended on its first token.
    pub(super) async fn process_prefill(
        &mut self,
        seq_id: SequenceId,
        batch: &ScheduledBatch,
    ) -> Result<bool> {
        let prompt_tokens = self
            .scheduler
            .get_prompt_tokens(seq_id)
            .ok_or_else(|| anyhow!("sequence {} not found", seq_id))?;

        let gen_config = self
            .scheduler
            .get_gen_config(seq_id)
            .ok_or_else(|| anyhow!("gen config for sequence {} not found", seq_id))?;

        let paged_cache = self
            .paged_cache
            .as_ref()
            .ok_or_else(|| anyhow!("paged cache not initialized"))?;

        let block_size = paged_cache.block_size();
        let blocks = batch
            .block_tables
            .get(&seq_id)
            .ok_or_else(|| anyhow!("block table for sequence {} not found", seq_id))?;

        // Prefix cache: skip KV computation for cached tokens
        let cached_count = batch.cached_token_counts.get(&seq_id).copied().unwrap_or(0);
        let prefill_start = cached_count;
        let prefill_tokens = &prompt_tokens[prefill_start..];

        if cached_count > 0 {
            tracing::info!(
                seq_id,
                cached_tokens = cached_count,
                remaining_tokens = prefill_tokens.len(),
                "Prefix cache hit: skipping {} tokens in prefill",
                cached_count,
            );
            crate::telemetry::record::record_prefix_cache_hit();
        } else {
            crate::telemetry::record::record_prefix_cache_miss();
        }

        // Build slot mapping only for uncached tokens
        let slot_mapping_vec = block_slots(blocks, block_size, prefill_start..prompt_tokens.len());
        let device = self.executor.device();
        let slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[prefill_tokens.len()], device)?;

        // Build block table tensor
        let bt_vec = block_ids(blocks);
        let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], device)?;

        // Create input tensor for uncached tokens only
        let input = self.executor.create_input_tensor(prefill_tokens)?;

        // Run prefill forward pass starting from cached offset
        self.seqs.count_forward();
        let logits = self
            .executor
            .model()
            .forward_with_paged_kv_cache(
                &input,
                paged_cache,
                &slot_mapping,
                &block_table_tensor,
                prompt_tokens.len(), // total seq_len (including cached)
                prefill_start,       // start_pos = skip cached tokens
                None,
                None,
            )
            .map_err(|e| anyhow!("prefill forward failed: {}", e))?;

        self.first_token(seq_id, &prompt_tokens, gen_config, &logits)
            .await
    }
}
