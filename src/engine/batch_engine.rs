//! Continuous batching engine
//!
//! Runs the scheduling loop: pulls batches from RequestScheduler,
//! executes batched forward passes, and streams tokens back to callers.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::BlockAllocator;
use boostr::inference::scheduler::{ScheduledBatch, SequenceId};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use super::batch_decode::process_decode_batch;
use super::executor::Executor;
use super::request_scheduler::RequestScheduler;
use super::types::FinishReason;

use crate::config::parse_dtype;

/// Continuous batching engine that drives the scheduling loop
pub struct BatchEngine<R: Runtime<DType = DType>> {
    executor: Arc<Executor<R>>,
    scheduler: Arc<RequestScheduler>,
    /// Shared paged KV cache (one large pool for all sequences)
    paged_cache: Option<LayeredPagedKvCache<R>>,
}

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
        + ModelClient<R>,
{
    /// Create a new batch engine
    pub fn new(executor: Arc<Executor<R>>, scheduler: Arc<RequestScheduler>) -> Result<Self> {
        let config = executor.config();
        let paged_cache = if config.inference.paged_attention {
            let num_layers = executor.model().num_layers();
            let num_kv_heads = executor.model().num_kv_heads().unwrap_or(8);
            let head_dim = executor.model().head_dim().unwrap_or(64);
            let block_size = config.inference.block_size;
            let kv_dtype = parse_dtype(config.dtype())?;

            // Total blocks in the shared pool
            let total_blocks = if config.inference.kv_pool_blocks > 0 {
                config.inference.kv_pool_blocks
            } else if let Some(shared) = executor.shared_allocator() {
                shared
                    .lock()
                    .expect("block allocator lock poisoned")
                    .total_blocks()
            } else {
                1024
            };

            Some(LayeredPagedKvCache::new(
                num_layers,
                total_blocks,
                block_size,
                num_kv_heads,
                head_dim,
                kv_dtype,
                executor.device(),
            ))
        } else {
            None
        };

        Ok(Self {
            executor,
            scheduler,
            paged_cache,
        })
    }

    /// Run the continuous batching loop. This is the main scheduling loop
    /// that should be spawned as a background task.
    pub async fn run(mut self) {
        let mut notify_rx = match self.scheduler.take_notify_rx() {
            Some(rx) => rx,
            None => {
                tracing::error!("BatchEngine: notify_rx already taken");
                return;
            }
        };

        tracing::info!("BatchEngine: continuous batching loop started");

        loop {
            // Wait for new work or process existing work
            if !self.scheduler.has_work() {
                match notify_rx.recv().await {
                    Some(()) => {}
                    None => {
                        tracing::info!("BatchEngine: notify channel closed, shutting down");
                        break;
                    }
                }
            }

            // Drain any additional notifications
            while notify_rx.try_recv().is_ok() {}

            // Schedule a batch
            match self.scheduler.schedule_batch() {
                Ok(Some(batch)) => {
                    if let Err(e) = self.process_batch(&batch).await {
                        tracing::error!("BatchEngine: batch processing error: {}", e);
                        // Abort all sequences in the failed batch
                        for seq_id in batch.all_sequences() {
                            let _ = self.scheduler.finish_sequence(seq_id);
                        }
                    }
                }
                Ok(None) => {
                    // No work available, yield and wait
                    tokio::task::yield_now().await;
                }
                Err(e) => {
                    tracing::error!("BatchEngine: scheduling error: {}", e);
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            }

            // Periodic cleanup
            self.scheduler.cleanup_finished();
        }
    }

    /// Process a single scheduled batch with optional chunked prefill.
    ///
    /// When `prefill_chunk_size > 0`, prefill is split into chunks. Between
    /// chunks, pending decode steps run to keep latency low for active sequences.
    async fn process_batch(&mut self, batch: &ScheduledBatch) -> Result<()> {
        let chunk_size = self.executor.config().inference.prefill_chunk_size;

        // Process prefill sequences
        for &seq_id in &batch.prefill_sequences {
            if chunk_size > 0 {
                self.process_prefill_chunked(seq_id, batch, chunk_size)
                    .await?;
            } else {
                self.process_prefill(seq_id, batch).await?;
            }
            self.scheduler.prefill_complete(seq_id)?;
        }

        // Process decode sequences (batched single-token generation)
        if !batch.decode_sequences.is_empty() {
            self.process_decode(&batch.decode_sequences, batch).await?;
        }

        Ok(())
    }

    /// Chunked prefill: process prompt in chunks, interleaving decode steps
    async fn process_prefill_chunked(
        &mut self,
        seq_id: SequenceId,
        batch: &ScheduledBatch,
        chunk_size: usize,
    ) -> Result<()> {
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
            .ok_or_else(|| anyhow!("block table for seq {} not found", seq_id))?
            .clone();

        let device = self.executor.device().clone();

        // Process prompt in chunks, starting after cached tokens
        let total = prompt_tokens.len();
        let mut processed = cached_count;
        let mut logits = None;

        while processed < total {
            let end = (processed + chunk_size).min(total);
            let chunk = &prompt_tokens[processed..end];

            // Build slot mapping for this chunk
            let slot_mapping_vec: Vec<i32> = (processed..end)
                .map(|i| {
                    let block_idx = i / block_size;
                    let block_offset = i % block_size;
                    if block_idx < blocks.len() {
                        (blocks[block_idx] as i32) * (block_size as i32) + (block_offset as i32)
                    } else {
                        -1
                    }
                })
                .collect();

            let slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[chunk.len()], &device);

            let bt_vec: Vec<i32> = blocks.iter().map(|&b| b as i32).collect();
            let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], &device);

            let input = self.executor.create_input_tensor(chunk)?;

            let paged_cache = self
                .paged_cache
                .as_ref()
                .ok_or_else(|| anyhow!("paged cache not initialized"))?;

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
                    )
                    .map_err(|e| {
                        anyhow!("chunked prefill failed at offset {}: {}", processed, e)
                    })?,
            );

            processed = end;

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
        let token_history = prompt_tokens.clone();
        let token_gpu =
            self.executor
                .logits_to_token_on_device(&logits, &token_history, &gen_config)?;
        let event = token_gpu
            .record_event()
            .map_err(|e| anyhow!("record event failed: {}", e))?;
        let token_id = Executor::<R>::read_token_id(&token_gpu, event)?;

        let is_eos = self.executor.tokenizer().is_eos(token_id);
        let text = if is_eos {
            String::new()
        } else {
            self.executor
                .tokenizer()
                .decode(&[token_id])
                .unwrap_or_default()
        };

        let finish = if is_eos {
            Some(FinishReason::Eos)
        } else {
            None
        };

        if let Some(tx) = self.scheduler.get_token_sender(seq_id) {
            let token = super::types::GeneratedToken {
                token_id,
                text,
                logprob: None,
                top_logprobs: None,
                finish_reason: finish,
            };
            let _ = tx.send(Ok(token)).await;
        }

        self.scheduler.append_to_history(seq_id, token_id);
        let finished = self.scheduler.append_token(seq_id, token_id)?;
        if finished || finish == Some(FinishReason::Eos) {
            self.scheduler.finish_sequence(seq_id)?;
        }

        Ok(())
    }

    /// Process a single prefill sequence
    async fn process_prefill(&mut self, seq_id: SequenceId, batch: &ScheduledBatch) -> Result<()> {
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
            crate::server::metrics::record_prefix_cache_hit();
        } else {
            crate::server::metrics::record_prefix_cache_miss();
        }

        // Build slot mapping only for uncached tokens
        let slot_mapping_vec: Vec<i32> = (prefill_start..prompt_tokens.len())
            .map(|i| {
                let block_idx = i / block_size;
                let block_offset = i % block_size;
                if block_idx < blocks.len() {
                    (blocks[block_idx] as i32) * (block_size as i32) + (block_offset as i32)
                } else {
                    -1
                }
            })
            .collect();

        let device = self.executor.device();
        let slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[prefill_tokens.len()], device);

        // Build block table tensor
        let bt_vec: Vec<i32> = blocks.iter().map(|&b| b as i32).collect();
        let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], device);

        // Create input tensor for uncached tokens only
        let input = self.executor.create_input_tensor(prefill_tokens)?;

        // Run prefill forward pass starting from cached offset
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
            )
            .map_err(|e| anyhow!("prefill forward failed: {}", e))?;

        // Sample first token
        let token_history = prompt_tokens.clone();
        let token_gpu =
            self.executor
                .logits_to_token_on_device(&logits, &token_history, &gen_config)?;
        let event = token_gpu
            .record_event()
            .map_err(|e| anyhow!("record event failed: {}", e))?;
        let token_id = Executor::<R>::read_token_id(&token_gpu, event)?;

        // Send token to caller
        let text = if self.executor.tokenizer().is_eos(token_id) {
            String::new()
        } else {
            self.executor
                .tokenizer()
                .decode(&[token_id])
                .unwrap_or_default()
        };

        let finish = if self.executor.tokenizer().is_eos(token_id) {
            Some(FinishReason::Eos)
        } else {
            None
        };

        if let Some(tx) = self.scheduler.get_token_sender(seq_id) {
            let token = super::types::GeneratedToken {
                token_id,
                text,
                logprob: None,
                top_logprobs: None,
                finish_reason: finish,
            };
            let _ = tx.send(Ok(token)).await;
        }

        self.scheduler.append_to_history(seq_id, token_id);

        // Track in scheduler
        let finished = self.scheduler.append_token(seq_id, token_id)?;
        if finished || finish == Some(FinishReason::Eos) {
            self.scheduler.finish_sequence(seq_id)?;
        }

        Ok(())
    }

    /// Process decode step for multiple sequences (true batched single-token generation).
    ///
    /// Delegates to `batch_decode::process_decode_batch` which runs a single batched
    /// forward pass for all decode sequences and samples one token per sequence.
    async fn process_decode(
        &mut self,
        decode_seqs: &[SequenceId],
        batch: &ScheduledBatch,
    ) -> Result<()> {
        let paged_cache = self
            .paged_cache
            .as_ref()
            .ok_or_else(|| anyhow!("paged cache not initialized"))?;
        process_decode_batch(
            &self.executor,
            &self.scheduler,
            paged_cache,
            decode_seqs,
            batch,
        )
        .await
    }
}
