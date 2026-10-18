//! Continuous batching engine
//!
//! Runs the scheduling loop: pulls batches from RequestScheduler,
//! executes batched forward passes, and streams tokens back to callers.
//!
//! Each sequence samples through its own sampler state, so grammar,
//! mirostat and the penalties apply as in the single-sequence loop.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::BlockAllocator;
use boostr::inference::scheduler::{ScheduledBatch, SequenceId};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use super::batch_decode::process_decode_batch;
use super::kv_recompute::{block_ids, block_slots};
use super::request_scheduler::RequestScheduler;
use super::seq_sampler::BatchSeqs;
use crate::config::GenerationConfig;
use crate::engine::executor::Executor;
use crate::engine::types::FinishReason;

/// Continuous batching engine that drives the scheduling loop
pub struct BatchEngine<R: Runtime<DType = DType>> {
    executor: Arc<Executor<R>>,
    scheduler: Arc<RequestScheduler>,
    /// Shared paged KV cache (one large pool for all sequences)
    paged_cache: Option<LayeredPagedKvCache<R>>,
    /// Prefill chunk size for this engine. `None` until set explicitly by
    /// `with_prefill_chunk_size`, in which case `process_batch` falls back
    /// to `inference.prefill_chunk_size` from the executor's config.
    prefill_chunk_size: Option<usize>,
    /// Sampling state of every sequence past its first prefill.
    seqs: BatchSeqs<R>,
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
        + ModelClient<R>
        + RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    /// Create a new batch engine
    pub fn new(executor: Arc<Executor<R>>, scheduler: Arc<RequestScheduler>) -> Result<Self> {
        if executor.model().needs_gdn_state() {
            return Err(crate::engine::qwen35_unsupported(
                "continuous batching",
                None,
                Some("disable paged attention or set max_batch_size = 1"),
            ));
        }
        let config = executor.config();
        let paged_cache = if config.inference.paged_attention {
            let num_layers = executor.model().num_layers();
            let num_kv_heads = executor.model().num_kv_heads().unwrap_or(8);
            let head_dim = executor.model().head_dim().unwrap_or(64);
            let block_size = config.inference.block_size;
            let kv_dtype = config.kv_cache_dtype()?;

            // Total blocks in the shared pool
            let total_blocks = if config.inference.kv_pool_blocks > 0 {
                config.inference.kv_pool_blocks
            } else if let Some(shared) = executor.shared_allocator() {
                shared
                    .lock()
                    .map_err(|e| anyhow!("Block allocator lock poisoned: {e}"))?
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
            )?)
        } else {
            None
        };

        let seqs = BatchSeqs::new(executor.vocab_size(), executor.device())?;
        Ok(Self {
            executor,
            scheduler,
            paged_cache,
            prefill_chunk_size: None,
            seqs,
        })
    }

    /// Override the prefill chunk size for this engine, taking precedence
    /// over `inference.prefill_chunk_size` from the executor's config. A
    /// value of `0` disables chunking.
    pub fn with_prefill_chunk_size(mut self, chunk_size: usize) -> Self {
        self.prefill_chunk_size = Some(chunk_size);
        self
    }

    /// Run the continuous batching loop. Spawn it as a background task.
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

            self.tick().await;
        }
    }

    /// Schedule and process one batch, then drop the state of every
    /// sequence that ended.
    async fn tick(&mut self) {
        match self.scheduler.schedule_batch() {
            Ok(Some(batch)) => {
                if let Err(e) = self.process_batch(&batch).await {
                    tracing::error!("BatchEngine: batch processing error: {:#}", e);
                    // End every sequence in the failed batch and tell its caller.
                    for seq_id in batch.all_sequences() {
                        self.seqs.end_seq(seq_id);
                        if let Some(tx) = self.scheduler.get_token_sender(seq_id) {
                            let err = anyhow!(
                                "the batch this request ran in failed: {e:#}. Retry the request."
                            );
                            let _ = tx.send(Err(err)).await;
                        }
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

        // Periodic cleanup. A cancelled request leaves the registry, so its
        // sampler state ends here.
        self.scheduler.cleanup_finished();
        let scheduler = &self.scheduler;
        self.seqs.end_gone(|seq_id| scheduler.is_live(seq_id));
    }

    /// The prefill chunk size in force: the engine's own, else the config's.
    fn chunk_size(&self) -> usize {
        self.prefill_chunk_size
            .unwrap_or(self.executor.config().inference.prefill_chunk_size)
    }

    /// Process a single scheduled batch with optional chunked prefill.
    ///
    /// When `prefill_chunk_size > 0`, prefill is split into chunks. Between
    /// chunks, pending decode steps run to keep latency low for active sequences.
    async fn process_batch(&mut self, batch: &ScheduledBatch) -> Result<()> {
        let chunk_size = self.chunk_size();

        // Preempted sequences lost their KV blocks. Their next decode refills them.
        self.seqs.mark_stale(&batch.preempted_sequences);

        // Process prefill sequences
        for &seq_id in &batch.prefill_sequences {
            let ended = if chunk_size > 0 {
                self.process_prefill_chunked(seq_id, batch, chunk_size)
                    .await?
            } else {
                self.process_prefill(seq_id, batch).await?
            };
            // A sequence that ended on its first token is no longer running.
            if !ended {
                self.scheduler.prefill_complete(seq_id)?;
            }
        }

        // Process decode sequences (batched single-token generation)
        if !batch.decode_sequences.is_empty() {
            self.process_decode(&batch.decode_sequences, batch).await?;
        }

        Ok(())
    }

    /// Chunked prefill: process prompt in chunks, interleaving decode steps.
    ///
    /// Returns whether the sequence ended on its first token.
    async fn process_prefill_chunked(
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

            let slot_mapping_vec = block_slots(&blocks, block_size, processed..end);
            let slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[chunk.len()], &device)?;

            let bt_vec = block_ids(&blocks);
            let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], &device)?;

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
                        None,
                        None,
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
        let token_id = self.seqs.sample(&self.executor, seq_id, logits)?;

        let is_eos = self.executor.tokenizer().is_eos(token_id);
        let finish = if is_eos {
            Some(FinishReason::Eos)
        } else {
            None
        };

        let finished = self.scheduler.append_token(seq_id, token_id)?;
        let text = self.scheduler.token_text(
            seq_id,
            self.executor.tokenizer(),
            token_id,
            is_eos,
            finished,
        );

        if let Some(tx) = self.scheduler.get_token_sender(seq_id) {
            let token = crate::engine::types::GeneratedToken {
                token_id,
                text,
                logprob: None,
                top_logprobs: None,
                finish_reason: finish,
            };
            let _ = tx.send(Ok(token)).await;
        }

        self.scheduler.append_to_history(seq_id, token_id);
        let ended = finished || is_eos;
        if ended {
            self.seqs.end_seq(seq_id);
            self.scheduler.finish_sequence(seq_id)?;
        }
        Ok(ended)
    }

    /// Process a single prefill sequence.
    ///
    /// Returns whether the sequence ended on its first token.
    async fn process_prefill(
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

    /// Process decode step for multiple sequences (true batched single-token generation).
    ///
    /// Delegates to `batch_decode::process_decode_batch` which runs a single batched
    /// forward pass for all decode sequences and samples one token per sequence.
    async fn process_decode(
        &mut self,
        decode_seqs: &[SequenceId],
        batch: &ScheduledBatch,
    ) -> Result<()> {
        let chunk_size = self.chunk_size();
        let paged_cache = self
            .paged_cache
            .as_ref()
            .ok_or_else(|| anyhow!("paged cache not initialized"))?;
        process_decode_batch(
            &self.executor,
            &self.scheduler,
            paged_cache,
            &mut self.seqs,
            decode_seqs,
            batch,
            chunk_size,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::inference::memory::CpuBlockAllocator;
    use boostr::inference::scheduler::SchedulerConfig;
    use boostr::CpuRuntime;
    use futures::StreamExt;

    use crate::config::Constraint;
    use crate::engine::requests::request_registry::RequestHandle;
    use crate::engine::test_support::{
        check_two_field_instance, full_vocab_executor, seeded_executor, two_field_bias,
        two_field_schema, PagedPool, FULL_VOCAB_CTX,
    };
    use crate::engine::types::GeneratedToken;
    use crate::tokenizer::token_piece_bytes;

    const SEED: u64 = 0x5eed;
    /// Ticks a test allows before it calls the engine stuck.
    const TICK_LIMIT: usize = 64;
    /// A pool large enough that no sequence is preempted.
    const POOL: PagedPool = PagedPool {
        block_size: 4,
        blocks: 16,
    };
    /// One digit from each pair, so the model picks within the grammar.
    const THREE_DIGITS: &str = r#"root ::= ("1" | "2") ("3" | "4") ("5" | "6")"#;

    struct Rig {
        executor: Arc<Executor<CpuRuntime>>,
        scheduler: Arc<RequestScheduler>,
        engine: BatchEngine<CpuRuntime>,
    }

    /// A batching engine over the seeded tiny llama, admitting two sequences.
    fn rig(pool: PagedPool) -> Result<Rig> {
        rig_on(seeded_executor(SEED, Some(pool))?, pool, 16)
    }

    /// A batching engine over `executor`, admitting two sequences of up to
    /// `max_seq_len` tokens.
    fn rig_on(executor: Executor<CpuRuntime>, pool: PagedPool, max_seq_len: usize) -> Result<Rig> {
        let executor = Arc::new(executor);
        let config = SchedulerConfig {
            max_batch_size: 2,
            max_batch_tokens: 4096,
            max_seq_len,
            block_size: pool.block_size,
            ..Default::default()
        };
        let allocator = CpuBlockAllocator::new(pool.blocks, pool.block_size);
        let scheduler = RequestScheduler::new(allocator, config);
        let engine = BatchEngine::new(Arc::clone(&executor), Arc::clone(&scheduler))?
            .with_prefill_chunk_size(0);
        Ok(Rig {
            executor,
            scheduler,
            engine,
        })
    }

    fn greedy(max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_tokens,
            repeat_penalty: 1.0,
            ..GenerationConfig::greedy()
        }
    }

    fn grammar(rule: &str, max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            constraint: Some(Constraint::Gbnf(rule.to_string())),
            ..greedy(max_tokens)
        }
    }

    fn mirostat(seed: u64, max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_tokens,
            mirostat_mode: 2,
            seed: Some(seed),
            repeat_penalty: 1.0,
            ..GenerationConfig::default()
        }
    }

    /// The ids `prompt` produces alone through the single-sequence loop.
    async fn alone(
        executor: &Executor<CpuRuntime>,
        prompt: &[u32],
        cfg: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let mut stream = std::pin::pin!(executor.generate(prompt, cfg));
        let mut ids = Vec::new();
        while let Some(item) = stream.next().await {
            ids.push(item?.token_id);
        }
        Ok(ids)
    }

    /// Tick until the scheduler holds no work.
    async fn drain(rig: &mut Rig) -> Result<()> {
        for _ in 0..TICK_LIMIT {
            if !rig.scheduler.has_work() {
                return Ok(());
            }
            rig.engine.tick().await;
        }
        Err(anyhow!(
            "the batch engine still had work after {TICK_LIMIT} ticks"
        ))
    }

    /// Every item `handle` received so far.
    fn received(handle: &mut RequestHandle) -> Vec<Result<GeneratedToken>> {
        let mut items = Vec::new();
        while let Ok(item) = handle.token_rx.try_recv() {
            items.push(item);
        }
        items
    }

    /// The ids `handle` received so far. A streamed error fails the call.
    fn ids(handle: &mut RequestHandle) -> Result<Vec<u32>> {
        received(handle)
            .into_iter()
            .map(|item| item.map(|t| t.token_id))
            .collect()
    }

    /// The bytes `ids` decode to.
    fn text(executor: &Executor<CpuRuntime>, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| token_piece_bytes(executor.tokenizer(), id))
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Whether `text` is one digit from each pair of `pairs`.
    fn one_of_each(text: &str, pairs: &[[char; 2]]) -> bool {
        text.chars().count() == pairs.len()
            && text.chars().zip(pairs).all(|(c, pair)| pair.contains(&c))
    }

    #[tokio::test]
    async fn a_grammar_request_matches_its_unbatched_run() -> Result<()> {
        let mut rig = rig(POOL)?;
        let prompt = [1, 2, 3];
        let cfg = grammar(THREE_DIGITS, 3);
        let expected = alone(&rig.executor, &prompt, &cfg).await?;

        let mut handle = rig.scheduler.submit(prompt.to_vec(), cfg)?;
        drain(&mut rig).await?;
        let got = ids(&mut handle)?;

        assert_eq!(got, expected);
        let produced = text(&rig.executor, &got);
        assert!(
            one_of_each(&produced, &[['1', '2'], ['3', '4'], ['5', '6']]),
            "batched output {produced:?} breaks the grammar"
        );
        assert_eq!(
            rig.engine.seqs.len(),
            0,
            "a finished sequence keeps its state"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_grammar_row_and_a_greedy_row_keep_their_own_sampling() -> Result<()> {
        let mut rig = rig(POOL)?;
        let (constrained, plain) = ([1, 2, 3], [7, 8, 9, 10]);
        let (constrained_cfg, plain_cfg) = (grammar(THREE_DIGITS, 3), greedy(3));
        let constrained_alone = alone(&rig.executor, &constrained, &constrained_cfg).await?;
        let plain_alone = alone(&rig.executor, &plain, &plain_cfg).await?;

        let mut a = rig
            .scheduler
            .submit(constrained.to_vec(), constrained_cfg)?;
        let mut b = rig.scheduler.submit(plain.to_vec(), plain_cfg)?;
        rig.engine.tick().await;
        assert_eq!(
            rig.engine.seqs.len(),
            2,
            "both sequences prefill in one batch"
        );
        drain(&mut rig).await?;

        assert_eq!(ids(&mut a)?, constrained_alone);
        assert_eq!(ids(&mut b)?, plain_alone);
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn mirostat_rows_match_their_unbatched_runs() -> Result<()> {
        let mut rig = rig(POOL)?;
        let prompt = [1, 2, 3];
        let (first_cfg, second_cfg) = (mirostat(7, 4), mirostat(11, 4));
        let first_alone = alone(&rig.executor, &prompt, &first_cfg).await?;
        let second_alone = alone(&rig.executor, &prompt, &second_cfg).await?;

        let mut first = rig.scheduler.submit(prompt.to_vec(), first_cfg)?;
        let mut second = rig.scheduler.submit(prompt.to_vec(), second_cfg)?;
        drain(&mut rig).await?;

        assert_eq!(ids(&mut first)?, first_alone);
        assert_eq!(ids(&mut second)?, second_alone);
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    /// The batched rows sample into a shared scratch. The single-sequence
    /// loop allocates. A seeded draw must not tell them apart.
    #[tokio::test]
    async fn seeded_sampling_rows_match_their_unbatched_runs() -> Result<()> {
        let mut rig = rig(POOL)?;
        let prompt = [2, 4, 6];
        let seeded = |seed| GenerationConfig {
            max_tokens: 4,
            temperature: 1.0,
            seed: Some(seed),
            frequency_penalty: 0.5,
            ..GenerationConfig::default()
        };
        let (first_cfg, second_cfg) = (seeded(3), seeded(9));
        let first_alone = alone(&rig.executor, &prompt, &first_cfg).await?;
        let second_alone = alone(&rig.executor, &prompt, &second_cfg).await?;

        let mut first = rig.scheduler.submit(prompt.to_vec(), first_cfg)?;
        let mut second = rig.scheduler.submit(prompt.to_vec(), second_cfg)?;
        drain(&mut rig).await?;

        assert_eq!(ids(&mut first)?, first_alone);
        assert_eq!(ids(&mut second)?, second_alone);
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_batch_ends_the_state_and_reports_the_error() -> Result<()> {
        let mut rig = rig(POOL)?;
        let mut handle = rig.scheduler.submit(vec![1, 2, 3], greedy(4))?;
        rig.engine.tick().await;
        assert_eq!(rig.engine.seqs.len(), 1);

        // The next decode step fails: the engine has no KV cache to run on.
        rig.engine.paged_cache = None;
        rig.engine.tick().await;

        assert_eq!(rig.engine.seqs.len(), 0);
        assert!(!rig.scheduler.is_live(handle.seq_id));
        let items = received(&mut handle);
        assert!(
            matches!(items.first(), Some(Ok(_))),
            "the prefill token arrives"
        );
        assert!(
            items.last().is_some_and(|item| item.is_err()),
            "the caller hears the batch error"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_cancelled_request_ends_its_state() -> Result<()> {
        let mut rig = rig(POOL)?;
        let handle = rig.scheduler.submit(vec![1, 2, 3], greedy(4))?;
        rig.engine.tick().await;
        assert_eq!(rig.engine.seqs.len(), 1);

        rig.scheduler.cancel(handle.seq_id)?;
        rig.engine.tick().await;
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    /// Three blocks of four: after both prefills the pool is empty, so the
    /// second sequence is preempted and resumed on fresh blocks. The first
    /// finishes in that step and frees the blocks the second grows into.
    #[tokio::test]
    async fn a_preempted_grammar_sequence_resumes_with_its_state_and_kv() -> Result<()> {
        let mut rig = rig(PagedPool {
            block_size: 4,
            blocks: 3,
        })?;
        let rule = r#"root ::= ("1" | "2") ("3" | "4") ("5" | "6") ("7" | "8")"#;
        let (short, long) = ([1, 2, 3], [4, 5, 6]);
        let (short_cfg, long_cfg) = (greedy(2), grammar(rule, 4));
        let short_alone = alone(&rig.executor, &short, &short_cfg).await?;
        let long_alone = alone(&rig.executor, &long, &long_cfg).await?;

        let mut a = rig.scheduler.submit(short.to_vec(), short_cfg)?;
        let mut b = rig.scheduler.submit(long.to_vec(), long_cfg)?;
        drain(&mut rig).await?;

        let preemptions = rig
            .scheduler
            .stats()
            .map_or(0, |stats| stats.total_preemptions);
        assert!(preemptions >= 1, "the pool must force a preemption");
        assert!(
            rig.engine.seqs.recomputes >= 1,
            "the resumed sequence refills its KV"
        );

        assert_eq!(ids(&mut a)?, short_alone);
        let long_ids = ids(&mut b)?;
        assert_eq!(long_ids, long_alone);
        let produced = text(&rig.executor, &long_ids);
        assert!(
            one_of_each(&produced, &[['1', '2'], ['3', '4'], ['5', '6'], ['7', '8']]),
            "the grammar state did not survive preemption: {produced:?}"
        );
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    /// A JSON-schema row produces its unbatched ids, a valid instance, and
    /// ends by EOS once the object closes.
    #[tokio::test]
    async fn a_json_schema_row_matches_its_unbatched_run_and_ends_by_eos() -> Result<()> {
        let mut rig = rig_on(full_vocab_executor(Some(POOL))?, POOL, FULL_VOCAB_CTX)?;
        let cfg = GenerationConfig {
            constraint: Some(Constraint::JsonSchema(two_field_schema())),
            logit_bias: two_field_bias(&rig.executor)?,
            ..greedy(40)
        };
        let prompt = [1];
        let expected = alone(&rig.executor, &prompt, &cfg).await?;

        let mut handle = rig.scheduler.submit(prompt.to_vec(), cfg)?;
        drain(&mut rig).await?;
        let got = ids(&mut handle)?;

        assert_eq!(got, expected);
        let eos = rig.executor.tokenizer().eos_token_id();
        assert_eq!(got.last().copied(), eos, "the run ends by EOS");
        let body = &got[..got.len().saturating_sub(1)];
        check_two_field_instance(&text(&rig.executor, body))?;
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }
}
