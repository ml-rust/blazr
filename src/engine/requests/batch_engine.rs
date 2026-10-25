//! Continuous batching engine
//!
//! Runs the scheduling loop: pulls batches from RequestScheduler,
//! executes batched forward passes, and streams tokens back to callers.
//!
//! Each sequence samples through its own sampler state, so grammar,
//! mirostat and the penalties apply as in the single-sequence loop. The
//! prefill lives in `batch_prefill`, the loop in `batch_run`, and every
//! token leaves through `batch_emit`.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::BlockAllocator;
use boostr::inference::scheduler::{ScheduledBatch, SequenceId};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, TypeConversionOps, UnaryOps,
};

use super::batch_decode::process_decode_batch;
use super::cancel_marks::CancelPhase;
use super::request_cancel::end_cancelled;
use super::request_scheduler::RequestScheduler;
use super::seq_sampler::BatchSeqs;
use crate::engine::executor::Executor;

/// Continuous batching engine that drives the scheduling loop
pub struct BatchEngine<R: Runtime<DType = DType>> {
    pub(super) executor: Arc<Executor<R>>,
    pub(super) scheduler: Arc<RequestScheduler>,
    /// Shared paged KV cache (one large pool for all sequences)
    pub(super) paged_cache: Option<LayeredPagedKvCache<R>>,
    /// Prefill chunk size for this engine. `None` until set explicitly by
    /// `with_prefill_chunk_size`, in which case `process_batch` falls back
    /// to `inference.prefill_chunk_size` from the executor's config.
    prefill_chunk_size: Option<usize>,
    /// Sampling state of every sequence past its first prefill.
    pub(super) seqs: BatchSeqs<R>,
    /// Runs after each prefill chunk with the tokens prefilled so far, so a
    /// test can act between chunks.
    #[cfg(test)]
    pub(crate) after_chunk: Option<Box<dyn FnMut(usize) + Send>>,
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
        let model = executor.model()?;
        if model.needs_gdn_state() {
            return Err(crate::engine::qwen35_unsupported(
                "continuous batching",
                None,
                Some("set max_batch_size = 1, or use CUDA lanes with paged_attention off"),
            ));
        }
        let config = executor.config();
        let paged_cache = if config.inference.paged_attention {
            let num_layers = model.num_layers();
            let num_kv_heads = model.num_kv_heads().unwrap_or(8);
            let head_dim = model.head_dim().unwrap_or(64);
            let block_size = config.inference.block_size;
            let kv_dtype = config.kv_cache_dtype()?;
            let total_blocks = Self::kv_pool_blocks(&executor)?;

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

    /// Override the prefill chunk size for this engine, taking precedence
    /// over `inference.prefill_chunk_size` from the executor's config. A
    /// value of `0` disables chunking.
    pub fn with_prefill_chunk_size(mut self, chunk_size: usize) -> Self {
        self.prefill_chunk_size = Some(chunk_size);
        self
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
    pub(super) async fn process_batch(&mut self, batch: &ScheduledBatch) -> Result<()> {
        let chunk_size = self.chunk_size();

        // Preempted sequences lost their KV blocks. Their next decode refills them.
        self.seqs.mark_stale(&batch.preempted_sequences);

        // Process prefill sequences
        for &seq_id in &batch.prefill_sequences {
            // A sequence cancelled after admission skips its prefill forward.
            if !self.scheduler.is_live(seq_id) {
                end_cancelled(
                    &self.scheduler,
                    &mut self.seqs,
                    seq_id,
                    CancelPhase::Prefill,
                )?;
                continue;
            }
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

    /// Process decode step for multiple sequences (true batched single-token generation).
    ///
    /// Delegates to `batch_decode::process_decode_batch` which runs a single batched
    /// forward pass for all decode sequences and samples one token per sequence.
    pub(super) async fn process_decode(
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
    use boostr::CpuRuntime;

    use crate::config::{Constraint, GenerationConfig};
    use crate::engine::requests::batch_test_rig::{
        alone, drain, free_blocks, greedy, ids, received, rig, rig_on, Rig, POOL, SEED, TINY_CTX,
    };
    use crate::engine::test_support::{
        check_two_field_instance, full_vocab_executor, seeded_executor, two_field_bias,
        two_field_schema, PagedPool, FULL_VOCAB_CTX,
    };
    use crate::tokenizer::token_piece_bytes;

    /// One digit from each pair, so the model picks within the grammar.
    const THREE_DIGITS: &str = r#"root ::= ("1" | "2") ("3" | "4") ("5" | "6")"#;

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

    /// With no engine override the chunk size is `inference.prefill_chunk_size`.
    /// The handle drops after the first of five chunks, so no other runs.
    #[tokio::test]
    async fn the_config_chunk_size_stops_a_cancelled_prefill() -> Result<()> {
        let mut executor = seeded_executor(SEED, Some(POOL))?;
        executor.config.inference.prefill_chunk_size = 2;
        let Rig {
            executor,
            scheduler,
            ..
        } = rig_on(executor, POOL, TINY_CTX)?;
        let engine = BatchEngine::new(Arc::clone(&executor), Arc::clone(&scheduler))?;
        let mut rig = Rig {
            executor,
            scheduler,
            engine,
        };
        let start = free_blocks(&rig)?;
        let handle = rig.scheduler.submit((1..=9).collect(), greedy(4))?;
        let seq_id = handle.seq_id;
        let mut slot = Some(handle);
        rig.engine.after_chunk = Some(Box::new(move |_: usize| drop(slot.take())));

        rig.engine.tick().await;
        rig.engine.after_chunk = None;

        assert_eq!(rig.engine.seqs.forwards, 1, "one chunk ran");
        assert!(!rig.scheduler.is_live(seq_id));
        assert!(!rig.engine.has_live_work());
        assert_eq!(free_blocks(&rig)?, start, "the prompt blocks are free");
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
