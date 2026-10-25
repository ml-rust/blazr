//! Request scheduler — bridges HTTP requests to boostr's SequenceScheduler
//!
//! This module wraps boostr's continuous batching SequenceScheduler and provides:
//! - Async request submission from HTTP handlers
//! - Token streaming back to callers via channels
//! - A scheduling loop that produces batches for the executor
//!
//! The per-request metadata and channels live in [`RequestRegistry`], shared
//! with the other schedulers. Cancellation lives in `request_cancel`, its
//! marks in `cancel_marks`.

use std::sync::{Arc, Mutex, Weak};

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc, oneshot};

use boostr::inference::memory::{BlockAllocator, CpuBlockAllocator};
use boostr::inference::prefix_cache::PrefixCache;
use boostr::inference::scheduler::{
    ScheduledBatch, SchedulerConfig, SequenceId, SequenceRequest, SequenceScheduler, SequenceState,
};

use super::cancel_marks::PendingCancels;
use super::queue_depth::QueueDepth;
use super::request_registry::{CancelRequest, GenerationScheduler, RequestHandle, RequestRegistry};
use super::slo_policy::SloBatchPolicy;
use crate::config::GenerationConfig;
use crate::engine::types::{FinishReason, GeneratedToken};
use crate::engine::{token_budget, StopCut};

/// Wakeups the scheduling loop can miss without blocking a submitting handler.
const NOTIFY_CHANNEL_CAPACITY: usize = 64;

/// A pending generation request submitted via the HTTP API
pub struct InferenceRequest {
    /// Unique sequence ID
    pub seq_id: SequenceId,
    /// Tokenized prompt
    pub prompt_tokens: Vec<u32>,
    /// Generation parameters
    pub gen_config: GenerationConfig,
    /// Channel to stream generated tokens back to the caller
    pub token_tx: mpsc::Sender<Result<GeneratedToken>>,
    /// Signal when the request is fully complete
    pub _done_tx: Option<oneshot::Sender<()>>,
}

/// The request scheduler wraps boostr's SequenceScheduler and manages
/// the lifecycle of inference requests in a continuous batching loop.
pub struct RequestScheduler {
    /// Inner boostr scheduler (protected by mutex for multi-threaded access)
    pub(super) inner: Mutex<SequenceScheduler<CpuBlockAllocator>>,
    /// Per-sequence metadata (gen config, token channels)
    pub(super) registry: RequestRegistry,
    /// Running requests cancelled since the engine last dropped them.
    pub(super) cancels: PendingCancels,
    /// This scheduler, for the handles it hands out to cancel through.
    me: Weak<RequestScheduler>,
    /// Channel to notify the scheduling loop of new work
    pub(super) notify_tx: mpsc::Sender<()>,
    /// Receiver end (held by the scheduling loop)
    notify_rx: Mutex<Option<mpsc::Receiver<()>>>,
    /// Waiting plus preempted sequences, published as `blazr_queue_depth`.
    pub(super) queue_depth: QueueDepth,
    /// Holds batch growth while a P95 latency target is breached.
    slo_policy: Option<Arc<SloBatchPolicy>>,
    /// `SchedulerConfig::max_batch_size`, the cap the policy sizes against.
    max_batch_size: usize,
    /// The most tokens one sequence holds: the least of
    /// `SchedulerConfig::max_seq_len`, `max_batch_tokens` and the token
    /// slots of the whole block pool.
    context_len: usize,
}

impl RequestScheduler {
    /// Create a new request scheduler with the given allocator and config
    pub fn new(allocator: CpuBlockAllocator, config: SchedulerConfig) -> Arc<Self> {
        Self::build(allocator, config, None, None)
    }

    /// Create a request scheduler with prefix caching enabled.
    pub fn with_prefix_cache(
        allocator: CpuBlockAllocator,
        config: SchedulerConfig,
        prefix_cache: PrefixCache<CpuBlockAllocator>,
    ) -> Arc<Self> {
        Self::build(allocator, config, Some(prefix_cache), None)
    }

    /// Create a request scheduler with an optional prefix cache and an
    /// optional SLO batch policy.
    pub fn build(
        allocator: CpuBlockAllocator,
        config: SchedulerConfig,
        prefix_cache: Option<PrefixCache<CpuBlockAllocator>>,
        slo_policy: Option<Arc<SloBatchPolicy>>,
    ) -> Arc<Self> {
        let (notify_tx, notify_rx) = mpsc::channel(NOTIFY_CHANNEL_CAPACITY);
        let max_batch_size = config.max_batch_size;
        // A sequence longer than the pool or the batch token budget never
        // schedules again, so a prompt must fit both.
        let pool_tokens = allocator.total_blocks().saturating_mul(config.block_size);
        let context_len = config
            .max_seq_len
            .min(pool_tokens)
            .min(config.max_batch_tokens);
        let mut scheduler = SequenceScheduler::new(allocator, config);
        if let Some(cache) = prefix_cache {
            scheduler = scheduler.with_prefix_cache(cache);
        }
        Arc::new_cyclic(|me| Self {
            inner: Mutex::new(scheduler),
            registry: RequestRegistry::new(),
            cancels: PendingCancels::new(),
            me: me.clone(),
            notify_tx,
            notify_rx: Mutex::new(Some(notify_rx)),
            queue_depth: QueueDepth::new(),
            slo_policy,
            max_batch_size,
            context_len,
        })
    }

    /// Publish the waiting count of `sched` as this scheduler's queue depth.
    pub(super) fn publish_depth(&self, sched: &SequenceScheduler<CpuBlockAllocator>) {
        let stats = sched.stats();
        self.queue_depth
            .set(stats.waiting_count + stats.preempted_count);
    }

    /// Requests waiting for a batch slot or for token-budget admission.
    pub fn queue_depth(&self) -> usize {
        self.queue_depth.get()
    }

    /// Submit a new inference request. Returns a handle for receiving tokens.
    ///
    /// `max_tokens` is capped by [`token_budget`] over this scheduler's
    /// context, as the single-sequence loop caps it. A request with no
    /// budget left, or an empty prompt, ends at once with no tokens.
    ///
    /// # Errors
    ///
    /// An error when the prompt is longer than the context, or when the
    /// request cannot be queued.
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        mut gen_config: GenerationConfig,
    ) -> Result<RequestHandle> {
        if prompt_tokens.len() > self.context_len {
            return Err(anyhow!(
                "the prompt of {} tokens exceeds the {}-token context this scheduler serves; \
                 shorten the prompt",
                prompt_tokens.len(),
                self.context_len
            ));
        }
        gen_config.max_tokens =
            token_budget(gen_config.max_tokens, self.context_len, prompt_tokens.len());
        let seq_id = self.registry.next_id();
        if gen_config.max_tokens == 0 || prompt_tokens.is_empty() {
            // Nothing to decode: the caller's stream closes with no tokens.
            let handle = self.registry.insert(seq_id, prompt_tokens, gen_config)?;
            self.registry.finish(seq_id)?;
            return Ok(handle);
        }
        let request = SequenceRequest::new(seq_id, prompt_tokens.clone())
            .with_max_tokens(gen_config.max_tokens);

        // The registry entry comes first: the engine reads the prompt from it
        // as soon as the sequence can be scheduled.
        let handle = self.registry.insert(seq_id, prompt_tokens, gen_config)?;
        let queued = self
            .inner
            .lock()
            .map_err(|e| anyhow!("scheduler lock: {e}"))
            .and_then(|mut sched| {
                sched.add_request(request).map_err(|e| anyhow!("{e}"))?;
                self.publish_depth(&sched);
                Ok(())
            });
        if let Err(e) = queued {
            self.registry.remove(seq_id)?;
            return Err(e);
        }

        // Wake the scheduling loop
        let _ = self.notify_tx.try_send(());

        let canceller: Weak<dyn CancelRequest> = self.me.clone();
        Ok(handle.cancel_on_drop(canceller))
    }

    /// Schedule the next batch of work. Called by the scheduling loop.
    /// Returns None if no work is available.
    ///
    /// With an SLO policy, a P95 breach stops the batch from growing past
    /// the sequences already running.
    ///
    /// Every cancelled request leaves the scheduler first, under the same
    /// lock, so a cancelled request is never admitted or resumed.
    pub fn schedule_batch(&self) -> Result<Option<ScheduledBatch>> {
        let mut sched = self
            .inner
            .lock()
            .map_err(|e| anyhow!("scheduler lock: {e}"))?;
        let aborted = self.abort_marked(&mut sched)?;
        let limit = match &self.slo_policy {
            Some(policy) => policy.batch_limit(sched.stats().running_count, self.max_batch_size),
            None => self.max_batch_size,
        };
        let batch = sched.schedule_with_limit(limit).map_err(|e| anyhow!("{e}"));
        self.publish_depth(&sched);
        drop(sched);
        self.forget_all(aborted)?;
        batch
    }

    /// Notify the scheduler that prefill completed for a sequence
    pub fn prefill_complete(&self, seq_id: SequenceId) -> Result<()> {
        let mut sched = self
            .inner
            .lock()
            .map_err(|e| anyhow!("scheduler lock: {e}"))?;
        sched.prefill_complete(seq_id).map_err(|e| anyhow!("{e}"))
    }

    /// Append a generated token to a sequence. Returns true if sequence is finished.
    pub fn append_token(&self, seq_id: SequenceId, token: u32) -> Result<bool> {
        let mut sched = self
            .inner
            .lock()
            .map_err(|e| anyhow!("scheduler lock: {e}"))?;
        sched
            .append_token(seq_id, token)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Mark a sequence as finished and clean up
    pub fn finish_sequence(&self, seq_id: SequenceId) -> Result<()> {
        {
            let mut sched = self
                .inner
                .lock()
                .map_err(|e| anyhow!("scheduler lock: {e}"))?;
            // Only finish if still tracked (can already be finished via append_token)
            if sched.has_sequence(seq_id) {
                if let Some(state) = sched.get_sequence_state(seq_id) {
                    if state != SequenceState::Finished {
                        sched.finish_sequence(seq_id).map_err(|e| anyhow!("{e}"))?;
                    }
                }
            }
        }
        // Signal completion and clean up metadata
        self.registry.finish(seq_id)
    }

    /// Get the generation config for a sequence
    pub fn get_gen_config(&self, seq_id: SequenceId) -> Option<GenerationConfig> {
        self.registry.gen_config(seq_id)
    }

    /// Get the token sender for a sequence (to stream tokens back to caller)
    pub fn get_token_sender(
        &self,
        seq_id: SequenceId,
    ) -> Option<mpsc::Sender<Result<GeneratedToken>>> {
        self.registry.token_sender(seq_id)
    }

    /// The text `token_id` adds to `seq_id`'s response, cut at its stops.
    /// See [`RequestRegistry::token_text`].
    pub fn token_text(
        &self,
        seq_id: SequenceId,
        tokenizer: &crate::tokenizer::AnyTokenizer,
        token_id: u32,
        finish: Option<FinishReason>,
    ) -> StopCut {
        self.registry
            .token_text(seq_id, tokenizer, token_id, finish)
    }

    /// Get token history for a sequence (prompt + generated so far)
    pub fn get_token_history(&self, seq_id: SequenceId) -> Option<Vec<u32>> {
        self.registry.token_history(seq_id)
    }

    /// Append to token history for a sequence
    pub fn append_to_history(&self, seq_id: SequenceId, token: u32) {
        self.registry.push_history(seq_id, token);
    }

    /// Get the prompt tokens for a sequence
    pub fn get_prompt_tokens(&self, seq_id: SequenceId) -> Option<Vec<u32>> {
        self.registry.prompt_tokens(seq_id)
    }

    /// Whether `seq_id`'s request is still carried. A finished or cancelled
    /// request is not, even before the engine drops it.
    pub fn is_live(&self, seq_id: SequenceId) -> bool {
        self.registry.contains(seq_id) && !self.cancels.contains(seq_id)
    }

    /// Get block table for a sequence from the inner scheduler
    pub fn get_block_table(&self, seq_id: SequenceId) -> Option<Vec<u32>> {
        let sched = self.inner.lock().ok()?;
        sched.get_block_table(seq_id).map(|bt| bt.blocks.to_vec())
    }

    /// Check if there's any work to do
    pub fn has_work(&self) -> bool {
        self.inner.lock().map(|s| s.has_work()).unwrap_or(false)
    }

    /// Take the notify receiver (only callable once, by the scheduling loop)
    pub fn take_notify_rx(&self) -> Option<mpsc::Receiver<()>> {
        self.notify_rx.lock().ok()?.take()
    }

    /// Get scheduler stats
    pub fn stats(&self) -> Option<boostr::inference::SchedulerStats> {
        self.inner.lock().ok().map(|s| s.stats())
    }

    /// Get block allocator stats (for KV cache utilization metrics)
    pub fn block_stats(&self) -> Option<boostr::inference::memory::BlockAllocatorStats> {
        self.inner.lock().ok().map(|s| s.block_stats())
    }

    /// Clean up finished sequences
    pub fn cleanup_finished(&self) {
        if let Ok(mut sched) = self.inner.lock() {
            sched.cleanup_finished();
        }
    }

    /// Get notify sender (for waking the loop from outside)
    pub fn notify_sender(&self) -> mpsc::Sender<()> {
        self.notify_tx.clone()
    }
}

impl GenerationScheduler for RequestScheduler {
    fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        gen_config: GenerationConfig,
    ) -> Result<RequestHandle> {
        RequestScheduler::submit(self, prompt_tokens, gen_config)
    }

    fn block_stats(&self) -> Option<boostr::inference::memory::BlockAllocatorStats> {
        RequestScheduler::block_stats(self)
    }

    fn queue_depth(&self) -> usize {
        RequestScheduler::queue_depth(self)
    }

    fn fault(&self) -> Option<String> {
        self.registry.fault()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LatencySlo;
    use crate::telemetry::names::QUEUE_DEPTH;
    use crate::telemetry::test_recorder::TestRecorder;

    fn scheduler(
        max_batch_size: usize,
        policy: Option<Arc<SloBatchPolicy>>,
    ) -> Arc<RequestScheduler> {
        let config = SchedulerConfig {
            max_batch_size,
            max_batch_tokens: 4096,
            block_size: 4,
            ..Default::default()
        };
        RequestScheduler::build(CpuBlockAllocator::new(256, 4), config, None, policy)
    }

    fn gen_config() -> GenerationConfig {
        GenerationConfig {
            max_tokens: 8,
            ..GenerationConfig::default()
        }
    }

    fn prefills(scheduler: &RequestScheduler) -> usize {
        scheduler
            .schedule_batch()
            .ok()
            .flatten()
            .map_or(0, |batch| batch.prefill_sequences.len())
    }

    #[test]
    fn queue_depth_follows_submit_schedule_and_cancel() {
        let recorder = TestRecorder::install();
        let scheduler = scheduler(2, None);
        let mut handles = Vec::new();
        for _ in 0..3 {
            handles.push(
                scheduler
                    .submit(vec![1, 2, 3], gen_config())
                    .expect("submit"),
            );
        }
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(3.0));

        // Two slots: two requests start, one keeps waiting.
        assert_eq!(prefills(&scheduler), 2);
        assert_eq!(scheduler.queue_depth(), 1);
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(1.0));

        scheduler.cancel(handles[2].seq_id).expect("cancel");
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(0.0));

        drop(scheduler);
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(0.0));
    }

    /// A pool of two 4-token blocks gives an 8-token context.
    #[tokio::test]
    async fn max_tokens_is_capped_by_the_context_the_pool_holds() {
        let config = SchedulerConfig {
            max_batch_size: 2,
            block_size: 4,
            ..Default::default()
        };
        let scheduler = RequestScheduler::new(CpuBlockAllocator::new(2, 4), config);
        let long = GenerationConfig {
            max_tokens: 100,
            ..GenerationConfig::default()
        };

        let handle = scheduler
            .submit(vec![1, 2, 3], long.clone())
            .expect("submit");
        let capped = scheduler
            .get_gen_config(handle.seq_id)
            .map(|c| c.max_tokens);
        assert_eq!(capped, Some(5));

        let mut full = scheduler
            .submit(vec![1; 8], long.clone())
            .expect("a prompt that fills the context is admitted");
        assert!(full.token_rx.recv().await.is_none(), "no token is decoded");
        assert!(!scheduler.is_live(full.seq_id));

        let err = scheduler.submit(vec![1; 9], long);
        assert!(err.is_err_and(|e| e.to_string().contains("exceeds the 8-token context")));
    }

    #[test]
    fn an_slo_breach_stops_batch_growth_until_latency_recovers() {
        let slo = LatencySlo {
            ttft_p50_ms: None,
            ttft_p95_ms: Some(100),
            ttft_p99_ms: None,
            itl_p50_ms: None,
            itl_p95_ms: Some(20),
            itl_p99_ms: None,
            e2e_p99_ms: None,
        };
        let policy = Arc::new(SloBatchPolicy::from_slo(&slo).expect("policy"));
        let scheduler = scheduler(4, Some(Arc::clone(&policy)));

        let _first = scheduler.submit(vec![1, 2], gen_config()).expect("submit");
        assert_eq!(prefills(&scheduler), 1);

        // The P95 ITL breaks its target: the running batch of one stays one.
        policy.observe_itl_ms(80.0);
        let _second = scheduler.submit(vec![3, 4], gen_config()).expect("submit");
        let _third = scheduler.submit(vec![5, 6], gen_config()).expect("submit");
        assert_eq!(prefills(&scheduler), 0);
        assert_eq!(scheduler.queue_depth(), 2);

        // Fast samples bring the P95 under target: both waiting requests join.
        for _ in 0..40 {
            policy.observe_itl_ms(5.0);
        }
        assert_eq!(prefills(&scheduler), 2);
        assert_eq!(scheduler.queue_depth(), 0);
    }
}
