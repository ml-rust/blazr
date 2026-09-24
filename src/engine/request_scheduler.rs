//! Request scheduler — bridges HTTP requests to boostr's SequenceScheduler
//!
//! This module wraps boostr's continuous batching SequenceScheduler and provides:
//! - Async request submission from HTTP handlers
//! - Token streaming back to callers via channels
//! - A scheduling loop that produces batches for the executor
//!
//! The per-request metadata and channels live in [`RequestRegistry`], shared
//! with the other schedulers.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc, oneshot};

use boostr::inference::memory::CpuBlockAllocator;
use boostr::inference::scheduler::{
    ScheduledBatch, SchedulerConfig, SequenceId, SequenceRequest, SequenceScheduler, SequenceState,
};

use super::request_registry::{GenerationScheduler, RequestHandle, RequestRegistry};
use super::types::GeneratedToken;
use crate::config::GenerationConfig;

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
    inner: Mutex<SequenceScheduler<CpuBlockAllocator>>,
    /// Per-sequence metadata (gen config, token channels)
    registry: RequestRegistry,
    /// Channel to notify the scheduling loop of new work
    notify_tx: mpsc::Sender<()>,
    /// Receiver end (held by the scheduling loop)
    notify_rx: Mutex<Option<mpsc::Receiver<()>>>,
}

impl RequestScheduler {
    /// Create a new request scheduler with the given allocator and config
    pub fn new(allocator: CpuBlockAllocator, config: SchedulerConfig) -> Arc<Self> {
        let (notify_tx, notify_rx) = mpsc::channel(NOTIFY_CHANNEL_CAPACITY);
        Arc::new(Self {
            inner: Mutex::new(SequenceScheduler::new(allocator, config)),
            registry: RequestRegistry::new(),
            notify_tx,
            notify_rx: Mutex::new(Some(notify_rx)),
        })
    }

    /// Create a request scheduler with prefix caching enabled.
    pub fn with_prefix_cache(
        allocator: CpuBlockAllocator,
        config: SchedulerConfig,
        prefix_cache: boostr::inference::prefix_cache::PrefixCache<CpuBlockAllocator>,
    ) -> Arc<Self> {
        let (notify_tx, notify_rx) = mpsc::channel(NOTIFY_CHANNEL_CAPACITY);
        let scheduler = SequenceScheduler::new(allocator, config).with_prefix_cache(prefix_cache);
        Arc::new(Self {
            inner: Mutex::new(scheduler),
            registry: RequestRegistry::new(),
            notify_tx,
            notify_rx: Mutex::new(Some(notify_rx)),
        })
    }

    /// Submit a new inference request. Returns a handle for receiving tokens.
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        gen_config: GenerationConfig,
    ) -> Result<RequestHandle> {
        let seq_id = self.registry.next_id();
        let request = SequenceRequest::new(seq_id, prompt_tokens.clone())
            .with_max_tokens(gen_config.max_tokens);

        {
            let mut sched = self
                .inner
                .lock()
                .map_err(|e| anyhow!("scheduler lock: {e}"))?;
            sched.add_request(request).map_err(|e| anyhow!("{e}"))?;
        }

        let handle = self.registry.insert(seq_id, prompt_tokens, gen_config)?;

        // Wake the scheduling loop
        let _ = self.notify_tx.try_send(());

        Ok(handle)
    }

    /// Cancel a pending or running request
    pub fn cancel(&self, seq_id: SequenceId) -> Result<()> {
        {
            let mut sched = self
                .inner
                .lock()
                .map_err(|e| anyhow!("scheduler lock: {e}"))?;
            sched.abort_sequence(seq_id).map_err(|e| anyhow!("{e}"))?;
        }
        self.registry.remove(seq_id)
    }

    /// Schedule the next batch of work. Called by the scheduling loop.
    /// Returns None if no work is available.
    pub fn schedule_batch(&self) -> Result<Option<ScheduledBatch>> {
        let mut sched = self
            .inner
            .lock()
            .map_err(|e| anyhow!("scheduler lock: {e}"))?;
        sched.schedule().map_err(|e| anyhow!("{e}"))
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
            // Only finish if still tracked (might already be finished via append_token)
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
}
