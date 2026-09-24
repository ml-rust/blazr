//! Queue in front of a fixed set of `qwen35` decode lanes.
//!
//! A submitted request joins [`LaneScheduler`]'s waiting queue and gets a
//! token channel from the shared [`RequestRegistry`]. [`LaneEngine`] takes
//! requests off the queue, puts each into a free lane, steps every occupied
//! lane together, and gives a lane back as soon as its request ends.
//!
//! The scheduler itself never touches a model or a device: it holds the queue,
//! the lane count and the per-request state, so the HTTP handlers reach it
//! through [`GenerationScheduler`] exactly as they reach the paged scheduler.
//!
//! [`LaneEngine`]: super::lane_engine::LaneEngine

#![cfg(feature = "cuda")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use tokio::sync::mpsc;

use boostr::inference::scheduler::SequenceId;

use super::request_registry::{GenerationScheduler, RequestHandle, RequestRegistry};
use crate::config::GenerationConfig;

/// Wakeups the driver can miss without blocking a submitting handler.
const NOTIFY_CHANNEL_CAPACITY: usize = 64;

/// Requests waiting for a `qwen35` decode lane, and the state of the requests
/// already in one.
pub struct LaneScheduler {
    /// Generation config, token history and channels of every live request.
    registry: RequestRegistry,
    /// Submitted requests with no lane yet, oldest first.
    waiting: Mutex<VecDeque<SequenceId>>,
    /// Lanes the session the driver opens will hold.
    lanes: usize,
    /// Wakes the driver when a request is queued.
    notify_tx: mpsc::Sender<()>,
    /// Receiver end, taken once by the driver.
    notify_rx: Mutex<Option<mpsc::Receiver<()>>>,
}

impl LaneScheduler {
    /// A scheduler feeding `lanes` lanes.
    ///
    /// # Errors
    ///
    /// An error when `lanes` is 0.
    pub fn new(lanes: usize) -> Result<Arc<Self>> {
        if lanes == 0 {
            return Err(anyhow!(
                "a lane scheduler needs at least one lane, got lanes=0; set \
                 inference.max_batch_size to the number of sequences to decode at once"
            ));
        }
        let (notify_tx, notify_rx) = mpsc::channel(NOTIFY_CHANNEL_CAPACITY);
        Ok(Arc::new(Self {
            registry: RequestRegistry::new(),
            waiting: Mutex::new(VecDeque::new()),
            lanes,
            notify_tx,
            notify_rx: Mutex::new(Some(notify_rx)),
        }))
    }

    /// Queue `prompt_tokens` and hand back the channel its tokens arrive on.
    ///
    /// # Errors
    ///
    /// An error when the queue or the registry lock is poisoned.
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        gen_config: GenerationConfig,
    ) -> Result<RequestHandle> {
        let seq_id = self.registry.next_id();
        let handle = self.registry.insert(seq_id, prompt_tokens, gen_config)?;
        self.lock_waiting()?.push_back(seq_id);
        let _ = self.notify_tx.try_send(());
        Ok(handle)
    }

    /// Drop a request: out of the queue, out of the registry.
    ///
    /// A request already in a lane keeps that lane until the driver's next
    /// step finds its channel gone and releases it.
    ///
    /// # Errors
    ///
    /// An error when the queue or the registry lock is poisoned.
    pub fn cancel(&self, seq_id: SequenceId) -> Result<()> {
        self.lock_waiting()?.retain(|&id| id != seq_id);
        self.registry.remove(seq_id)
    }

    /// The oldest waiting request, removed from the queue.
    pub fn next_waiting(&self) -> Option<SequenceId> {
        self.waiting.lock().ok()?.pop_front()
    }

    /// How many requests are waiting for a lane.
    pub fn waiting_len(&self) -> usize {
        self.waiting.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Lanes the driver opens for this scheduler.
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// The per-request state, for the driver to read and stream through.
    pub fn registry(&self) -> &RequestRegistry {
        &self.registry
    }

    /// Take the wakeup receiver. The driver calls this once.
    pub fn take_notify_rx(&self) -> Option<mpsc::Receiver<()>> {
        self.notify_rx.lock().ok()?.take()
    }

    /// The waiting queue, or an error naming the poisoned lock.
    fn lock_waiting(&self) -> Result<std::sync::MutexGuard<'_, VecDeque<SequenceId>>> {
        self.waiting
            .lock()
            .map_err(|e| anyhow!("lane scheduler queue lock is poisoned: {e}; restart the server"))
    }
}

impl GenerationScheduler for LaneScheduler {
    fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        gen_config: GenerationConfig,
    ) -> Result<RequestHandle> {
        LaneScheduler::submit(self, prompt_tokens, gen_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GenerationConfig {
        GenerationConfig {
            max_tokens: 3,
            ..GenerationConfig::default()
        }
    }

    #[test]
    fn zero_lanes_is_refused() {
        let err = LaneScheduler::new(0).err().map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|m| m.contains("at least one lane")),
            "{err:?}"
        );
    }

    #[test]
    fn submit_queues_in_order() {
        let scheduler = LaneScheduler::new(2).expect("new");
        let first = scheduler.submit(vec![1, 2], config()).expect("submit");
        let second = scheduler.submit(vec![3], config()).expect("submit");
        assert_eq!(scheduler.waiting_len(), 2);
        assert_eq!(scheduler.next_waiting(), Some(first.seq_id));
        assert_eq!(scheduler.next_waiting(), Some(second.seq_id));
        assert_eq!(scheduler.next_waiting(), None);
        assert_eq!(
            scheduler.registry().prompt_tokens(second.seq_id),
            Some(vec![3])
        );
    }

    #[test]
    fn cancel_drops_a_waiting_request() {
        let scheduler = LaneScheduler::new(2).expect("new");
        let first = scheduler.submit(vec![1], config()).expect("submit");
        let second = scheduler.submit(vec![2], config()).expect("submit");
        scheduler.cancel(first.seq_id).expect("cancel");
        assert_eq!(scheduler.waiting_len(), 1);
        assert!(!scheduler.registry().contains(first.seq_id));
        assert_eq!(scheduler.next_waiting(), Some(second.seq_id));
    }

    #[test]
    fn the_notify_receiver_is_taken_once() {
        let scheduler = LaneScheduler::new(1).expect("new");
        assert!(scheduler.take_notify_rx().is_some());
        assert!(scheduler.take_notify_rx().is_none());
    }
}
