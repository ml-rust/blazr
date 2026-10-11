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
//! The queue length is published as `blazr_queue_depth`. With an SLO
//! policy, a P95 latency breach stops the engine from filling more lanes.
//!
//! [`LaneEngine`]: super::lane_engine::LaneEngine

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use tokio::sync::mpsc;

use boostr::inference::scheduler::SequenceId;

use super::queue_depth::QueueDepth;
use super::request_registry::{GenerationScheduler, RequestHandle, RequestRegistry};
use super::slo_policy::SloBatchPolicy;
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
    /// The waiting queue's length, published as `blazr_queue_depth`.
    queue_depth: QueueDepth,
    /// Holds lane filling while a P95 latency target is breached.
    slo_policy: Option<Arc<SloBatchPolicy>>,
}

impl LaneScheduler {
    /// A scheduler feeding `lanes` lanes.
    ///
    /// # Errors
    ///
    /// An error when `lanes` is 0.
    pub fn new(lanes: usize) -> Result<Arc<Self>> {
        Self::with_slo_policy(lanes, None)
    }

    /// A scheduler feeding `lanes` lanes, with an optional SLO policy.
    ///
    /// # Errors
    ///
    /// An error when `lanes` is 0.
    pub fn with_slo_policy(
        lanes: usize,
        slo_policy: Option<Arc<SloBatchPolicy>>,
    ) -> Result<Arc<Self>> {
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
            queue_depth: QueueDepth::new(),
            slo_policy,
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
        {
            let mut waiting = self.lock_waiting()?;
            waiting.push_back(seq_id);
            self.queue_depth.set(waiting.len());
        }
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
        {
            let mut waiting = self.lock_waiting()?;
            waiting.retain(|&id| id != seq_id);
            self.queue_depth.set(waiting.len());
        }
        self.registry.remove(seq_id)
    }

    /// The oldest waiting request, removed from the queue.
    pub fn next_waiting(&self) -> Option<SequenceId> {
        let mut waiting = self.waiting.lock().ok()?;
        let next = waiting.pop_front();
        self.queue_depth.set(waiting.len());
        next
    }

    /// The most lanes the driver fills when `live` lanes are occupied.
    ///
    /// Without an SLO breach this is every lane. Under a breach it is
    /// `live`, with a floor of 1, so the occupied set does not grow.
    pub fn admission_limit(&self, live: usize) -> usize {
        match &self.slo_policy {
            Some(policy) => policy.batch_limit(live, self.lanes),
            None => self.lanes,
        }
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

    fn queue_depth(&self) -> usize {
        self.waiting_len()
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
    fn queue_depth_follows_submit_admit_and_cancel() {
        use crate::server::metrics::names::QUEUE_DEPTH;
        use crate::server::test_recorder::TestRecorder;

        let recorder = TestRecorder::install();
        let scheduler = LaneScheduler::new(2).expect("new");
        let first = scheduler.submit(vec![1], config()).expect("submit");
        let _second = scheduler.submit(vec![2], config()).expect("submit");
        let third = scheduler.submit(vec![3], config()).expect("submit");
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(3.0));
        assert_eq!(scheduler.next_waiting(), Some(first.seq_id));
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(2.0));
        scheduler.cancel(third.seq_id).expect("cancel");
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(1.0));
        drop(scheduler);
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(0.0));
    }

    #[test]
    fn an_slo_breach_stops_filling_lanes() {
        let slo = crate::config::LatencySlo {
            ttft_p50_ms: None,
            ttft_p95_ms: Some(100),
            ttft_p99_ms: None,
            itl_p50_ms: None,
            itl_p95_ms: None,
            itl_p99_ms: None,
            e2e_p99_ms: None,
        };
        let policy = Arc::new(SloBatchPolicy::from_slo(&slo).expect("policy"));
        let scheduler = LaneScheduler::with_slo_policy(4, Some(Arc::clone(&policy))).expect("new");
        assert_eq!(scheduler.admission_limit(2), 4);
        policy.observe_ttft_ms(300.0);
        assert_eq!(scheduler.admission_limit(2), 2);
        assert_eq!(scheduler.admission_limit(0), 1);
        assert_eq!(LaneScheduler::new(4).expect("new").admission_limit(2), 4);
    }

    #[test]
    fn the_notify_receiver_is_taken_once() {
        let scheduler = LaneScheduler::new(1).expect("new");
        assert!(scheduler.take_notify_rx().is_some());
        assert!(scheduler.take_notify_rx().is_none());
    }
}
