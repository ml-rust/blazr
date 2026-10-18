//! Cancellation of batched requests.
//!
//! A cancel can come from any task, most often from a dropped
//! [`RequestHandle`]. A request that waits for admission, or sits preempted,
//! leaves the scheduler at once and frees its blocks. A request the engine
//! runs is only marked: [`RequestScheduler::is_live`] turns false at once,
//! and the engine drops the sequence at its next check. So a batch in
//! flight never loses a sequence under it.
//!
//! The engine checks before it schedules a batch, before each prefill chunk,
//! before it builds a decode forward, and before it samples a row. A token
//! send to a closed channel ends the sequence at once too.
//!
//! [`RequestHandle`]: super::request_registry::RequestHandle

use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard, PoisonError, TryLockError};

use anyhow::{anyhow, Result};

use boostr::inference::memory::CpuBlockAllocator;
use boostr::inference::scheduler::{SequenceId, SequenceScheduler, SequenceState};
use boostr::{DType, Runtime};

use super::request_registry::CancelRequest;
use super::request_scheduler::RequestScheduler;
use super::seq_sampler::BatchSeqs;

/// Running requests marked cancelled that the engine has not dropped yet.
pub(super) struct PendingCancels {
    ids: Mutex<HashSet<SequenceId>>,
}

impl PendingCancels {
    /// An empty set.
    pub(super) fn new() -> Self {
        Self {
            ids: Mutex::new(HashSet::new()),
        }
    }

    /// The set. A poisoned lock still holds a plain id set, so it is used.
    fn ids(&self) -> MutexGuard<'_, HashSet<SequenceId>> {
        self.ids.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn mark(&self, seq_id: SequenceId) {
        self.ids().insert(seq_id);
    }

    /// Whether `seq_id` is marked cancelled.
    pub(super) fn contains(&self, seq_id: SequenceId) -> bool {
        self.ids().contains(&seq_id)
    }

    fn snapshot(&self) -> Vec<SequenceId> {
        self.ids().iter().copied().collect()
    }

    fn clear(&self, seq_id: SequenceId) {
        self.ids().remove(&seq_id);
    }
}

/// Where a cancelled sequence was when it left the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CancelPhase {
    /// Waiting for admission. It holds no blocks.
    Queued,
    /// Admitted, before its first token.
    Prefill,
    /// Past its first token.
    Decode,
    /// Running, dropped before the engine scheduled it again.
    Running,
    /// Preempted. Its blocks are already free, so it skips the recompute.
    Preempted,
    /// The scheduler had already ended it.
    Finished,
}

impl CancelPhase {
    /// The phase of a sequence in `state`.
    fn of(state: Option<SequenceState>) -> Self {
        match state {
            Some(SequenceState::Waiting) => Self::Queued,
            Some(SequenceState::Running) => Self::Running,
            Some(SequenceState::Preempted) => Self::Preempted,
            Some(SequenceState::Finished) | None => Self::Finished,
        }
    }

    /// The phase name the debug log carries.
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Running => "running",
            Self::Preempted => "preempted",
            Self::Finished => "finished",
        }
    }
}

impl RequestScheduler {
    /// Cancel a waiting or running request.
    ///
    /// A waiting or preempted request leaves the scheduler at once. A
    /// running one is marked, and the engine drops it at its next check.
    /// The call never waits on the scheduler lock: under contention the
    /// mark alone stands. A second call, or a call for a finished request,
    /// is a no-op.
    ///
    /// # Errors
    ///
    /// An error when a scheduler lock is poisoned or the abort fails.
    pub fn cancel(&self, seq_id: SequenceId) -> Result<()> {
        if !self.registry.contains(seq_id) {
            return Ok(());
        }
        self.cancels.mark(seq_id);
        let phase = match self.inner.try_lock() {
            Ok(mut sched) => {
                let phase = CancelPhase::of(sched.get_sequence_state(seq_id));
                let idle = matches!(phase, CancelPhase::Queued | CancelPhase::Preempted);
                if idle {
                    sched
                        .abort_sequence(seq_id)
                        .map_err(|e| anyhow!("aborting cancelled sequence {seq_id} failed: {e}"))?;
                    self.publish_depth(&sched);
                }
                idle.then_some(phase)
            }
            Err(TryLockError::WouldBlock) => None,
            Err(TryLockError::Poisoned(e)) => {
                return Err(anyhow!(
                    "the scheduler lock is poisoned: {e}; restart the server"
                ));
            }
        };
        match phase {
            Some(phase) => self.forget(seq_id, phase),
            None => {
                // Wake an idle engine so it drops the sequence now.
                let _ = self.notify_tx.try_send(());
                Ok(())
            }
        }
    }

    /// Drop every request marked cancelled: free its blocks and its state.
    ///
    /// The engine calls this between ticks, before it checks for work.
    ///
    /// # Errors
    ///
    /// An error when a scheduler lock is poisoned or an abort fails.
    pub(super) fn reap_cancelled(&self) -> Result<()> {
        let aborted = {
            let mut sched = self
                .inner
                .lock()
                .map_err(|e| anyhow!("scheduler lock: {e}"))?;
            let aborted = self.abort_marked(&mut sched)?;
            if !aborted.is_empty() {
                self.publish_depth(&sched);
            }
            aborted
        };
        self.forget_all(aborted)
    }

    /// End `seq_id` at once from inside the engine: free its blocks and drop
    /// its request. The engine calls this when it finds the sequence
    /// cancelled or its token channel closed.
    ///
    /// # Errors
    ///
    /// An error when a scheduler lock is poisoned or the abort fails.
    pub(super) fn cancel_now(&self, seq_id: SequenceId, phase: CancelPhase) -> Result<()> {
        {
            let mut sched = self
                .inner
                .lock()
                .map_err(|e| anyhow!("scheduler lock: {e}"))?;
            if CancelPhase::of(sched.get_sequence_state(seq_id)) != CancelPhase::Finished {
                sched
                    .abort_sequence(seq_id)
                    .map_err(|e| anyhow!("aborting cancelled sequence {seq_id} failed: {e}"))?;
                self.publish_depth(&sched);
            }
        }
        self.forget(seq_id, phase)
    }

    /// Abort every marked sequence `sched` still runs. Returns each marked
    /// id with its phase. The caller holds the scheduler lock.
    ///
    /// # Errors
    ///
    /// An error when an abort fails.
    pub(super) fn abort_marked(
        &self,
        sched: &mut SequenceScheduler<CpuBlockAllocator>,
    ) -> Result<Vec<(SequenceId, CancelPhase)>> {
        let mut aborted = Vec::new();
        for seq_id in self.cancels.snapshot() {
            let phase = CancelPhase::of(sched.get_sequence_state(seq_id));
            if phase != CancelPhase::Finished {
                sched
                    .abort_sequence(seq_id)
                    .map_err(|e| anyhow!("aborting cancelled sequence {seq_id} failed: {e}"))?;
            }
            aborted.push((seq_id, phase));
        }
        Ok(aborted)
    }

    /// Drop the request state of every id [`Self::abort_marked`] returned.
    /// The caller has released the scheduler lock.
    ///
    /// # Errors
    ///
    /// An error when the registry lock is poisoned.
    pub(super) fn forget_all(&self, aborted: Vec<(SequenceId, CancelPhase)>) -> Result<()> {
        for (seq_id, phase) in aborted {
            self.forget(seq_id, phase)?;
        }
        Ok(())
    }

    /// Drop `seq_id`'s request and its mark. Only the call that removes a
    /// carried request logs it, so each cancel counts once.
    fn forget(&self, seq_id: SequenceId, phase: CancelPhase) -> Result<()> {
        let carried = self.registry.remove(seq_id)?;
        self.cancels.clear(seq_id);
        if carried {
            tracing::debug!(
                seq_id,
                phase = phase.as_str(),
                "batched request cancelled; its KV blocks and batch slot are free"
            );
        }
        Ok(())
    }
}

impl CancelRequest for RequestScheduler {
    fn cancel_request(&self, seq_id: SequenceId) {
        if let Err(e) = self.cancel(seq_id) {
            tracing::error!(seq_id, "cancelling a dropped request failed: {e:#}");
        }
    }
}

/// End a cancelled sequence inside the engine: its sampler state, its KV
/// blocks and its request.
///
/// # Errors
///
/// An error when a scheduler lock is poisoned or the abort fails.
pub(super) fn end_cancelled<R: Runtime<DType = DType>>(
    scheduler: &RequestScheduler,
    seqs: &mut BatchSeqs<R>,
    seq_id: SequenceId,
    phase: CancelPhase,
) -> Result<()> {
    seqs.end_seq(seq_id);
    scheduler.cancel_now(seq_id, phase)
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::batch_test_rig::{alone, drain, free_blocks, greedy, ids, rig, Rig, POOL};
    use crate::engine::test_support::PagedPool;

    /// Ticks a test runs while it watches the queued requests.
    const WATCH_TICKS: usize = 32;

    /// Blocks `seq_id` holds in the scheduler.
    fn held(rig: &Rig, seq_id: SequenceId) -> usize {
        rig.scheduler
            .get_block_table(seq_id)
            .map_or(0, |blocks| blocks.len())
    }

    #[tokio::test]
    async fn a_dropped_handle_stops_decoding_and_frees_its_blocks() -> Result<()> {
        let mut rig = rig(POOL)?;
        let start = free_blocks(&rig)?;
        let mut handle = rig.scheduler.submit(vec![1, 2, 3], greedy(100))?;
        rig.engine.tick().await;
        assert_eq!(ids(&mut handle)?.len(), 1, "the prefill streams one token");
        let seq_id = handle.seq_id;
        let forwards = rig.engine.seqs.forwards;

        drop(handle);
        assert!(!rig.scheduler.is_live(seq_id), "the drop cancels at once");
        assert!(!rig.engine.has_live_work(), "only cancelled work was left");
        for _ in 0..3 {
            rig.engine.tick().await;
        }

        assert_eq!(
            rig.engine.seqs.forwards, forwards,
            "no forward runs for a dropped request"
        );
        assert!(!rig.scheduler.has_work());
        assert_eq!(free_blocks(&rig)?, start);
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn a_closed_channel_ends_the_sequence_at_its_next_token() -> Result<()> {
        let mut rig = rig(POOL)?;
        let start = free_blocks(&rig)?;
        let mut handle = rig.scheduler.submit(vec![1, 2, 3], greedy(8))?;
        rig.engine.tick().await;
        let forwards = rig.engine.seqs.forwards;

        // The receiver stops listening while the handle stays alive.
        handle.token_rx.close();
        rig.engine.tick().await;

        assert_eq!(rig.engine.seqs.forwards, forwards + 1, "one decode runs");
        assert!(
            !rig.scheduler.is_live(handle.seq_id),
            "the failed send ends it"
        );
        assert!(!rig.engine.has_live_work());
        assert_eq!(free_blocks(&rig)?, start);
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn the_other_row_keeps_its_output_when_one_is_dropped() -> Result<()> {
        let mut rig = rig(POOL)?;
        let start = free_blocks(&rig)?;
        let (dropped, kept) = ([1, 2, 3], [7, 8, 9, 10]);
        let kept_alone = alone(&rig.executor, &kept, &greedy(8)).await?;

        let gone = rig.scheduler.submit(dropped.to_vec(), greedy(8))?;
        let mut stays = rig.scheduler.submit(kept.to_vec(), greedy(8))?;
        rig.engine.tick().await;
        rig.engine.tick().await;
        assert_eq!(rig.engine.seqs.len(), 2, "both rows decode together");

        drop(gone);
        let forwards = rig.engine.seqs.forwards;
        rig.engine.tick().await;
        assert_eq!(rig.engine.seqs.forwards, forwards + 1);
        assert_eq!(rig.engine.seqs.len(), 1, "the dropped row left the batch");
        assert_eq!(
            free_blocks(&rig)?,
            start - held(&rig, stays.seq_id),
            "the dropped row's blocks are free"
        );

        drain(&mut rig).await?;
        assert_eq!(ids(&mut stays)?, kept_alone);
        assert_eq!(free_blocks(&rig)?, start);
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    /// Two long requests fill both slots. Of the two queued requests, one is
    /// cancelled at once. The other is cancelled while the engine holds the
    /// scheduler lock, so only its mark stands until the next schedule.
    #[tokio::test]
    async fn a_queued_request_is_never_admitted_once_cancelled() -> Result<()> {
        let mut rig = rig(POOL)?;
        let start = free_blocks(&rig)?;
        let (a_prompt, b_prompt) = ([1, 2, 3], [7, 8, 9, 10]);
        let a_alone = alone(&rig.executor, &a_prompt, &greedy(12)).await?;
        let b_alone = alone(&rig.executor, &b_prompt, &greedy(12)).await?;
        let mut a = rig.scheduler.submit(a_prompt.to_vec(), greedy(12))?;
        let mut b = rig.scheduler.submit(b_prompt.to_vec(), greedy(12))?;
        rig.engine.tick().await;
        let c = rig.scheduler.submit(vec![4, 5, 6], greedy(4))?;
        let d = rig.scheduler.submit(vec![11, 12], greedy(4))?;
        assert_eq!(rig.scheduler.queue_depth(), 2);
        let (c_id, d_id) = (c.seq_id, d.seq_id);

        drop(c);
        assert!(!rig.scheduler.is_live(c_id));
        assert_eq!(
            rig.scheduler.queue_depth(),
            1,
            "the free lock aborts at once"
        );

        {
            let _engine_holds = rig
                .scheduler
                .inner
                .lock()
                .map_err(|e| anyhow!("scheduler lock: {e}"))?;
            // `Drop` must not wait on the lock this thread holds.
            drop(d);
        }
        assert!(!rig.scheduler.is_live(d_id));
        assert_eq!(rig.scheduler.queue_depth(), 1, "only the mark stands");

        rig.engine.tick().await;
        assert_eq!(
            rig.scheduler.queue_depth(),
            0,
            "the schedule drops it first"
        );
        assert_eq!(
            free_blocks(&rig)?,
            start - held(&rig, a.seq_id) - held(&rig, b.seq_id),
            "no prompt block goes to a cancelled request"
        );
        for _ in 0..WATCH_TICKS {
            if !rig.engine.has_live_work() {
                break;
            }
            rig.engine.tick().await;
            for id in [c_id, d_id] {
                assert!(!rig.engine.seqs.contains(id), "sequence {id} was admitted");
                assert_eq!(held(&rig, id), 0, "sequence {id} holds blocks");
            }
        }

        assert!(!rig.scheduler.has_work());
        assert_eq!(ids(&mut a)?, a_alone);
        assert_eq!(ids(&mut b)?, b_alone);
        assert_eq!(free_blocks(&rig)?, start);
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    /// Chunks of two over a nine-token prompt. The handle drops after the
    /// first chunk, so the second chunk never runs.
    #[tokio::test]
    async fn a_cancel_between_prefill_chunks_stops_the_prefill() -> Result<()> {
        let Rig {
            executor,
            scheduler,
            engine,
        } = rig(POOL)?;
        let mut rig = Rig {
            executor,
            scheduler,
            engine: engine.with_prefill_chunk_size(2),
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
        assert_eq!(free_blocks(&rig)?, start);
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    /// Three blocks of four. After both prefills the pool is empty, so the
    /// next schedule preempts `b` and gives its block to the queued `c`.
    /// The batch is then full, so `b` stays preempted.
    #[tokio::test]
    async fn a_preempted_sequence_is_dropped_without_a_recompute() -> Result<()> {
        let pool = PagedPool {
            block_size: 4,
            blocks: 3,
        };
        let mut rig = rig(pool)?;
        let start = free_blocks(&rig)?;
        let (a_prompt, c_prompt) = ([1, 2, 3], [7, 8]);
        let a_alone = alone(&rig.executor, &a_prompt, &greedy(3)).await?;
        let c_alone = alone(&rig.executor, &c_prompt, &greedy(2)).await?;

        let mut a = rig.scheduler.submit(a_prompt.to_vec(), greedy(3))?;
        let b = rig.scheduler.submit(vec![4, 5, 6], greedy(8))?;
        rig.engine.tick().await;
        let mut c = rig.scheduler.submit(c_prompt.to_vec(), greedy(2))?;
        rig.engine.tick().await;
        let preempted = |rig: &Rig| rig.scheduler.stats().map_or(0, |s| s.preempted_count);
        assert_eq!(preempted(&rig), 1, "the pool must leave b preempted");

        let b_id = b.seq_id;
        drop(b);
        assert!(!rig.scheduler.is_live(b_id));
        assert_eq!(preempted(&rig), 0, "a preempted request leaves at once");

        drain(&mut rig).await?;
        assert_eq!(rig.engine.seqs.recomputes, 0, "no KV refill for b");
        assert_eq!(ids(&mut a)?, a_alone);
        assert_eq!(ids(&mut c)?, c_alone);
        assert_eq!(free_blocks(&rig)?, start);
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn cancelling_twice_or_after_the_finish_is_a_no_op() -> Result<()> {
        let mut rig = rig(POOL)?;
        let start = free_blocks(&rig)?;
        let running = rig.scheduler.submit(vec![1, 2, 3], greedy(8))?;
        rig.engine.tick().await;
        rig.scheduler.cancel(running.seq_id)?;
        rig.scheduler.cancel(running.seq_id)?;
        assert!(!rig.engine.has_live_work());
        rig.scheduler.cancel(running.seq_id)?;
        drop(running);
        assert!(!rig.engine.has_live_work());

        let mut finished = rig.scheduler.submit(vec![7, 8, 9, 10], greedy(2))?;
        drain(&mut rig).await?;
        assert_eq!(ids(&mut finished)?.len(), 2);
        let finished_id = finished.seq_id;
        rig.scheduler.cancel(finished_id)?;
        drop(finished);
        assert!(
            !rig.scheduler.cancels.contains(finished_id),
            "a finished request leaves no mark"
        );

        assert!(!rig.engine.has_live_work());
        assert_eq!(free_blocks(&rig)?, start);
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }
}
