//! The marks a batched cancel leaves for the engine, and the phase a
//! cancelled sequence left the scheduler in.
//!
//! A cancel from outside the engine marks a running request in
//! [`PendingCancels`]. The engine drops each marked sequence at its next
//! check. [`CancelPhase`] names where the sequence was, for the debug log.

use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard, PoisonError};

use boostr::inference::scheduler::{SequenceId, SequenceState};

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

    /// Mark `seq_id` cancelled.
    pub(super) fn mark(&self, seq_id: SequenceId) {
        self.ids().insert(seq_id);
    }

    /// Whether `seq_id` is marked cancelled.
    pub(super) fn contains(&self, seq_id: SequenceId) -> bool {
        self.ids().contains(&seq_id)
    }

    /// Every marked id.
    pub(super) fn snapshot(&self) -> Vec<SequenceId> {
        self.ids().iter().copied().collect()
    }

    /// Remove `seq_id`'s mark.
    pub(super) fn clear(&self, seq_id: SequenceId) {
        self.ids().remove(&seq_id);
    }

    /// Remove every mark.
    pub(super) fn clear_all(&self) {
        self.ids().clear();
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
    pub(super) fn of(state: Option<SequenceState>) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_are_set_cleared_and_cleared_all() {
        let marks = PendingCancels::new();
        marks.mark(3);
        marks.mark(5);
        assert!(marks.contains(3));
        marks.clear(3);
        assert!(!marks.contains(3));
        assert_eq!(marks.snapshot(), vec![5]);
        marks.clear_all();
        assert!(marks.snapshot().is_empty());
    }

    #[test]
    fn the_phase_follows_the_sequence_state() {
        assert_eq!(
            CancelPhase::of(Some(SequenceState::Waiting)),
            CancelPhase::Queued
        );
        assert_eq!(
            CancelPhase::of(Some(SequenceState::Preempted)),
            CancelPhase::Preempted
        );
        assert_eq!(CancelPhase::of(None), CancelPhase::Finished);
        assert_eq!(CancelPhase::Decode.as_str(), "decode");
    }
}
