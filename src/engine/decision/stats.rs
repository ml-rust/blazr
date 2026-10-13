//! Counts of what a model's decision worker did, for tests and operators.

use std::sync::atomic::{AtomicU64, Ordering};

/// Running counts, shared by a worker and its service.
#[derive(Debug, Default)]
pub struct DecisionStats {
    answered: AtomicU64,
    failed: AtomicU64,
    abandoned: AtomicU64,
    merged_runs: AtomicU64,
    merged_requests: AtomicU64,
    prefix_runs: AtomicU64,
}

/// A snapshot of [`DecisionStats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecisionCounts {
    /// Requests answered.
    pub answered: u64,
    /// Requests that ran and failed.
    pub failed: u64,
    /// Requests dropped before running: the caller went away.
    pub abandoned: u64,
    /// Runs that held two or more requests.
    pub merged_runs: u64,
    /// Requests answered by a merged run.
    pub merged_requests: u64,
    /// Requests run alone over their shared prefix.
    pub prefix_runs: u64,
}

impl DecisionStats {
    /// The counts now.
    pub fn snapshot(&self) -> DecisionCounts {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        DecisionCounts {
            answered: get(&self.answered),
            failed: get(&self.failed),
            abandoned: get(&self.abandoned),
            merged_runs: get(&self.merged_runs),
            merged_requests: get(&self.merged_requests),
            prefix_runs: get(&self.prefix_runs),
        }
    }

    pub(super) fn finished(&self, answered: bool) {
        let counter = if answered {
            &self.answered
        } else {
            &self.failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn abandoned(&self) {
        self.abandoned.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn merged(&self, requests: usize) {
        self.merged_runs.fetch_add(1, Ordering::Relaxed);
        self.merged_requests
            .fetch_add(requests as u64, Ordering::Relaxed);
    }

    pub(super) fn prefix_run(&self) {
        self.prefix_runs.fetch_add(1, Ordering::Relaxed);
    }
}
