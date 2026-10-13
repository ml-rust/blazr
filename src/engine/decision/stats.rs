//! Counts of what a model's decision worker did, for tests and operators.

use std::sync::atomic::{AtomicU64, Ordering};

use boostr::inference::decision::StateCacheStats;
use serde::Serialize;

/// Running counts, shared by a worker and its service.
#[derive(Debug, Default)]
pub struct DecisionStats {
    answered: AtomicU64,
    failed: AtomicU64,
    abandoned: AtomicU64,
    merged_runs: AtomicU64,
    merged_requests: AtomicU64,
    prefix_runs: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    cache_evictions: AtomicU64,
    cache_entries: AtomicU64,
    cache_bytes: AtomicU64,
    cache_capacity: AtomicU64,
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
    /// The worker's prefix-state cache.
    pub prefix_cache: StateCacheStats,
}

/// The prefix-state cache of a running decision worker, as `/v1/models`
/// reports it in the model's `decision` object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PrefixCacheReport {
    /// Whether the worker keeps prefix states: a capacity above 0 and a
    /// family whose rows share a state prefix (not Clef).
    pub enabled: bool,
    /// The byte bound in MiB, `inference.decision.prefix_cache_mb`.
    pub capacity_mb: u64,
    /// Bytes the kept states take on the device.
    pub bytes: u64,
    /// States kept.
    pub entries: u64,
    /// Requests whose prefix state was kept.
    pub hits: u64,
    /// Requests whose prefix state was run.
    pub misses: u64,
    /// States removed: least recently used out for room or for a run's
    /// lanes, and every state when the model unloads.
    pub evictions: u64,
}

impl From<StateCacheStats> for PrefixCacheReport {
    fn from(s: StateCacheStats) -> Self {
        Self {
            enabled: s.capacity > 0,
            capacity_mb: (s.capacity >> 20) as u64,
            bytes: s.bytes as u64,
            entries: s.entries as u64,
            hits: s.hits,
            misses: s.misses,
            evictions: s.evictions,
        }
    }
}

impl DecisionStats {
    /// The counts now.
    pub fn snapshot(&self) -> DecisionCounts {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        let size = |c: &AtomicU64| usize::try_from(get(c)).unwrap_or(usize::MAX);
        DecisionCounts {
            answered: get(&self.answered),
            failed: get(&self.failed),
            abandoned: get(&self.abandoned),
            merged_runs: get(&self.merged_runs),
            merged_requests: get(&self.merged_requests),
            prefix_runs: get(&self.prefix_runs),
            prefix_cache: StateCacheStats {
                hits: get(&self.cache_hits),
                misses: get(&self.cache_misses),
                evictions: get(&self.cache_evictions),
                entries: size(&self.cache_entries),
                bytes: size(&self.cache_bytes),
                capacity: size(&self.cache_capacity),
            },
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

    /// Publish the worker's prefix-state cache as it stands.
    pub(super) fn set_prefix_cache(&self, s: &StateCacheStats) {
        let set = |c: &AtomicU64, v: u64| c.store(v, Ordering::Relaxed);
        set(&self.cache_hits, s.hits);
        set(&self.cache_misses, s.misses);
        set(&self.cache_evictions, s.evictions);
        set(&self.cache_entries, s.entries as u64);
        set(&self.cache_bytes, s.bytes as u64);
        set(&self.cache_capacity, s.capacity as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cache_counts_round_trip_and_report_in_mib() {
        let stats = DecisionStats::default();
        let cache = StateCacheStats {
            hits: 3,
            misses: 2,
            evictions: 1,
            entries: 1,
            bytes: 5 << 20,
            capacity: 64 << 20,
        };
        stats.set_prefix_cache(&cache);
        assert_eq!(stats.snapshot().prefix_cache, cache);
        let report = PrefixCacheReport::from(cache);
        assert!(report.enabled);
        assert_eq!((report.capacity_mb, report.bytes), (64, 5 << 20));
        assert_eq!((report.hits, report.misses, report.evictions), (3, 2, 1));
        let off = PrefixCacheReport::from(StateCacheStats::default());
        assert!(!off.enabled);
    }
}
