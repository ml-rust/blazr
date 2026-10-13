//! Cache-aware replica choice for data-parallel serving.
//!
//! Each replica keeps its own KV and prefix caches. The router remembers
//! which replica last served each prompt prefix. A new request goes to the
//! replica that served its longest known prefix, so that replica's cache is
//! warm. A request with no known prefix goes to the least-loaded replica.
//! A warm replica that carries far more requests than the least-loaded one
//! also loses the request to the least-loaded replica.
//!
//! Prefixes are prompt token ids, hashed at every [`PREFIX_STEP`] tokens up
//! to [`MAX_PREFIX_TOKENS`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Tokens between two recorded prefix lengths.
pub const PREFIX_STEP: usize = 16;

/// Longest prefix the router hashes.
pub const MAX_PREFIX_TOKENS: usize = 1024;

/// Prefixes remembered per replica before the oldest half is dropped.
const MAX_PREFIXES_PER_REPLICA: usize = 4096;

/// In-flight requests a warm replica can carry above the least-loaded one
/// and still take a request for its prefix.
pub const MAX_WARM_IMBALANCE: usize = 4;

/// Chooses a replica per request and counts each replica's requests.
#[derive(Debug)]
pub struct CacheAwareRouter {
    replicas: usize,
    prefixes: Mutex<PrefixTable>,
    active: Vec<AtomicUsize>,
    /// Start of the least-loaded scan, so ties spread across replicas.
    next_start: AtomicUsize,
}

#[derive(Debug, Default)]
struct PrefixTable {
    /// Prefix hash to (replica, last use).
    entries: HashMap<u64, (usize, u64)>,
    clock: u64,
}

/// Why the router chose a replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteReason {
    /// The replica served this many leading tokens of the prompt before.
    Warm(usize),
    /// No replica holds a usable prefix. This one carries the fewest requests.
    LeastLoaded,
}

impl CacheAwareRouter {
    /// A router over `replicas` replicas. 0 is treated as 1.
    pub fn new(replicas: usize) -> Self {
        let replicas = replicas.max(1);
        Self {
            replicas,
            prefixes: Mutex::new(PrefixTable::default()),
            active: (0..replicas).map(|_| AtomicUsize::new(0)).collect(),
            next_start: AtomicUsize::new(0),
        }
    }

    /// The number of replicas.
    pub fn replicas(&self) -> usize {
        self.replicas
    }

    /// The replica for a prompt, and why.
    pub fn route(&self, prompt_tokens: &[u32]) -> (usize, RouteReason) {
        let least = self.least_loaded();
        let warm = {
            let table = self.lock();
            prefix_hashes(prompt_tokens)
                .into_iter()
                .rev()
                .find_map(|(len, hash)| table.entries.get(&hash).map(|&(idx, _)| (idx, len)))
        };
        if let Some((idx, len)) = warm {
            if idx < self.replicas
                && self.active(idx) <= self.active(least).saturating_add(MAX_WARM_IMBALANCE)
            {
                return (idx, RouteReason::Warm(len));
            }
        }
        (least, RouteReason::LeastLoaded)
    }

    /// Remember that `replica` now holds the prefixes of `prompt_tokens`.
    pub fn record(&self, replica: usize, prompt_tokens: &[u32]) {
        if replica >= self.replicas {
            return;
        }
        let hashes = prefix_hashes(prompt_tokens);
        if hashes.is_empty() {
            return;
        }
        let mut table = self.lock();
        let cap = MAX_PREFIXES_PER_REPLICA.saturating_mul(self.replicas);
        if table.entries.len() + hashes.len() > cap {
            table.drop_oldest_half();
        }
        table.clock += 1;
        let now = table.clock;
        for (_, hash) in hashes {
            table.entries.insert(hash, (replica, now));
        }
    }

    /// Count one more request on `replica`.
    pub fn acquire(&self, replica: usize) {
        if let Some(count) = self.active.get(replica) {
            count.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Count one request fewer on `replica`.
    pub fn release(&self, replica: usize) {
        if let Some(count) = self.active.get(replica) {
            let _ = count.try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                Some(n.saturating_sub(1))
            });
        }
    }

    /// Requests in flight on `replica`.
    pub fn active(&self, replica: usize) -> usize {
        self.active
            .get(replica)
            .map_or(0, |count| count.load(Ordering::Acquire))
    }

    /// The replica with the fewest requests in flight. Ties go to the first
    /// one after a start index that moves on every call.
    fn least_loaded(&self) -> usize {
        let start = self.next_start.fetch_add(1, Ordering::Relaxed) % self.replicas;
        (0..self.replicas)
            .map(|offset| (start + offset) % self.replicas)
            .min_by_key(|&idx| self.active(idx))
            .unwrap_or(0)
    }

    fn lock(&self) -> MutexGuard<'_, PrefixTable> {
        self.prefixes.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl PrefixTable {
    /// Keep the newer half of the entries.
    fn drop_oldest_half(&mut self) {
        let mut stamps: Vec<u64> = self.entries.values().map(|&(_, at)| at).collect();
        if stamps.is_empty() {
            return;
        }
        let mid = stamps.len() / 2;
        let (_, &mut cutoff, _) = stamps.select_nth_unstable(mid);
        self.entries.retain(|_, &mut (_, at)| at > cutoff);
    }
}

/// `(length, hash)` for each prefix of `tokens` at a multiple of
/// [`PREFIX_STEP`], shortest first, up to [`MAX_PREFIX_TOKENS`].
fn prefix_hashes(tokens: &[u32]) -> Vec<(usize, u64)> {
    let limit = tokens.len().min(MAX_PREFIX_TOKENS);
    let mut out = Vec::with_capacity(limit / PREFIX_STEP);
    // FNV-1a over the little-endian bytes of each token id.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, token) in tokens[..limit].iter().enumerate() {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        let len = i + 1;
        if len % PREFIX_STEP == 0 {
            // Mix in the length so prefixes of different lengths differ.
            out.push((len, hash ^ (len as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| seed * 10_000 + i).collect()
    }

    #[test]
    fn a_short_prompt_has_no_prefix() {
        assert!(prefix_hashes(&prompt(1, PREFIX_STEP - 1)).is_empty());
        assert_eq!(prefix_hashes(&prompt(1, PREFIX_STEP * 3 + 2)).len(), 3);
        let long = prefix_hashes(&prompt(1, MAX_PREFIX_TOKENS * 2));
        assert_eq!(long.len(), MAX_PREFIX_TOKENS / PREFIX_STEP);
    }

    #[test]
    fn a_shared_prefix_routes_to_the_replica_that_served_it() {
        let router = CacheAwareRouter::new(3);
        let system = prompt(7, 64);
        router.record(2, &system);

        let mut follow_up = system.clone();
        follow_up.extend(prompt(9, 40));
        assert_eq!(router.route(&follow_up), (2, RouteReason::Warm(64)));
    }

    #[test]
    fn the_longest_known_prefix_wins() {
        let router = CacheAwareRouter::new(2);
        let base = prompt(3, 32);
        let mut longer = base.clone();
        longer.extend(prompt(4, 32));
        // Replica 1 served the long prompt. Replica 0 then served the base,
        // so it holds the newest copy of the two short prefixes.
        router.record(1, &longer);
        router.record(0, &base);
        assert_eq!(router.route(&longer), (1, RouteReason::Warm(64)));
        let mut other = base.clone();
        other.extend(prompt(5, 32));
        assert_eq!(router.route(&other), (0, RouteReason::Warm(32)));
    }

    #[test]
    fn an_unknown_prefix_goes_to_the_least_loaded_replica() {
        let router = CacheAwareRouter::new(3);
        router.acquire(0);
        router.acquire(0);
        router.acquire(2);
        assert_eq!(router.route(&prompt(1, 64)), (1, RouteReason::LeastLoaded));
        router.release(0);
        router.release(0);
        router.acquire(1);
        assert_eq!(router.route(&prompt(1, 64)), (0, RouteReason::LeastLoaded));
    }

    #[test]
    fn an_overloaded_warm_replica_loses_to_the_least_loaded() {
        let router = CacheAwareRouter::new(2);
        let p = prompt(1, 48);
        router.record(0, &p);
        for _ in 0..MAX_WARM_IMBALANCE {
            router.acquire(0);
        }
        assert_eq!(router.route(&p), (0, RouteReason::Warm(48)));
        router.acquire(0);
        assert_eq!(router.route(&p), (1, RouteReason::LeastLoaded));
    }

    #[test]
    fn idle_ties_spread_over_the_replicas() {
        let router = CacheAwareRouter::new(3);
        let picks: Vec<usize> = (0..3).map(|_| router.route(&[]).0).collect();
        let mut sorted = picks.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2], "{picks:?}");
    }

    #[test]
    fn release_never_goes_below_zero() {
        let router = CacheAwareRouter::new(1);
        router.release(0);
        assert_eq!(router.active(0), 0);
    }

    #[test]
    fn a_full_table_keeps_the_newest_prefixes() {
        let router = CacheAwareRouter::new(1);
        let per_prompt = MAX_PREFIX_TOKENS / PREFIX_STEP;
        let prompts = MAX_PREFIXES_PER_REPLICA / per_prompt + 4;
        for seed in 0..prompts as u32 {
            router.record(0, &prompt(seed, MAX_PREFIX_TOKENS));
        }
        let size = router.lock().entries.len();
        assert!(size <= MAX_PREFIXES_PER_REPLICA, "{size}");
        let newest = prompt(prompts as u32 - 1, MAX_PREFIX_TOKENS);
        assert_eq!(
            router.route(&newest),
            (0, RouteReason::Warm(MAX_PREFIX_TOKENS))
        );
    }
}
