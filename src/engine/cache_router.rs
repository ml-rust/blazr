//! Cache-aware request router for data-parallel inference.
//!
//! Routes requests to the replica most likely to have warm KV cache
//! for the request's prompt prefix, reducing redundant prefill computation.
//!
//! Strategy: Track a hash of the prompt prefix per replica. Route new requests
//! to the replica that served the most similar prefix. Falls back to
//! least-loaded replica if no prefix match is found.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use tokio::sync::RwLock;

use super::Executor;

/// Maximum number of prefix entries to track per replica
const MAX_PREFIX_ENTRIES: usize = 256;

/// Minimum prefix length to consider for caching (in chars)
const MIN_PREFIX_LEN: usize = 32;

/// Cache-aware router that tracks prompt prefixes served by each replica
pub struct CacheAwareRouter<R: Runtime<DType = DType>> {
    replicas: Vec<Arc<Executor<R>>>,
    /// Map from prefix hash → replica index
    prefix_map: RwLock<HashMap<u64, usize>>,
    /// Per-replica active request count (for load balancing)
    active_counts: Vec<AtomicUsize>,
}

impl<R: Runtime<DType = DType>> CacheAwareRouter<R>
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
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    pub fn new(replicas: Vec<Arc<Executor<R>>>) -> Self {
        let active_counts = (0..replicas.len()).map(|_| AtomicUsize::new(0)).collect();
        Self {
            replicas,
            prefix_map: RwLock::new(HashMap::new()),
            active_counts,
        }
    }

    /// Route a request to the best replica.
    ///
    /// 1. If prompt_prefix is provided and a replica has served it before, route there.
    /// 2. Otherwise, route to the replica with fewest active requests (least-loaded).
    pub fn route(&self, prompt_prefix: Option<&str>) -> Arc<Executor<R>> {
        // Try prefix match
        if let Some(prefix) = prompt_prefix {
            if prefix.len() >= MIN_PREFIX_LEN {
                let hash = hash_prefix(prefix);
                // Use try_read to avoid blocking — fall back to least-loaded on contention
                if let Ok(map) = self.prefix_map.try_read() {
                    if let Some(&replica_idx) = map.get(&hash) {
                        if replica_idx < self.replicas.len() {
                            return Arc::clone(&self.replicas[replica_idx]);
                        }
                    }
                }
            }
        }

        // Least-loaded fallback
        let idx = self.least_loaded();
        Arc::clone(&self.replicas[idx])
    }

    /// Record that a replica served a request with a given prefix.
    pub fn record(&self, replica_idx: usize, prompt_prefix: &str) {
        if prompt_prefix.len() < MIN_PREFIX_LEN || replica_idx >= self.replicas.len() {
            return;
        }

        let hash = hash_prefix(prompt_prefix);
        // Use try_write to avoid blocking the hot path
        if let Ok(mut map) = self.prefix_map.try_write() {
            // Evict oldest entries if map is too large
            if map.len() >= MAX_PREFIX_ENTRIES * self.replicas.len() {
                // Simple eviction: clear half the map
                let keys: Vec<u64> = map.keys().copied().collect();
                for key in keys.iter().take(keys.len() / 2) {
                    map.remove(key);
                }
            }
            map.insert(hash, replica_idx);
        }
    }

    /// Find the replica with the fewest active requests
    fn least_loaded(&self) -> usize {
        self.active_counts
            .iter()
            .enumerate()
            .min_by_key(|(_, count)| count.load(Ordering::Relaxed))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Increment active count for a replica
    pub fn acquire(&self, replica_idx: usize) {
        if replica_idx < self.active_counts.len() {
            self.active_counts[replica_idx].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Decrement active count for a replica
    pub fn release(&self, replica_idx: usize) {
        if replica_idx < self.active_counts.len() {
            self.active_counts[replica_idx].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Hash a prompt prefix for cache lookup.
/// Uses the first MIN_PREFIX_LEN..512 chars to balance specificity vs collision rate.
fn hash_prefix(prefix: &str) -> u64 {
    let effective = &prefix[..prefix.len().min(512)];
    // FNV-1a hash
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in effective.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_prefix_deterministic() {
        let h1 = hash_prefix("Hello, this is a test prompt that is longer than 32 chars");
        let h2 = hash_prefix("Hello, this is a test prompt that is longer than 32 chars");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_prefix_different() {
        let h1 = hash_prefix("Hello, this is a test prompt that is longer than 32 chars");
        let h2 = hash_prefix("Different prompt that is also longer than thirty two characters");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_prefix_truncation() {
        let long = "a".repeat(1000);
        let also_long = format!("{}b", "a".repeat(512));
        // Both should hash only first 512 chars
        let h1 = hash_prefix(&long);
        let h2 = hash_prefix(&also_long);
        // They should be equal since first 512 chars are the same
        assert_eq!(h1, h2);
    }
}
