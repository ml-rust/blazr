//! The decision worker's prefix-state cache: boostr's `PrefixStateCache`
//! keyed by the weights the worker runs, the renderer family and the
//! prefix ids, with its counts published to the service and the metrics.
//!
//! Clef keeps nothing: its head reads every hidden row of the record, so
//! the whole record is its plan's prefix and no state outlives a request.

use boostr::inference::decision::run::CachedPrefix;
use boostr::inference::decision::{
    cached_prefix, DecisionBackbone, DecisionPlan, PrefixStateCache, RunnerConfig, StateCacheStats,
};
use boostr::model::ModelClient;
use boostr::{DType, Runtime};

use super::stats::DecisionStats;
use crate::model::decision_template::{Family, Readout};
use crate::telemetry::decision::record_prefix_cache;

/// What a kept prefix state depends on. Equal keys give equal states.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct PrefixKey {
    /// The address of the loaded weights the worker runs. A LoRA or Kev
    /// variant is a model of its own, at its own address, so variants never
    /// share a state. The worker's `Weak` keeps the allocation, so no other
    /// model takes the address while the worker runs.
    model: usize,
    /// The renderer family: the prompt layout the ids follow.
    family: Family,
    /// The plan's prefix ids: the state, cut at the family's state
    /// boundary, when the request plans there.
    tokens: Vec<u32>,
}

/// One worker's prefix states and where it reports them.
pub(super) struct WorkerCache<R: Runtime<DType = DType>> {
    cache: PrefixStateCache<PrefixKey, R>,
    model: usize,
    family: Family,
}

impl<R: Runtime<DType = DType>> WorkerCache<R>
where
    R::Client: ModelClient<R>,
{
    /// A cache of `capacity` bytes for the weights at address `model`
    /// served as `family`. Clef, and `capacity` 0, keep nothing.
    pub(super) fn new(capacity: usize, model: usize, family: Family) -> Self {
        let capacity = match family.readout() {
            Readout::JointSchema => 0,
            Readout::LabelToken | Readout::Pointer => capacity,
        };
        Self {
            cache: PrefixStateCache::new(capacity),
            model,
            family,
        }
    }

    /// Whether any state is kept.
    pub(super) fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// The counts and holdings now.
    pub(super) fn stats(&self) -> StateCacheStats {
        self.cache.stats()
    }

    fn key(&self, plan: &DecisionPlan) -> PrefixKey {
        PrefixKey {
            model: self.model,
            family: self.family,
            tokens: plan.prefix.tokens().to_vec(),
        }
    }

    /// The prefix state `plan` runs over, as boostr's `cached_prefix` gives
    /// it. `None`: the run prefills its own prefix, if it has one.
    ///
    /// # Errors
    ///
    /// boostr's plan, budget, cancel and backbone errors.
    pub(super) fn prefix<C, B>(
        &mut self,
        client: &C,
        backbone: &B,
        plan: &DecisionPlan,
        cfg: &RunnerConfig,
    ) -> boostr::error::Result<Option<CachedPrefix<R>>>
    where
        C: ModelClient<R>,
        B: DecisionBackbone<R>,
    {
        let key = self.key(plan);
        cached_prefix(&mut self.cache, client, backbone, plan, key, cfg)
    }

    /// Drop every state: the weights unloaded, or a run panicked and the
    /// cache may be inconsistent.
    pub(super) fn release(&mut self) -> usize {
        self.cache.clear()
    }
}

/// Publish what the cache did since `before` to `stats` and to the
/// metrics of `model`.
pub(super) fn publish(
    model: &str,
    stats: &DecisionStats,
    before: &StateCacheStats,
    now: &StateCacheStats,
) {
    if before == now {
        return;
    }
    stats.set_prefix_cache(now);
    record_prefix_cache(
        model,
        now.hits.saturating_sub(before.hits),
        now.misses.saturating_sub(before.misses),
        now.evictions.saturating_sub(before.evictions),
        now.bytes,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::CpuRuntime;

    fn key(model: usize, family: Family, tokens: &[u32]) -> PrefixKey {
        PrefixKey {
            model,
            family,
            tokens: tokens.to_vec(),
        }
    }

    /// A state is shared only by the same weights, the same family and the
    /// same prefix ids: a LoRA variant runs other weights.
    #[test]
    fn keys_tell_weights_families_and_ids_apart() {
        let base = key(0x1000, Family::Decider, &[1, 2, 3]);
        assert_eq!(base, key(0x1000, Family::Decider, &[1, 2, 3]));
        assert_ne!(base, key(0x2000, Family::Decider, &[1, 2, 3]), "a variant");
        assert_ne!(base, key(0x1000, Family::Jebadiah, &[1, 2, 3]));
        assert_ne!(base, key(0x1000, Family::Decider, &[1, 2, 4]));
        assert_ne!(base, key(0x1000, Family::Decider, &[1, 2]));
    }

    /// Clef and a 0 capacity keep nothing; the label-token and pointer
    /// families keep up to the capacity.
    #[test]
    fn clef_and_zero_capacity_are_off() {
        let on = WorkerCache::<CpuRuntime>::new(1 << 20, 1, Family::Kev);
        assert_eq!(on.stats().capacity, 1 << 20);
        let clef = WorkerCache::<CpuRuntime>::new(1 << 20, 1, Family::Clef);
        assert_eq!(clef.stats().capacity, 0);
        let off = WorkerCache::<CpuRuntime>::new(0, 1, Family::Decider);
        assert_eq!(off.stats(), StateCacheStats::default());
        assert!(off.is_empty());
    }

    #[test]
    fn publish_skips_an_unchanged_cache() {
        let stats = DecisionStats::default();
        let now = StateCacheStats {
            hits: 1,
            capacity: 8,
            ..StateCacheStats::default()
        };
        publish("d", &stats, &now, &now);
        assert_eq!(stats.snapshot().prefix_cache, StateCacheStats::default());
        publish("d", &stats, &StateCacheStats::default(), &now);
        assert_eq!(stats.snapshot().prefix_cache, now);
    }
}
