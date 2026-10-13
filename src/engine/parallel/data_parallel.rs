//! Data parallelism: one model loaded once per replica, requests spread
//! across the replicas.
//!
//! A replica is whatever serves one copy of the model. `blazr serve` uses
//! an executor plus its batching scheduler. Tests use a mock. The
//! [`CacheAwareRouter`] picks the replica for each request. The returned
//! [`ReplicaLease`] counts the request against that replica until it drops.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use super::cache_router::{CacheAwareRouter, RouteReason};

/// Replicas of one model and the router that spreads requests over them.
#[derive(Debug)]
pub struct DataParallelGroup<T> {
    model_name: String,
    replicas: Vec<T>,
    router: CacheAwareRouter,
}

impl<T> DataParallelGroup<T> {
    /// A group over `replicas` of `model_name`.
    ///
    /// # Errors
    ///
    /// An error when `replicas` is empty.
    pub fn new(model_name: String, replicas: Vec<T>) -> Result<Self> {
        if replicas.is_empty() {
            return Err(anyhow!(
                "a data-parallel group of '{model_name}' got no replica. Set \
                 inference.data_parallel to 1 or more"
            ));
        }
        let router = CacheAwareRouter::new(replicas.len());
        Ok(Self {
            model_name,
            replicas,
            router,
        })
    }

    /// The model every replica serves.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// The number of replicas.
    pub fn num_replicas(&self) -> usize {
        self.replicas.len()
    }

    /// Every replica, in device order.
    pub fn replicas(&self) -> &[T] {
        &self.replicas
    }

    /// Requests in flight on replica `index`.
    pub fn active(&self, index: usize) -> usize {
        self.router.active(index)
    }

    /// Pick the replica for `prompt_tokens` and count the request on it.
    ///
    /// The chosen replica is recorded as the holder of the prompt's
    /// prefixes, so a later request that shares them goes there too.
    pub fn select(self: &Arc<Self>, prompt_tokens: &[u32]) -> ReplicaLease<T> {
        let (index, reason) = self.router.route(prompt_tokens);
        self.router.acquire(index);
        self.router.record(index, prompt_tokens);
        ReplicaLease {
            group: Arc::clone(self),
            index,
            reason,
        }
    }
}

/// Route through `group` when it serves the request's model.
///
/// `serves` tells whether a replica serves the model the request named. It
/// is read on the first replica. `None` means the request takes the
/// single-executor path.
pub fn route_through<T>(
    group: Option<&Arc<DataParallelGroup<T>>>,
    serves: impl FnOnce(&T) -> bool,
    prompt_tokens: &[u32],
) -> Option<ReplicaLease<T>> {
    let group = group?;
    let first = group.replicas.first()?;
    serves(first).then(|| group.select(prompt_tokens))
}

/// One request's hold on a replica. Dropping it ends the request there.
#[derive(Debug)]
pub struct ReplicaLease<T> {
    group: Arc<DataParallelGroup<T>>,
    index: usize,
    reason: RouteReason,
}

impl<T> ReplicaLease<T> {
    /// The chosen replica's index.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Why the router chose this replica.
    pub fn reason(&self) -> RouteReason {
        self.reason
    }

    /// The chosen replica.
    pub fn replica(&self) -> &T {
        &self.group.replicas[self.index]
    }
}

impl<T> Drop for ReplicaLease<T> {
    fn drop(&mut self) {
        self.group.router.release(self.index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in for an executor: a device name and a model name.
    #[derive(Debug)]
    struct MockReplica {
        device: &'static str,
        model: &'static str,
    }

    fn group(n: usize) -> Arc<DataParallelGroup<MockReplica>> {
        const DEVICES: [&str; 4] = ["cpu:0", "cpu:1", "cpu:2", "cpu:3"];
        let replicas = DEVICES[..n]
            .iter()
            .map(|&device| MockReplica {
                device,
                model: "tiny",
            })
            .collect();
        Arc::new(DataParallelGroup::new("tiny".into(), replicas).expect("group"))
    }

    fn prompt(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| seed * 10_000 + i).collect()
    }

    #[test]
    fn an_empty_group_is_an_error() {
        let err = DataParallelGroup::<MockReplica>::new("m".into(), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("inference.data_parallel"), "{err}");
    }

    #[test]
    fn concurrent_cold_requests_spread_least_loaded() {
        let group = group(3);
        let leases: Vec<_> = (0..3).map(|i| group.select(&prompt(i, 64))).collect();
        let mut used: Vec<usize> = leases.iter().map(ReplicaLease::index).collect();
        used.sort_unstable();
        assert_eq!(used, vec![0, 1, 2]);
        assert!(leases
            .iter()
            .all(|l| l.reason() == RouteReason::LeastLoaded));
        // A fourth cold request joins a replica with one request, not two.
        let fourth = group.select(&prompt(9, 64));
        assert_eq!(group.active(fourth.index()), 2);
    }

    #[test]
    fn a_follow_up_with_a_shared_prefix_returns_to_its_warm_replica() {
        let group = group(4);
        let conversation = prompt(5, 96);
        let first = group.select(&conversation);
        let warm = first.index();
        drop(first);

        // Load the warm replica a little. The prefix still wins.
        let _busy = group.select(&conversation);
        let mut next_turn = conversation.clone();
        next_turn.extend(prompt(6, 30));
        let follow_up = group.select(&next_turn);
        assert_eq!(follow_up.index(), warm);
        assert_eq!(follow_up.reason(), RouteReason::Warm(96));
        assert_eq!(
            follow_up.replica().device,
            ["cpu:0", "cpu:1", "cpu:2", "cpu:3"][warm]
        );
    }

    #[test]
    fn a_lease_counts_until_it_drops() {
        let group = group(2);
        let lease = group.select(&prompt(1, 32));
        let index = lease.index();
        assert_eq!(group.active(index), 1);
        drop(lease);
        assert_eq!(group.active(index), 0);
    }

    #[test]
    fn route_through_takes_the_group_only_for_its_model() {
        let group = group(2);
        let tokens = prompt(1, 32);
        assert!(route_through::<MockReplica>(None, |_| true, &tokens).is_none());
        assert!(route_through(Some(&group), |r| r.model == "other", &tokens).is_none());
        let lease = route_through(Some(&group), |r| r.model == "tiny", &tokens);
        assert!(lease.is_some_and(|l| l.replica().model == "tiny"));
    }
}
