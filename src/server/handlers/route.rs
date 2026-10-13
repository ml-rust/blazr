//! Which executor and scheduler serve one generation request.
//!
//! `blazr serve --model` loads one executor at start, or one per replica
//! when `inference.data_parallel` is above 1. A request for that model runs
//! on the loaded executor and its batching scheduler. With replicas, the
//! cache-aware router picks one. A request for any other model runs on the
//! executor the model scheduler returns, one sequence at a time.

use std::sync::Arc;

use crate::engine::{
    route_through, DataParallelGroup, Executor, GenerationScheduler, ReplicaLease,
};

use super::state::ServerRuntime;

/// One loaded copy of the start model.
pub struct ServeReplica {
    pub executor: Arc<Executor<ServerRuntime>>,
    /// The batching scheduler on this copy. `None` generates one sequence
    /// at a time.
    pub scheduler: Option<Arc<dyn GenerationScheduler>>,
    /// The device the copy runs on, such as `cuda:1` or `cpu`.
    pub device: String,
}

impl ServeReplica {
    /// Whether `executor` is this copy.
    pub fn serves(&self, executor: &Arc<Executor<ServerRuntime>>) -> bool {
        Arc::ptr_eq(&self.executor, executor)
    }
}

/// The executors `blazr serve` loaded at start.
pub enum Serving {
    /// No model at start. Models load on demand and run one sequence at a time.
    OnDemand,
    /// One copy of the start model.
    Single(ServeReplica),
    /// `inference.data_parallel` copies of the start model.
    DataParallel(Arc<DataParallelGroup<ServeReplica>>),
}

/// The executor and scheduler for one request.
///
/// Hold the route until the request ends. With replicas, it counts the
/// request against the chosen replica until it drops.
pub struct Route {
    pub executor: Arc<Executor<ServerRuntime>>,
    pub scheduler: Option<Arc<dyn GenerationScheduler>>,
    _lease: Option<ReplicaLease<ServeReplica>>,
}

impl Serving {
    /// The route for a request that `executor` resolved to.
    ///
    /// `prompt_tokens` steers the replica choice toward a warm prefix cache.
    pub fn route(&self, executor: Arc<Executor<ServerRuntime>>, prompt_tokens: &[u32]) -> Route {
        match self {
            Self::DataParallel(group) => {
                match route_through(Some(group), |r| r.serves(&executor), prompt_tokens) {
                    Some(lease) => Route {
                        executor: Arc::clone(&lease.replica().executor),
                        scheduler: lease.replica().scheduler.clone(),
                        _lease: Some(lease),
                    },
                    None => Route::single(executor, None),
                }
            }
            Self::Single(replica) if replica.serves(&executor) => {
                Route::single(executor, replica.scheduler.clone())
            }
            Self::Single(_) | Self::OnDemand => Route::single(executor, None),
        }
    }

    /// The model the data-parallel replicas serve, with their count.
    /// `None` without data parallelism.
    pub fn replicated_model(&self) -> Option<(&str, usize)> {
        match self {
            Self::DataParallel(group) => Some((group.model_name(), group.num_replicas())),
            Self::Single(_) | Self::OnDemand => None,
        }
    }

    /// The number of data-parallel replicas. 1 without data parallelism.
    pub fn replica_count(&self) -> usize {
        match self {
            Self::DataParallel(group) => group.num_replicas(),
            Self::Single(_) | Self::OnDemand => 1,
        }
    }

    /// Requests in flight on each replica, by replica index.
    pub fn replica_load(&self) -> Vec<usize> {
        match self {
            Self::DataParallel(group) => {
                (0..group.num_replicas()).map(|i| group.active(i)).collect()
            }
            Self::Single(_) | Self::OnDemand => Vec::new(),
        }
    }

    /// Every batching scheduler, for the metrics endpoint.
    pub fn schedulers(&self) -> Vec<Arc<dyn GenerationScheduler>> {
        match self {
            Self::DataParallel(group) => group
                .replicas()
                .iter()
                .filter_map(|r| r.scheduler.clone())
                .collect(),
            Self::Single(replica) => replica.scheduler.iter().cloned().collect(),
            Self::OnDemand => Vec::new(),
        }
    }
}

impl Route {
    fn single(
        executor: Arc<Executor<ServerRuntime>>,
        scheduler: Option<Arc<dyn GenerationScheduler>>,
    ) -> Self {
        Self {
            executor,
            scheduler,
            _lease: None,
        }
    }
}
