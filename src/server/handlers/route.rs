//! Which executor and scheduler serve one generation request.
//!
//! `blazr serve --model` loads one executor at start, or one per replica
//! when `inference.data_parallel` is above 1. A request for that model runs
//! on the loaded executor and its batching scheduler. With replicas, the
//! cache-aware router picks one. A request for any other model runs on the
//! executor the model scheduler returns, one sequence at a time.
//!
//! One copy of the start model stays evictable. The model scheduler owns its
//! executor and batching scheduler, and [`StartReplica`] only points at them,
//! so a request that names another model frees the device. When the start
//! model loads again, it runs one sequence at a time.

use std::sync::{Arc, Weak};

use crate::config::GenerationConfig;
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

/// The one copy of the start model, while the model scheduler keeps it loaded.
pub struct StartReplica {
    executor: Weak<Executor<ServerRuntime>>,
    scheduler: Option<Weak<dyn GenerationScheduler>>,
}

impl StartReplica {
    /// Point at `replica`, whose owner is the model scheduler.
    pub fn new(replica: &ServeReplica) -> Self {
        Self {
            executor: Arc::downgrade(&replica.executor),
            scheduler: replica.scheduler.as_ref().map(Arc::downgrade),
        }
    }

    /// Whether `executor` is this copy. The address stays reserved while this
    /// pointer lives, so a later load never compares equal.
    fn serves(&self, executor: &Arc<Executor<ServerRuntime>>) -> bool {
        std::ptr::eq(self.executor.as_ptr(), Arc::as_ptr(executor))
    }

    /// The batching scheduler, until an eviction stops it.
    fn scheduler(&self) -> Option<Arc<dyn GenerationScheduler>> {
        self.scheduler.as_ref()?.upgrade()
    }
}

/// The executors `blazr serve` loaded at start.
pub enum Serving {
    /// No model at start. Models load on demand and run one sequence at a time.
    OnDemand,
    /// The start model, evictable. See [`StartReplica`].
    Start(StartReplica),
    /// One copy of a model this value owns, behind a pinned model scheduler.
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
            Self::Start(start) if start.serves(&executor) => {
                let scheduler = start.scheduler();
                Route::single(executor, scheduler)
            }
            Self::Single(replica) if replica.serves(&executor) => {
                Route::single(executor, replica.scheduler.clone())
            }
            Self::Start(_) | Self::Single(_) | Self::OnDemand => Route::single(executor, None),
        }
    }

    /// The model the data-parallel replicas serve, with their count.
    /// `None` without data parallelism.
    pub fn replicated_model(&self) -> Option<(&str, usize)> {
        match self {
            Self::DataParallel(group) => Some((group.model_name(), group.num_replicas())),
            Self::Start(_) | Self::Single(_) | Self::OnDemand => None,
        }
    }

    /// The number of data-parallel replicas. 1 without data parallelism.
    pub fn replica_count(&self) -> usize {
        match self {
            Self::DataParallel(group) => group.num_replicas(),
            Self::Start(_) | Self::Single(_) | Self::OnDemand => 1,
        }
    }

    /// Requests in flight on each replica, by replica index.
    pub fn replica_load(&self) -> Vec<usize> {
        match self {
            Self::DataParallel(group) => {
                (0..group.num_replicas()).map(|i| group.active(i)).collect()
            }
            Self::Start(_) | Self::Single(_) | Self::OnDemand => Vec::new(),
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
            Self::Start(start) => start.scheduler().into_iter().collect(),
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

    /// Why this route's scheduler cannot run `gen_config`, or `None`. The
    /// handler answers it with 400. See [`GenerationScheduler::refusal`].
    pub fn refusal(&self, gen_config: &GenerationConfig) -> Option<String> {
        self.scheduler.as_ref()?.refusal(gen_config)
    }
}

/// A real paged engine thread over a model the scheduler owns.
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use super::*;
    use crate::engine::batch_test_rig::{rig_over, Rig, POOL};
    use crate::engine::test_support::{seeded_full_vocab_paged, FULL_VOCAB_CTX};
    use crate::engine::{EngineThread, FailRequests, Scheduler, ThreadedScheduler};

    /// The start model as `load_serving` leaves it, with its engine running.
    async fn started(models: &Scheduler<ServerRuntime>, path: std::path::PathBuf) -> Serving {
        let executor = seeded_full_vocab_paged(0x5eed, POOL).expect("executor");
        let served = models.register_base("m", path, executor).await;
        let Rig {
            scheduler, engine, ..
        } = rig_over(Arc::clone(&served), POOL, FULL_VOCAB_CTX).expect("rig");
        let owner: Arc<dyn FailRequests> = scheduler.clone();
        let thread = EngineThread::spawn("blazr-batch", owner, move |stop| engine.run_until(stop))
            .expect("thread");
        let batching: Arc<dyn GenerationScheduler> =
            Arc::new(ThreadedScheduler::new(scheduler, thread));
        let replica = ServeReplica {
            executor: served,
            scheduler: Some(Arc::clone(&batching)),
            device: "cpu".to_string(),
        };
        let start = StartReplica::new(&replica);
        models.attach_batching("m", batching).await;
        Serving::Start(start)
    }

    /// `make_room` frees the first model: the entry, the batching scheduler and the engine thread release the executor.
    #[tokio::test]
    async fn evicting_the_start_model_frees_its_executor() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir(dir.path().join("n")).expect("mkdir");
        let models = Scheduler::<ServerRuntime>::new(dir.path().to_path_buf(), Default::default());
        let serving = started(&models, dir.path().join("m")).await;
        let executor = models.get_executor("m").await.expect("loaded");
        let weak = Arc::downgrade(&executor);
        let route = serving.route(Arc::clone(&executor), &[1, 2, 3]);
        assert!(route.scheduler.is_some(), "the start model batches");
        drop(route);
        assert_eq!(serving.schedulers().len(), 1);
        drop(executor);

        // A request for another model evicts `m` before it loads.
        let other = models
            .load_base("n", |_| seeded_full_vocab_paged(0x5eed, POOL))
            .await
            .expect("second model loads");
        assert!(
            weak.upgrade().is_none(),
            "the evicted start model is still held"
        );
        assert!(serving.schedulers().is_empty(), "the engine stopped");
        assert!(serving.route(other, &[1]).scheduler.is_none());
    }

    /// A request in flight keeps the model until it ends, then the model frees.
    #[tokio::test]
    async fn a_route_in_flight_outlives_the_eviction() {
        let dir = tempfile::tempdir().expect("temp dir");
        let models = Scheduler::<ServerRuntime>::new(dir.path().to_path_buf(), Default::default());
        let serving = started(&models, dir.path().join("m")).await;
        let executor = models.get_executor("m").await.expect("loaded");
        let weak = Arc::downgrade(&executor);
        let route = serving.route(executor, &[1, 2, 3]);
        assert!(models.unload("m").await);
        assert!(weak.upgrade().is_some(), "the route still runs on it");
        assert!(route.scheduler.is_some());
        drop(route);
        assert!(weak.upgrade().is_none());
    }

    /// A held model stays through an unload and a load of another model.
    #[tokio::test]
    async fn a_held_model_is_never_evicted() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir(dir.path().join("n")).expect("mkdir");
        let models = Scheduler::<ServerRuntime>::new(dir.path().to_path_buf(), Default::default());
        let _serving = started(&models, dir.path().join("m")).await;
        models.hold("m").await;
        assert!(!models.unload("m").await);
        models
            .load_base("n", |_| seeded_full_vocab_paged(0x5eed, POOL))
            .await
            .expect("second model loads");
        assert!(models.is_loaded("m").await);
        assert!(models.is_loaded("n").await);
    }
}
