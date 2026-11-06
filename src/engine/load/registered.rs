//! The load side of [`Scheduler`]: one load per model name at a time, run
//! off the async executor, registered whole or not at all.
//!
//! A load and its registration run together on a blocking thread, and
//! that thread holds the name's [`LoadGate`] to the end. A request that
//! stops waiting leaves the load to finish and register, so the next
//! request finds the model loaded and loads nothing again. A load that
//! fails or panics registers nothing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use boostr::{DType, Runtime};

use super::blocking::{has_blocking_pool, run_blocking, BlockingStopped};
use super::gate::LoadGate;
use crate::engine::lora::eviction;
use crate::engine::scheduler::{residents, ModelEntry};
use crate::engine::{Executor, ModelNotFound, Scheduler};
use crate::loader::same_weights;

/// The loaded models of a scheduler, by name.
pub(crate) type Models<R> = HashMap<String, ModelEntry<R>>;

/// Take the least recently used base models out of `models`, each with its LoRA
/// variants first, until fewer than `max_loaded` bases are loaded. Only base
/// models count against the limit.
///
/// Dropping an entry can join its engine thread. The caller releases the models
/// lock first, then [`release`]s what this returns.
pub(crate) fn make_room<R: Runtime<DType = DType>>(
    models: &mut Models<R>,
    max_loaded: usize,
) -> Vec<ModelEntry<R>> {
    let mut evicted = Vec::new();
    loop {
        let order = {
            let residents = residents(models);
            if eviction::base_count(&residents) < max_loaded {
                break;
            }
            let Some(lru) = eviction::lru_base(&residents) else {
                break;
            };
            eviction::unload_order(&residents, &lru)
        };
        for name in order {
            tracing::info!("Evicting model: {}", name);
            evicted.extend(models.remove(&name));
        }
        metrics::counter!("blazr_scheduler_evictions_total").increment(1);
    }
    evicted
}

/// Drop `evicted` on a blocking thread. An entry's drop joins its engine thread
/// and frees device memory, so it never runs on an async worker or under the
/// models lock. Returns when both end.
pub(crate) async fn release<R: Runtime<DType = DType>>(evicted: Vec<ModelEntry<R>>) {
    if evicted.is_empty() {
        return;
    }
    if let Err(e) = run_blocking(move || drop(evicted)).await {
        tracing::error!("dropping an evicted model stopped: {e}");
    }
}

impl<R: Runtime<DType = DType>> Scheduler<R> {
    /// The key of the loaded model `name` names: `name` itself, else the base
    /// model with the same weights file. A server started on
    /// `--model /abs/q.gguf` holds that key, and a client sends the listed id
    /// `q.gguf`. `None` when no loaded model matches.
    pub(crate) async fn loaded_key(&self, name: &str) -> Option<String> {
        let bases: Vec<(String, PathBuf)> = {
            let models = self.models.read().await;
            if models.contains_key(name) {
                return Some(name.to_string());
            }
            models
                .iter()
                .filter(|(_, entry)| entry.variant_of.is_none())
                .map(|(key, entry)| (key.clone(), entry.path.clone()))
                .collect()
        };
        if bases.is_empty() {
            return None;
        }
        let names = self.operator_names.clone();
        let model_dir = self.model_dir.clone();
        let target = name.to_string();
        run_blocking(move || {
            let path = names.resolve(&model_dir, &target)?;
            bases
                .into_iter()
                .find(|(_, base)| same_weights(base, &path))
                .map(|(key, _)| key)
        })
        .await
        .ok()
        .flatten()
    }

    /// The executor of the loaded model `name` names, with its access set to
    /// now. See [`Self::loaded_key`].
    pub(crate) async fn served_copy(&self, name: &str) -> Option<Arc<Executor<R>>> {
        if let Some(executor) = self.touch(name).await {
            return Some(executor);
        }
        let key = self.loaded_key(name).await?;
        self.touch(&key).await
    }

    /// The executor of the loaded `name`, with its last access set to now.
    /// A variant's access counts as one for its base too, so the base is
    /// never the least recently used model while its variant serves.
    pub(crate) async fn touch(&self, name: &str) -> Option<Arc<Executor<R>>> {
        let mut models = self.models.write().await;
        let now = Instant::now();
        let entry = models.get_mut(name)?;
        entry.last_accessed = now;
        let executor = Arc::clone(&entry.executor);
        if let Some(base) = entry.variant_of.clone() {
            if let Some(base) = models.get_mut(&base) {
                base.last_accessed = now;
            }
        }
        Some(executor)
    }

    /// The executor of the base model `name`, started by `load` from the
    /// model's path when `name` is not loaded. One load of `name` runs at
    /// a time: a caller that finds one in flight waits, then gets the
    /// executor it registered. A name for the weights of a loaded model gets
    /// that model's executor and loads nothing. LRU bases are evicted before
    /// the load, to free the device, and again at registration, for loads of
    /// other models that finished meanwhile. Each eviction frees its memory
    /// before the next step runs.
    ///
    /// # Errors
    ///
    /// [`ModelNotFound`] when `name` resolves to no path, the error of
    /// `load`, or a load that panicked. Each registers nothing.
    pub(crate) async fn load_base<F>(&self, name: &str, load: F) -> Result<Arc<Executor<R>>>
    where
        F: FnOnce(&Path) -> Result<Executor<R>> + Send + 'static,
    {
        let gate = self.gates.acquire(name).await;
        if let Some(executor) = self.served_copy(name).await {
            return Ok(executor);
        }
        let names = self.operator_names.clone();
        let model_dir = self.model_dir.clone();
        let target = name.to_string();
        let path = run_blocking(move || names.resolve(&model_dir, &target))
            .await
            .map_err(|e| anyhow!("model '{name}': the model lookup stopped: {e}"))?
            .ok_or_else(|| ModelNotFound::new(name))?;
        let evicted = make_room(&mut *self.models.write().await, self.max_loaded);
        release(evicted).await;

        let key = name.to_string();
        let max_loaded = self.max_loaded;
        self.run_registered(
            gate,
            move || load(&path).map(|executor| (executor, path)),
            move |(executor, path), models, evicted| {
                evicted.extend(make_room(models, max_loaded));
                let executor = Arc::new(executor);
                models.insert(
                    key,
                    ModelEntry {
                        executor: Arc::clone(&executor),
                        last_accessed: Instant::now(),
                        path,
                        keep_alive: None,
                        variant_of: None,
                        batching: None,
                        held: false,
                    },
                );
                metrics::counter!("blazr_scheduler_loads_total").increment(1);
                executor
            },
            |e| anyhow!("model '{name}': the load stopped: {e}"),
        )
        .await
    }

    /// Run `work`, then `register` its value into the loaded models under
    /// the models lock, both on a blocking thread that holds `gate` until
    /// both end. A caller that stops waiting leaves both to finish. A
    /// `work` error or panic skips `register`. The entries `register` puts in
    /// its last argument drop after the lock is released.
    ///
    /// With no blocking pool (see
    /// [`has_blocking_pool`](super::blocking::has_blocking_pool)), `work`
    /// runs on the calling thread, and `register` takes the lock without
    /// blocking the caller's executor.
    ///
    /// # Errors
    ///
    /// The error of `work`, or `stopped` over a panic in `work`.
    pub(crate) async fn run_registered<T, O, E>(
        &self,
        gate: LoadGate,
        work: impl FnOnce() -> Result<T, E> + Send + 'static,
        register: impl FnOnce(T, &mut Models<R>, &mut Vec<ModelEntry<R>>) -> O + Send + 'static,
        stopped: impl FnOnce(BlockingStopped) -> E,
    ) -> Result<O, E>
    where
        T: Send + 'static,
        O: Send + 'static,
        E: Send + 'static,
    {
        if has_blocking_pool() {
            let models = Arc::clone(&self.models);
            return run_blocking(move || {
                let _gate = gate;
                let value = work()?;
                let mut evicted = Vec::new();
                let output = register(value, &mut models.blocking_write(), &mut evicted);
                drop(evicted);
                Ok(output)
            })
            .await
            .map_err(stopped)?;
        }
        let value = run_blocking(work).await.map_err(stopped)??;
        let mut evicted = Vec::new();
        let output = register(value, &mut *self.models.write().await, &mut evicted);
        release(evicted).await;
        drop(gate);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use boostr::{CpuDevice, CpuRuntime};

    use crate::engine::test_support::tiny_executor;

    /// How long a fake load holds its blocking thread.
    const LOAD: Duration = Duration::from_millis(400);
    /// A ticker's period while a load is in flight.
    const TICK: Duration = Duration::from_millis(10);

    /// A scheduler over a model directory that holds the model `m`.
    fn scheduler_with_m() -> (tempfile::TempDir, Scheduler<CpuRuntime>) {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir(dir.path().join("m")).expect("mkdir");
        let scheduler = Scheduler::new(dir.path().to_path_buf(), CpuDevice::new());
        (dir, scheduler)
    }

    /// A load that holds its thread for [`LOAD`], counting its runs.
    fn slow_load(
        runs: &Arc<AtomicUsize>,
    ) -> impl FnOnce(&Path) -> Result<Executor<CpuRuntime>> + Send + 'static {
        let runs = Arc::clone(runs);
        move |_: &Path| {
            runs.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(LOAD);
            tiny_executor()
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_load_in_flight_leaves_the_runtime_serving() {
        let (_dir, scheduler) = scheduler_with_m();
        let runs = Arc::new(AtomicUsize::new(0));
        let ticker = async {
            let mut ticks = 0;
            while ticks < 10 {
                tokio::time::sleep(TICK).await;
                ticks += 1;
            }
            Instant::now()
        };
        let load = async {
            let executor = scheduler.load_base("m", slow_load(&runs)).await;
            (executor, Instant::now())
        };
        // One thread runs both: a load on it would hold the ticker back.
        let (ticked, (executor, loaded)) = tokio::join!(ticker, load);
        assert!(executor.is_ok(), "{:?}", executor.err());
        assert!(ticked < loaded, "the ticker stalled behind the load");
        assert!(scheduler.is_loaded("m").await);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn racing_loads_of_one_model_load_it_once() {
        let (_dir, scheduler) = scheduler_with_m();
        let runs = Arc::new(AtomicUsize::new(0));
        let (a, b) = tokio::join!(
            scheduler.load_base("m", slow_load(&runs)),
            scheduler.load_base("m", slow_load(&runs)),
        );
        let (a, b) = (a.expect("first"), b.expect("second"));
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&a, &b), "both share the one executor");
    }

    #[tokio::test]
    async fn a_failed_or_panicked_load_registers_nothing() {
        let (_dir, scheduler) = scheduler_with_m();
        let failed = scheduler
            .load_base("m", |_| Err(anyhow!("weights do not read")))
            .await;
        let message = failed.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains("weights do not read"), "{message}");
        assert!(!scheduler.is_loaded("m").await);

        let panicked = scheduler
            .load_base("m", |_| -> Result<Executor<CpuRuntime>> {
                panic!("upload failed")
            })
            .await;
        let message = panicked.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(message.contains("upload failed"), "{message}");
        assert!(!scheduler.is_loaded("m").await);

        // The gate freed: the next load runs and registers.
        let runs = Arc::new(AtomicUsize::new(0));
        assert!(scheduler.load_base("m", slow_load(&runs)).await.is_ok());
        assert!(scheduler.is_loaded("m").await);
    }

    #[tokio::test]
    async fn a_dropped_request_still_registers_its_load() {
        let (_dir, scheduler) = scheduler_with_m();
        let runs = Arc::new(AtomicUsize::new(0));
        let gave_up = tokio::time::timeout(LOAD / 2, scheduler.load_base("m", slow_load(&runs)));
        assert!(gave_up.await.is_err(), "the request stopped waiting");
        assert_eq!(runs.load(Ordering::SeqCst), 1, "its load is in flight");
        assert!(!scheduler.is_loaded("m").await);

        // The next request waits for the load in flight, not a second one.
        let executor = scheduler.load_base("m", slow_load(&runs)).await;
        assert!(executor.is_ok(), "{:?}", executor.err());
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn an_unknown_model_evicts_nothing() {
        let (_dir, scheduler) = scheduler_with_m();
        let runs = Arc::new(AtomicUsize::new(0));
        assert!(scheduler.load_base("m", slow_load(&runs)).await.is_ok());
        let missing = scheduler.load_base("absent", slow_load(&runs)).await;
        let error = missing.err().expect("refused");
        assert_eq!(
            error.downcast_ref::<ModelNotFound>(),
            Some(&ModelNotFound::new("absent")),
            "{error:#}"
        );
        assert!(scheduler.is_loaded("m").await);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    /// `--model /abs/q.gguf` with `q.gguf` also in the model directory: the
    /// listed id, the bare name and the path reach one executor and load nothing.
    #[tokio::test]
    async fn every_name_of_the_start_file_reaches_the_loaded_copy() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("q.gguf");
        std::fs::write(&file, b"").expect("write");
        let spelled = file.display().to_string();
        let scheduler = Scheduler::<CpuRuntime>::new(dir.path().to_path_buf(), CpuDevice::new())
            .with_operator_model(Some(spelled.clone()));
        let runs = Arc::new(AtomicUsize::new(0));
        let first = scheduler
            .load_base(&spelled, slow_load(&runs))
            .await
            .expect("start model");
        for name in ["q.gguf", "q"] {
            let found = scheduler
                .load_base(name, slow_load(&runs))
                .await
                .expect("another name of the same file");
            assert!(Arc::ptr_eq(&first, &found), "{name} loaded a second copy");
        }
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert_eq!(scheduler.loaded_key("q").await, Some(spelled.clone()));
        assert!(scheduler.is_loaded(&spelled).await);

        // A keep-alive of zero through the listed id unloads the copy.
        scheduler
            .set_keep_alive("q.gguf", Some(Duration::ZERO))
            .await;
        assert!(!scheduler.is_loaded(&spelled).await);
    }

    /// An eviction drops its entry off the models lock: the drop reads the lock.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_evicted_entry_drops_off_the_models_lock() {
        let (_dir, scheduler) = scheduler_with_m();
        std::fs::create_dir(scheduler.model_dir().join("n")).expect("mkdir");
        let runs = Arc::new(AtomicUsize::new(0));
        scheduler
            .load_base("m", slow_load(&runs))
            .await
            .expect("first model");
        let readable = Arc::new(AtomicUsize::new(0));
        scheduler
            .attach_batching(
                "m",
                Arc::new(DropProbe {
                    models: Arc::clone(&scheduler.models),
                    readable: Arc::clone(&readable),
                }),
            )
            .await;
        scheduler
            .load_base("n", slow_load(&runs))
            .await
            .expect("second model");
        assert_eq!(
            readable.load(Ordering::SeqCst),
            1,
            "no drop saw a free lock"
        );
    }

    /// A batching scheduler whose drop counts a free models lock.
    struct DropProbe {
        models: Arc<tokio::sync::RwLock<Models<CpuRuntime>>>,
        readable: Arc<AtomicUsize>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            if self.models.try_read().is_ok() {
                self.readable.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    impl crate::engine::GenerationScheduler for DropProbe {
        fn submit(
            &self,
            _: Vec<u32>,
            _: crate::config::GenerationConfig,
        ) -> Result<crate::engine::RequestHandle> {
            Err(anyhow!("the probe takes no request"))
        }
    }
}
