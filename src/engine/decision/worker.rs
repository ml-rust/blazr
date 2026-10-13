//! The decision worker: one blocking thread per decision model, fed by a
//! queue. It takes every waiting request at once, runs the requests with no
//! shared prefix as one merged run, runs each request with a shared prefix
//! alone over that prefix, and sends each request its answers. The model's
//! readout head, label-token, pointer or joint-schema, scores every run. A
//! joint-schema record is its plan's prefix, so it always runs alone.
//!
//! A request with a shared prefix reads its prefix state from the worker's
//! prefix-state cache when an earlier request kept it, and keeps the state
//! it runs. The worker drops every kept state once its model unloads.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use boostr::inference::decision::{
    merge_requests, split_result, CancelToken, DecisionRequest, RunnerConfig,
};
use boostr::model::{LoadedModel, ModelClient, Qwen35Model};
use boostr::{DType, Runtime};
use tokio::sync::oneshot;

use super::error::DecisionFailure;
use super::prefix_cache::{publish, WorkerCache};
use super::readout::{ReadoutHead, ReadoutItem};
use super::service::Answered;
use super::stats::DecisionStats;
use crate::model::decision_template::Family;
use crate::telemetry::decision::{record_decision, set_decision_queue_rows};

/// Where a job's answers go.
pub(super) type Reply = oneshot::Sender<Result<Answered, DecisionFailure>>;

/// How often an idle worker that keeps prefix states checks whether its
/// model unloaded, to free them.
const RELEASE_CHECK: Duration = Duration::from_secs(1);

/// One queued request.
pub(super) struct Job {
    pub request: DecisionRequest<ReadoutItem>,
    pub rows: usize,
    pub prefill_tokens: usize,
    pub cancel: CancelToken,
    pub reply: Reply,
    pub queued: Instant,
}

impl Job {
    /// The caller went away: its response is dropped or it cancelled.
    fn abandoned(&self) -> bool {
        self.reply.is_closed() || self.cancel.is_cancelled()
    }
}

/// What a worker is told at start.
pub(super) struct WorkerSettings {
    /// The model name, for metrics and messages.
    pub name: String,
    /// The lane-state budget of one forward pass.
    pub max_lane_bytes: Option<usize>,
    /// How long to wait after a request for more to merge with it.
    pub batch_wait: Duration,
    /// The byte bound of the prefix-state cache. 0 keeps no state.
    pub prefix_cache_bytes: usize,
    /// The family the model's prompts are rendered for.
    pub family: Family,
}

/// What the worker shares with its service.
pub(super) struct Shared {
    /// Rows queued or running.
    pub queued_rows: AtomicUsize,
    /// False once the worker thread has stopped.
    pub alive: AtomicBool,
    /// What the worker did.
    pub stats: DecisionStats,
}

/// Clears [`Shared::alive`] when the thread ends, by return or by panic.
struct AliveGuard(Arc<Shared>);

impl Drop for AliveGuard {
    fn drop(&mut self) {
        self.0.alive.store(false, Ordering::SeqCst);
    }
}

struct Worker<R: Runtime<DType = DType>> {
    settings: WorkerSettings,
    model: Weak<LoadedModel<R>>,
    head: ReadoutHead<R>,
    client: R::Client,
    shared: Arc<Shared>,
    cache: WorkerCache<R>,
}

/// Start the worker thread of `model` on `device`, scoring with `head`.
/// The thread runs until every sender of the returned queue drops.
///
/// # Errors
///
/// [`DecisionFailure::Internal`] when the platform cannot start a thread,
/// as in a browser build.
pub(super) fn spawn<R>(
    settings: WorkerSettings,
    model: Weak<LoadedModel<R>>,
    head: ReadoutHead<R>,
    device: R::Device,
    shared: Arc<Shared>,
) -> Result<Sender<Job>, DecisionFailure>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: ModelClient<R>,
    R::Device: Send + 'static,
    LoadedModel<R>: Send + Sync + 'static,
    ReadoutHead<R>: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    shared.alive.store(true, Ordering::SeqCst);
    let guard = AliveGuard(Arc::clone(&shared));
    std::thread::Builder::new()
        .name("blazr-decision".into())
        .spawn(move || {
            let _guard = guard;
            let client = R::default_client(&device);
            let address = model.as_ptr() as *const () as usize;
            let cache = WorkerCache::new(settings.prefix_cache_bytes, address, settings.family);
            Worker {
                settings,
                model,
                head,
                client,
                shared,
                cache,
            }
            .serve(rx);
        })
        .map_err(|e| {
            DecisionFailure::Internal(format!("the decision worker thread did not start: {e}"))
        })?;
    Ok(tx)
}

/// A boostr run error as a request failure.
fn run_failure(e: boostr::error::Error) -> DecisionFailure {
    match e {
        boostr::error::Error::Cancelled { .. } => {
            DecisionFailure::Unavailable("the request was cancelled".into())
        }
        other => DecisionFailure::Internal(format!("the decision run failed: {other}")),
    }
}

impl<R> Worker<R>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    /// Take every waiting job, run them, repeat until the queue closes.
    /// While it keeps prefix states, an idle worker checks every
    /// [`RELEASE_CHECK`] whether its model unloaded.
    fn serve(mut self, rx: Receiver<Job>) {
        self.shared.stats.set_prefix_cache(&self.cache.stats());
        loop {
            let first = if self.cache.is_empty() {
                match rx.recv() {
                    Ok(job) => job,
                    Err(_) => break,
                }
            } else {
                match rx.recv_timeout(RELEASE_CHECK) {
                    Ok(job) => job,
                    Err(RecvTimeoutError::Timeout) => {
                        if self.model.strong_count() == 0 {
                            self.drop_states();
                        }
                        continue;
                    }
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            };
            let mut jobs = vec![first];
            let deadline = Instant::now() + self.settings.batch_wait;
            loop {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                match rx.recv_timeout(left) {
                    Ok(job) => jobs.push(job),
                    Err(_) => break,
                }
            }
            jobs.extend(rx.try_iter());
            self.run_batch(jobs);
        }
    }

    /// Drop every kept prefix state and publish it.
    fn drop_states(&mut self) {
        let before = self.cache.stats();
        self.cache.release();
        let name = &self.settings.name;
        publish(name, &self.shared.stats, &before, &self.cache.stats());
    }

    fn run_batch(&mut self, jobs: Vec<Job>) {
        let mut live = Vec::with_capacity(jobs.len());
        for job in jobs {
            if job.abandoned() {
                self.shared.stats.abandoned();
                self.release(&job);
            } else {
                live.push(job);
            }
        }
        if live.is_empty() {
            return;
        }
        let Some(loaded) = self.model.upgrade() else {
            self.drop_states();
            let msg = format!(
                "model '{}' was unloaded; send the request again to load it",
                self.settings.name
            );
            for job in live {
                self.finish(job, Err(DecisionFailure::Unavailable(msg.clone())));
            }
            return;
        };
        let LoadedModel::Qwen35(model) = &*loaded else {
            let msg = format!("model '{}' is not a qwen35 model", self.settings.name);
            for job in live {
                self.finish(job, Err(DecisionFailure::NotDecisionModel(msg.clone())));
            }
            return;
        };
        let (alone, merged): (Vec<Job>, Vec<Job>) = live
            .into_iter()
            .partition(|job| !job.request.plan.prefix.is_empty());
        self.run_merged(model, merged);
        for job in alone {
            self.shared.stats.prefix_run();
            let result = self.run(model, &job.request, Some(job.cancel.clone()));
            self.finish(job, result);
        }
    }

    /// Requests with no shared prefix: one merged run, cut back per request.
    /// A merged run that fails runs each request alone, so each gets its own
    /// outcome. The merged run reads no cancel flag: one caller leaving does
    /// not stop the others.
    fn run_merged(&mut self, model: &Qwen35Model<R>, jobs: Vec<Job>) {
        if jobs.len() >= 2 {
            let requests = jobs.iter().map(|job| job.request.clone()).collect();
            let outcome = merge_requests(requests)
                .map_err(run_failure)
                .and_then(|merged| {
                    let answered = self.run(model, &merged.request, None)?;
                    split_result(answered.result, &merged.parts).map_err(run_failure)
                });
            match outcome {
                Ok(results) if results.len() == jobs.len() => {
                    self.shared.stats.merged(jobs.len());
                    for (job, result) in jobs.into_iter().zip(results) {
                        let answered = Answered {
                            result,
                            cached_input_tokens: 0,
                        };
                        self.finish(job, Ok(answered));
                    }
                    return;
                }
                Ok(results) => tracing::warn!(
                    "decision model '{}': a merged run of {} requests returned {} results; \
                     running each alone",
                    self.settings.name,
                    jobs.len(),
                    results.len()
                ),
                Err(e) => tracing::warn!(
                    "decision model '{}': a merged run of {} requests failed ({e}); running \
                     each alone",
                    self.settings.name,
                    jobs.len()
                ),
            }
        }
        for job in jobs {
            let result = self.run(model, &job.request, Some(job.cancel.clone()));
            self.finish(job, result);
        }
    }

    /// One run of `request`, its prefix state from the cache when the
    /// cache holds it. A panic in the run fails the request, drops every
    /// kept state, and leaves the worker serving.
    fn run(
        &mut self,
        model: &Qwen35Model<R>,
        request: &DecisionRequest<ReadoutItem>,
        cancel: Option<CancelToken>,
    ) -> Result<Answered, DecisionFailure> {
        let cfg = RunnerConfig {
            max_lane_bytes: self.settings.max_lane_bytes,
            cancel,
            ..RunnerConfig::default()
        };
        let before = self.cache.stats();
        let plan = &request.plan;
        let attempt = catch_unwind(AssertUnwindSafe(|| {
            let prefix = self
                .cache
                .prefix(&self.client, model, plan, &cfg)
                .map_err(run_failure)?;
            let state = prefix.as_ref().map(|p| &*p.state);
            let result = self
                .head
                .run(&self.client, model, request, state, &cfg)
                .map_err(run_failure)?;
            let hit = prefix.as_ref().is_some_and(|p| p.hit);
            Ok(Answered {
                result,
                cached_input_tokens: if hit { plan.prefix_len() } else { 0 },
            })
        }));
        let outcome = attempt.unwrap_or_else(|_| {
            self.cache.release();
            Err(DecisionFailure::Internal(
                "the decision run panicked; the request was not answered".into(),
            ))
        });
        let name = &self.settings.name;
        publish(name, &self.shared.stats, &before, &self.cache.stats());
        outcome
    }

    /// Release a job's rows from the queue count.
    fn release(&self, job: &Job) {
        let before = self
            .shared
            .queued_rows
            .fetch_sub(job.rows, Ordering::SeqCst);
        set_decision_queue_rows(&self.settings.name, before.saturating_sub(job.rows));
    }

    /// Release a job, record it, and send its outcome.
    fn finish(&self, job: Job, result: Result<Answered, DecisionFailure>) {
        self.release(&job);
        let answered = result.is_ok();
        self.shared.stats.finished(answered);
        record_decision(
            &self.settings.name,
            job.rows,
            job.prefill_tokens,
            job.queued.elapsed().as_secs_f64(),
            answered,
        );
        // A caller that went away while its request ran takes no answer.
        let _ = job.reply.send(result);
    }
}
