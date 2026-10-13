//! One decision model's service: its renderer, calibration, limits, and the
//! queue to its worker thread.

use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use boostr::inference::decision::{CancelToken, DecisionResult, TemperaturePolicy};
use boostr::model::{LoadedModel, ModelClient};
use boostr::{DType, Runtime};
use tokio::sync::oneshot;

use super::error::DecisionFailure;
use super::limits::DecisionLimits;
use super::prepare::{prepare, PrefixRule, PreparedRequest};
use super::readout::ReadoutHead;
use super::stats::{DecisionCounts, DecisionStats, PrefixCacheReport};
use super::worker::{spawn, Job, Shared, WorkerSettings};
use crate::config::DecisionConfig;
use crate::loader::decision::{Calibration, DecisionInfo, LoadedRenderer};
use crate::model::decision_template::{DecisionRenderer, Family, RenderedDecision};
use crate::telemetry::decision::set_decision_queue_rows;

/// A decision model ready to answer.
pub struct DecisionService<R: Runtime<DType = DType>> {
    name: String,
    loaded: LoadedRenderer,
    limits: DecisionLimits,
    prefix: PrefixRule,
    max_queue_rows: usize,
    queue: Sender<Job>,
    shared: Arc<Shared>,
    model: Weak<LoadedModel<R>>,
}

/// A request's answers, and what its run read from the prefix-state cache.
#[derive(Debug, Clone, PartialEq)]
pub struct Answered {
    /// The answers.
    pub result: DecisionResult,
    /// Prompt tokens the run did not run because the worker kept their
    /// prefix state: the prefix length on a cache hit, else 0.
    pub cached_input_tokens: usize,
}

/// A queued request. Dropping it before its answer arrives cancels it: a
/// queued request is skipped, a running one stops at its next pass.
pub struct Pending {
    rx: oneshot::Receiver<Result<Answered, DecisionFailure>>,
    cancel: CancelToken,
}

impl Pending {
    /// The answers.
    ///
    /// # Errors
    ///
    /// The run's failure, or [`DecisionFailure::Unavailable`] when the
    /// worker stopped before answering.
    pub async fn wait(self) -> Result<DecisionResult, DecisionFailure> {
        self.wait_answered().await.map(|answered| answered.result)
    }

    /// The answers and the prompt tokens the prefix-state cache saved.
    ///
    /// # Errors
    ///
    /// As [`Pending::wait`].
    pub async fn wait_answered(mut self) -> Result<Answered, DecisionFailure> {
        match (&mut self.rx).await {
            Ok(result) => result,
            Err(_) => Err(DecisionFailure::Unavailable(
                "the decision worker stopped before answering; send the request again".into(),
            )),
        }
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl<R: Runtime<DType = DType>> DecisionService<R> {
    /// Start the worker of `model` on `device` and serve `loaded` as `name`.
    /// A Kev model's pointer head and a Clef model's joint-schema head load
    /// here, on `device`.
    ///
    /// # Errors
    ///
    /// [`DecisionFailure::Internal`] when the worker thread does not start
    /// or the head does not load or fit the model.
    pub fn start(
        name: &str,
        loaded: LoadedRenderer,
        model: &Arc<LoadedModel<R>>,
        device: R::Device,
        cfg: &DecisionConfig,
    ) -> Result<Self, DecisionFailure>
    where
        R: 'static,
        R::Client: ModelClient<R>,
        R::Device: Send + 'static,
        LoadedModel<R>: Send + Sync + 'static,
        ReadoutHead<R>: Send + 'static,
    {
        let shared = Arc::new(Shared {
            queued_rows: Default::default(),
            alive: Default::default(),
            stats: DecisionStats::default(),
        });
        let settings = WorkerSettings {
            name: name.to_string(),
            max_lane_bytes: cfg.max_lane_bytes(),
            batch_wait: Duration::from_millis(cfg.batch_wait_ms),
            prefix_cache_bytes: cfg.prefix_cache_mb.saturating_mul(1 << 20),
            family: loaded.renderer.family(),
        };
        let head = ReadoutHead::load(
            loaded.pointer_head.as_deref(),
            loaded.joint_head.as_deref(),
            model,
            &device,
        )?;
        let weak = Arc::downgrade(model);
        let queue = spawn(settings, weak.clone(), head, device, Arc::clone(&shared))?;
        Ok(Self {
            name: name.to_string(),
            limits: DecisionLimits::new(cfg, loaded.renderer.family()),
            loaded,
            prefix: PrefixRule {
                min_tokens: cfg.shared_min_tokens,
                at_state: cfg.prefix_cache_mb > 0,
            },
            max_queue_rows: cfg.max_queue_rows,
            queue,
            shared,
            model: weak,
        })
    }

    /// The model name it serves.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The family whose prompt the model reads.
    pub fn family(&self) -> Family {
        self.loaded.renderer.family()
    }

    /// The renderer of the model's prompts.
    pub fn renderer(&self) -> &DecisionRenderer {
        &self.loaded.renderer
    }

    /// The `/v1/models` report, the contract checked with the tokenizer.
    pub fn info(&self) -> &DecisionInfo {
        &self.loaded.info
    }

    /// The fitted temperatures. `None`: every answer is raw.
    pub fn calibration(&self) -> Option<&Calibration> {
        self.loaded.calibration.as_ref()
    }

    /// What every response of this model carries in `warnings`.
    pub fn warnings(&self) -> &[String] {
        &self.loaded.warnings
    }

    /// The row and token limits of one request.
    pub fn limits(&self) -> DecisionLimits {
        self.limits
    }

    /// What the worker did so far.
    pub fn stats(&self) -> DecisionCounts {
        self.shared.stats.snapshot()
    }

    /// The worker's prefix-state cache, as `/v1/models` reports it.
    pub fn prefix_cache(&self) -> PrefixCacheReport {
        PrefixCacheReport::from(self.stats().prefix_cache)
    }

    /// Whether the worker thread still runs.
    pub fn is_alive(&self) -> bool {
        self.shared.alive.load(Ordering::SeqCst)
    }

    /// Whether this service runs on `model`, the weights a load produced.
    pub fn serves(&self, model: &Arc<LoadedModel<R>>) -> bool {
        std::ptr::eq(self.model.as_ptr(), Arc::as_ptr(model))
    }

    /// The temperatures of a request: the fitted ones, or 1 for a raw
    /// request or an uncalibrated model.
    pub fn temperature_policy(&self, raw: bool) -> TemperaturePolicy {
        match self.calibration() {
            Some(calibration) if !raw => calibration.policy(),
            _ => TemperaturePolicy::Fixed(1.0),
        }
    }

    /// Check `rendered` against the limits and build its run request.
    ///
    /// # Errors
    ///
    /// [`DecisionFailure::TooLarge`] over a limit; planning failures.
    pub fn prepare(
        &self,
        rendered: &RenderedDecision,
        raw: bool,
    ) -> Result<PreparedRequest, DecisionFailure> {
        self.limits.check(&rendered.rows)?;
        prepare(rendered, self.temperature_policy(raw), self.prefix)
    }

    /// Queue `prepared` on the worker.
    ///
    /// # Errors
    ///
    /// [`DecisionFailure::Busy`] when the queue would pass
    /// `max_queue_rows`, [`DecisionFailure::TooLarge`] when the request
    /// alone passes it, [`DecisionFailure::Unavailable`] when the worker
    /// stopped.
    pub fn submit(&self, prepared: PreparedRequest) -> Result<Pending, DecisionFailure> {
        let rows = prepared.rows;
        if rows > self.max_queue_rows {
            return Err(DecisionFailure::TooLarge(format!(
                "the request expands to {rows} rows, more than the {} the queue holds \
                 (inference.decision.max_queue_rows)",
                self.max_queue_rows
            )));
        }
        let queued = &self.shared.queued_rows;
        let mut current = queued.load(Ordering::SeqCst);
        loop {
            if current + rows > self.max_queue_rows {
                return Err(DecisionFailure::Busy(format!(
                    "server busy: {current} rows queued, the limit is {} \
                     (inference.decision.max_queue_rows); retry later",
                    self.max_queue_rows
                )));
            }
            match queued.compare_exchange(
                current,
                current + rows,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(now) => current = now,
            }
        }
        set_decision_queue_rows(&self.name, current + rows);
        let (reply, rx) = oneshot::channel();
        let cancel = CancelToken::new();
        let job = Job {
            request: prepared.request,
            rows,
            prefill_tokens: prepared.prefill_tokens,
            cancel: cancel.clone(),
            reply,
            queued: Instant::now(),
        };
        if self.queue.send(job).is_err() {
            let before = queued.fetch_sub(rows, Ordering::SeqCst);
            set_decision_queue_rows(&self.name, before.saturating_sub(rows));
            return Err(DecisionFailure::Unavailable(format!(
                "the decision worker of model '{}' stopped; send the request again",
                self.name
            )));
        }
        Ok(Pending { rx, cancel })
    }
}
