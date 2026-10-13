//! The decision service of each model a scheduler loaded, and the
//! scheduler's decision entry points: the service of a model, and the
//! `/v1/models` report of a model without loading it.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::SystemTime;

use boostr::model::{LoadedModel, ModelClient};
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, UnaryOps,
};

use super::error::DecisionFailure;
use super::service::DecisionService;
use super::stats::DecisionCounts;
use crate::engine::Scheduler;
use crate::loader::decision::{
    decision_info, detect_decision_with, load_renderer, read_gguf_facts, DecisionInfo, GgufFacts,
};

/// The decision services of one scheduler, and the GGUF headers it read.
pub struct DecisionRegistry<R: Runtime<DType = DType>> {
    services: Mutex<HashMap<String, Arc<DecisionService<R>>>>,
    headers: Mutex<HashMap<PathBuf, (Option<SystemTime>, GgufFacts)>>,
}

impl<R: Runtime<DType = DType>> Default for DecisionRegistry<R> {
    fn default() -> Self {
        Self {
            services: Mutex::new(HashMap::new()),
            headers: Mutex::new(HashMap::new()),
        }
    }
}

/// A lock that a panicking holder left poisoned still guards a map whose
/// entries are whole: every insert and remove is one call.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl<R: Runtime<DType = DType>> DecisionRegistry<R> {
    /// The service of `name` when it is running on `model`.
    fn current(&self, name: &str, model: &Arc<LoadedModel<R>>) -> Option<Arc<DecisionService<R>>> {
        lock(&self.services)
            .get(name)
            .filter(|s| s.serves(model) && s.is_alive())
            .cloned()
    }

    /// The live service of `name`, whatever weights it runs on.
    fn running(&self, name: &str) -> Option<Arc<DecisionService<R>>> {
        lock(&self.services)
            .get(name)
            .filter(|s| s.is_alive())
            .cloned()
    }

    /// Keep `service` for `name`, unless a live one on the same weights got
    /// there first; returns the one kept.
    fn keep(
        &self,
        name: &str,
        service: DecisionService<R>,
        model: &Arc<LoadedModel<R>>,
    ) -> Arc<DecisionService<R>> {
        let mut services = lock(&self.services);
        if let Some(existing) = services
            .get(name)
            .filter(|s| s.serves(model) && s.is_alive())
        {
            return Arc::clone(existing);
        }
        let service = Arc::new(service);
        services.insert(name.to_string(), Arc::clone(&service));
        service
    }

    /// What the GGUF header of `file` says, read once per file version.
    fn gguf_facts(&self, file: &Path) -> Result<GgufFacts, String> {
        let stamp = std::fs::metadata(file).and_then(|m| m.modified()).ok();
        if let Some((seen, facts)) = lock(&self.headers).get(file) {
            if *seen == stamp && stamp.is_some() {
                return Ok(facts.clone());
            }
        }
        let facts = read_gguf_facts(file)?;
        lock(&self.headers).insert(file.to_path_buf(), (stamp, facts.clone()));
        Ok(facts)
    }
}

impl<R> Scheduler<R>
where
    R: Runtime<DType = DType> + 'static,
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + boostr::SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
    R::Device: Send + 'static,
    LoadedModel<R>: Send + Sync + 'static,
{
    /// The model `/v1/systemone` answers with when a request names none:
    /// the start model.
    pub fn decision_start_model(&self) -> Option<&str> {
        self.decision_config.start_model.as_deref()
    }

    /// The `/v1/models` report of the model `name` at `path`, from its
    /// sidecars and GGUF header, without loading weights. A model whose
    /// service runs reports what its load checked. `None` for a model that
    /// is not a decision model.
    ///
    /// # Errors
    ///
    /// When a sidecar or GGUF header of the model does not read.
    pub fn decision_info(
        &self,
        name: &str,
        path: &Path,
    ) -> Result<Option<Arc<DecisionInfo>>, String> {
        if let Some(service) = self.decisions.running(name) {
            return Ok(Some(Arc::new(service.info().clone())));
        }
        let source = detect_decision_with(path, name, &self.decision_config, |file| {
            self.decisions.gguf_facts(file)
        })?;
        source.map(|s| decision_info(&s).map(Arc::new)).transpose()
    }

    /// What the decision worker of `name` did so far. `None` when the model
    /// has no running service.
    pub fn decision_stats(&self, name: &str) -> Option<DecisionCounts> {
        self.decisions.running(name).map(|s| s.stats())
    }

    /// The decision service of the model `name`, loading the model and
    /// starting its worker when needed. The weights are the ones every
    /// other route of the model runs on.
    ///
    /// # Errors
    ///
    /// [`DecisionFailure::ModelNotFound`] when no model has that name,
    /// [`DecisionFailure::NotDecisionModel`] when it is not a qwen35
    /// decision model, [`DecisionFailure::Internal`] when it does not load
    /// or its renderer, sidecars or worker do not start.
    pub async fn decision_service(
        &self,
        name: &str,
    ) -> Result<Arc<DecisionService<R>>, DecisionFailure> {
        let path = self.model_path(name).ok_or_else(|| {
            DecisionFailure::ModelNotFound(format!(
                "model '{name}' not found: it is neither a path nor an entry of {}",
                self.model_dir().display()
            ))
        })?;
        let source = detect_decision_with(&path, name, &self.decision_config, |file| {
            self.decisions.gguf_facts(file)
        })
        .map_err(DecisionFailure::Internal)?
        .ok_or_else(|| {
            DecisionFailure::NotDecisionModel(format!(
                "model '{name}' is not a System One decision model: it ships no \
                 decider_config.json, prompt_contract.json or jebadiah.json, its GGUF header \
                 names no decision model, and no --decision-family applies to it"
            ))
        })?;
        let executor = self.get_executor(name).await.map_err(|e| {
            DecisionFailure::Internal(format!("model '{name}' did not load: {e:#}"))
        })?;
        let model = executor.model_arc();
        if let Some(service) = self.decisions.current(name, model) {
            return Ok(service);
        }
        if !matches!(&**model, LoadedModel::Qwen35(_)) {
            return Err(DecisionFailure::NotDecisionModel(format!(
                "model '{name}' is named a {} decision model, but its weights are not a qwen35 \
                 text model; decision models read label tokens off a qwen35 backbone",
                source.family.name()
            )));
        }
        let loaded = load_renderer(&source, self.decision_config.allow_contract_mismatch)
            .map_err(|e| DecisionFailure::Internal(format!("model '{name}': {e}")))?;
        let service = DecisionService::start(
            name,
            loaded,
            model,
            executor.device().clone(),
            &self.decision_config,
        )?;
        Ok(self.decisions.keep(name, service, model))
    }
}
