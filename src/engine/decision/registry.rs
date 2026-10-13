//! The decision service of each model a scheduler loaded, and the
//! scheduler's decision entry points: the service of a model, and the
//! `/v1/models` report of a model without loading it.

use std::borrow::Cow;
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
use crate::config::DecisionConfig;
use crate::engine::Scheduler;
use crate::loader::decision::sidecar::{DECIDER_CONFIG, JEBADIAH_CARD, PROMPT_CONTRACT};
use crate::loader::decision::{
    decision_info, detect_decision_with, load_renderer, read_gguf_facts, DecisionInfo, GgufFacts,
};

/// The decision settings of a LoRA variant of `base` with its adapter at
/// `adapter`. `--decision-family` and `--decision-config` apply to the
/// variant when they apply to its base. A sidecar in the adapter directory
/// (`decider_config.json`, `prompt_contract.json` or `jebadiah.json`)
/// applies when no configured sidecar does, so the prompt contract can ship
/// with the adapter. The base's own sidecars apply otherwise.
pub fn variant_decision_config(cfg: &DecisionConfig, base: &str, adapter: &Path) -> DecisionConfig {
    let applies = cfg.overrides_apply_to(base);
    let adapter_dir = if adapter.is_dir() {
        Some(adapter)
    } else {
        adapter.parent()
    };
    let adapter_sidecar = adapter_dir.and_then(|dir| {
        [DECIDER_CONFIG, PROMPT_CONTRACT, JEBADIAH_CARD]
            .iter()
            .map(|file| dir.join(file))
            .find(|file| file.is_file())
    });
    DecisionConfig {
        start_model: None,
        family: cfg.family.filter(|_| applies),
        config: cfg.config.clone().filter(|_| applies).or(adapter_sidecar),
        ..cfg.clone()
    }
}

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
        let settings = self.decision_settings(name);
        let source = detect_decision_with(path, name, &settings, |file| {
            self.decisions.gguf_facts(file)
        })?;
        source.map(|s| decision_info(&s).map(Arc::new)).transpose()
    }

    /// The decision settings detection runs with for `name`: the server's,
    /// or for a LoRA variant, [`variant_decision_config`] over them.
    fn decision_settings(&self, name: &str) -> Cow<'_, DecisionConfig> {
        match self.variant(name) {
            Some(v) => Cow::Owned(variant_decision_config(
                &self.decision_config,
                &v.base,
                &v.path,
            )),
            None => Cow::Borrowed(&self.decision_config),
        }
    }

    /// The weights path of `name`: its own, or for a LoRA variant its
    /// base's.
    async fn decision_weights(&self, name: &str) -> Result<PathBuf, DecisionFailure> {
        let weights = self
            .variant(name)
            .map_or_else(|| name.to_string(), |v| v.base);
        if let Some(path) = self.model_path(&weights) {
            return Ok(path);
        }
        if let Some(error) = self.unknown_variant(name).await {
            return Err(error.into());
        }
        Err(DecisionFailure::ModelNotFound(format!(
            "model '{weights}' not found: it is neither a path nor an entry of {}",
            self.model_dir().display()
        )))
    }

    /// What the decision worker of `name` did so far. `None` when the model
    /// has no running service.
    pub fn decision_stats(&self, name: &str) -> Option<DecisionCounts> {
        self.decisions.running(name).map(|s| s.stats())
    }

    /// The decision service of the model `name`, loading the model and
    /// starting its worker when needed. The weights are the ones every
    /// other route of the model runs on. A LoRA variant id serves on the
    /// variant's model, detected on its base's weights.
    ///
    /// # Errors
    ///
    /// [`DecisionFailure::ModelNotFound`] when no model or adapter has that
    /// name, [`DecisionFailure::NotDecisionModel`] when it is not a qwen35
    /// decision model, [`DecisionFailure::Internal`] when it does not load
    /// or its renderer, sidecars or worker do not start, and the
    /// [`LoraError`](crate::engine::LoraError) kinds of a variant that does
    /// not build.
    pub async fn decision_service(
        &self,
        name: &str,
    ) -> Result<Arc<DecisionService<R>>, DecisionFailure> {
        let path = self.decision_weights(name).await?;
        let settings = self.decision_settings(name);
        let source = detect_decision_with(&path, name, &settings, |file| {
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
        let executor = self.get_executor(name).await.map_err(|e| match e
            .downcast::<crate::engine::LoraError>()
        {
            Ok(lora) => DecisionFailure::from(lora),
            Err(e) => DecisionFailure::Internal(format!("model '{name}' did not load: {e:#}")),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DecisionFamily;

    #[test]
    fn a_variant_takes_its_base_overrides_then_its_adapter_sidecar() {
        let adapter = tempfile::tempdir().expect("temp dir");
        let start = DecisionConfig {
            family: Some(DecisionFamily::Jebadiah),
            config: Some(PathBuf::from("/cfg/jebadiah.json")),
            start_model: Some("base".into()),
            max_rows: 7,
            ..DecisionConfig::default()
        };

        let applies = variant_decision_config(&start, "base", adapter.path());
        assert_eq!(applies.family, Some(DecisionFamily::Jebadiah));
        assert_eq!(applies.config, start.config);
        assert_eq!(applies.start_model, None);
        assert_eq!(applies.max_rows, 7);
        assert!(applies.overrides_apply_to("base:kev"));

        let other = variant_decision_config(&start, "other", adapter.path());
        assert_eq!(other.family, None);
        assert_eq!(other.config, None);

        let sidecar = adapter.path().join(DECIDER_CONFIG);
        std::fs::write(&sidecar, "{}").expect("write");
        let shipped = variant_decision_config(&start, "other", adapter.path());
        assert_eq!(shipped.config, Some(sidecar.clone()));
        let file = adapter.path().join("adapter_model.safetensors");
        let shipped = variant_decision_config(&start, "other", &file);
        assert_eq!(shipped.config, Some(sidecar));
        let configured = variant_decision_config(&start, "base", adapter.path());
        assert_eq!(configured.config, start.config);
    }
}
