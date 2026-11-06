//! Model scheduler for lifecycle management
//!
//! Manages loading, unloading, and eviction of models.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};

use crate::config::{DtypeRequest, InferenceOverlay, VisionOverrides, DEFAULT_NUM_CTX};
use tokio::sync::RwLock;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, UnaryOps,
};

pub use super::model_dir::AvailableModelInfo;
use super::model_dir::OperatorNames;
use crate::config::DecisionConfig;
use crate::engine::decision::DecisionRegistry;
use crate::engine::load::blocking::run_blocking;
use crate::engine::load::gate::LoadGates;
use crate::engine::load::job::LoadJob;
use crate::engine::load::registered::{release, Models};
use crate::engine::lora::eviction::{self, Resident};
use crate::engine::lora::LoraRegistry;
use crate::engine::{Executor, GenerationScheduler, ModelNotFound, ModelNotLoadable};
use crate::loader::NotLoadable;

/// Model entry in the scheduler
pub(crate) struct ModelEntry<R: Runtime<DType = DType>> {
    /// The executor wrapping the model
    pub(crate) executor: Arc<Executor<R>>,
    /// When the model, or one of its LoRA variants, was last accessed
    pub(crate) last_accessed: Instant,
    /// Model path for reloading. A variant holds its base's path.
    pub(crate) path: PathBuf,
    /// Per-model keep-alive duration. None = forever (no auto-unload).
    pub(crate) keep_alive: Option<Duration>,
    /// The base model of a LoRA variant. `None` for a base model.
    pub(crate) variant_of: Option<String>,
    /// The batching scheduler of the start model. Its engine thread holds the
    /// executor, so this entry owns it and an eviction stops the thread.
    pub(crate) batching: Option<Arc<dyn GenerationScheduler>>,
    /// Data-parallel replicas hold this executor for the server's life, so no
    /// eviction can free it, and eviction skips it.
    pub(crate) held: bool,
}

/// Every loaded model eviction can free.
pub(crate) fn residents<R: Runtime<DType = DType>>(models: &Models<R>) -> Vec<Resident<'_>> {
    models
        .iter()
        .filter(|(_, entry)| !entry.held)
        .map(|(name, entry)| Resident {
            name: name.as_str(),
            variant_of: entry.variant_of.as_deref(),
            last_accessed: entry.last_accessed,
            keep_alive: entry.keep_alive,
        })
        .collect()
}

/// Take `name` and, for a base model, each of its variants first out of
/// `models`. The caller releases the lock, then [`release`]s the entries. Empty
/// when `name` was not loaded.
fn remove_with_variants<R: Runtime<DType = DType>>(
    models: &mut Models<R>,
    name: &str,
) -> Vec<ModelEntry<R>> {
    let order = eviction::unload_order(&residents(models), name);
    order
        .iter()
        .filter_map(|victim| models.remove(victim))
        .collect()
}

/// Model scheduler
///
/// Manages model lifecycle including:
/// - Loading models on demand
/// - Caching loaded models
/// - Evicting least-recently-used models when memory is tight
pub struct Scheduler<R: Runtime<DType = DType>> {
    /// Loaded models by name: base models and LoRA variants. Shared with
    /// the blocking thread of each load, which registers its model.
    pub(crate) models: Arc<RwLock<Models<R>>>,
    /// One load at a time per model or variant name.
    pub(crate) gates: LoadGates,
    /// Maximum number of loaded models
    pub(crate) max_loaded: usize,
    /// Model directory for discovery
    pub(crate) model_dir: PathBuf,
    /// The names an operator stated. Each can be a path.
    pub(crate) operator_names: OperatorNames,
    /// Device for loading
    device: R::Device,
    /// Default context size for KV cache
    num_ctx: usize,
    /// `--chat-template`: a built-in name or a template file, applied to
    /// every model this scheduler loads.
    chat_template: Option<String>,
    /// `--mmproj` and the image token limits, applied to every model this
    /// scheduler loads.
    vision: VisionOverrides,
    /// The user config file's `inference` keys, written over every model's
    /// own values before the command-line vision flags.
    inference_overlay: InferenceOverlay,
    /// Tensor parallel degree from `BLAZR_TP_SIZE` or the user config file.
    /// `None` takes each model's `config.json` value, else 1.
    tensor_parallel_size: Option<usize>,
    /// `--dtype`, then `inference.dtype`. `None` takes each model's
    /// `torch_dtype`, else f32.
    dtype: Option<DtypeRequest>,
    /// `inference.decision` and the `--decision-*` flags.
    pub(crate) decision_config: DecisionConfig,
    /// The decision worker of each model that served `/v1/systemone`.
    pub(crate) decisions: DecisionRegistry<R>,
    /// The registered LoRA adapter variants. Shared with the blocking
    /// thread of each variant build, which registers its variant.
    pub(crate) lora: Arc<LoraRegistry>,
    /// Serve the registered models only: no load, unload or eviction.
    pub(crate) pinned: bool,
}

impl<R: Runtime<DType = DType>> Scheduler<R>
where
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
{
    /// Create a new scheduler
    pub fn new(model_dir: PathBuf, device: R::Device) -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            gates: LoadGates::default(),
            max_loaded: 1, // Default to single model
            model_dir,
            operator_names: OperatorNames::default(),
            device,
            num_ctx: DEFAULT_NUM_CTX,
            chat_template: None,
            vision: VisionOverrides::default(),
            inference_overlay: InferenceOverlay::default(),
            tensor_parallel_size: None,
            dtype: None,
            decision_config: DecisionConfig::default(),
            decisions: DecisionRegistry::default(),
            lora: Arc::new(LoraRegistry::new()),
            pinned: false,
        }
    }

    /// Serve only the models registered with [`Self::register_base`].
    ///
    /// A request for any other name is [`crate::engine::ModelNotFound`].
    /// No model loads, unloads or expires, so a front end with no weights,
    /// such as the disaggregated router, never loads a model by accident.
    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }

    /// Register `executor` under `name`, loaded from `path`, with no
    /// keep-alive. It replaces a model of the same name.
    pub async fn register_base(
        &self,
        name: &str,
        path: PathBuf,
        executor: Executor<R>,
    ) -> Arc<Executor<R>> {
        let executor = Arc::new(executor);
        self.models.write().await.insert(
            name.to_string(),
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
        executor
    }

    /// Give the loaded `name` its batching scheduler to own. An eviction of
    /// `name` then stops the scheduler's engine thread with it.
    pub async fn attach_batching(&self, name: &str, batching: Arc<dyn GenerationScheduler>) {
        if let Some(entry) = self.models.write().await.get_mut(name) {
            entry.batching = Some(batching);
        }
    }

    /// Keep the loaded `name` out of every eviction and unload, for a model
    /// that another owner holds for the server's life.
    pub async fn hold(&self, name: &str) {
        if let Some(entry) = self.models.write().await.get_mut(name) {
            entry.held = true;
        }
    }

    /// Detect, batch and bound System One decision models with `config`. Its
    /// `base` can be a path, because an operator states it.
    pub fn with_decision(mut self, config: DecisionConfig) -> Self {
        self.operator_names.set_decision_base(config.base.clone());
        self.decision_config = config;
        self
    }

    /// Accept `model` as a path: the `--model` of an operator. Every other name
    /// resolves inside the model directory, so no request reaches a path.
    pub fn with_operator_model(mut self, model: Option<String>) -> Self {
        self.operator_names.set_start(model);
        self
    }

    /// Load every model at the compute dtype `request` names. `None` takes
    /// each model's `torch_dtype`, else f32.
    pub fn with_dtype(mut self, request: Option<DtypeRequest>) -> Self {
        self.dtype = request;
        self
    }

    /// Write the user config file's `inference` keys over every model.
    pub fn with_inference_overlay(mut self, overlay: InferenceOverlay) -> Self {
        self.inference_overlay = overlay;
        self
    }

    /// Set the tensor parallel degree for every model. `None` takes each
    /// model's `config.json` value, else 1.
    pub fn with_tensor_parallel_size(mut self, size: Option<usize>) -> Self {
        self.tensor_parallel_size = size;
        self
    }

    /// The directory models are searched and pulled into.
    pub fn model_dir(&self) -> &std::path::Path {
        &self.model_dir
    }

    /// The device this scheduler loads onto: the server's device.
    pub fn device(&self) -> &R::Device {
        &self.device
    }

    /// Load the vision tower `overrides` names beside every model, with its
    /// image token limits.
    pub fn with_vision(mut self, overrides: VisionOverrides) -> Self {
        self.vision = overrides;
        self
    }

    /// Override every loaded model's chat template with a built-in name or a
    /// template file.
    pub fn with_chat_template(mut self, chat_template: Option<String>) -> Self {
        self.chat_template = chat_template;
        self
    }

    /// Set the context size for KV cache (like Ollama's num_ctx)
    pub fn with_num_ctx(mut self, num_ctx: usize) -> Self {
        self.num_ctx = num_ctx;
        self
    }

    /// Set maximum number of concurrent loaded models
    pub fn with_max_loaded(mut self, max: usize) -> Self {
        self.max_loaded = max;
        self
    }

    /// Get an executor for a model, loading it if necessary.
    ///
    /// A registered LoRA variant id, `<model>:<adapter>`, builds the variant
    /// from its base, loading the base first when needed. A Kev directory
    /// builds as the variant of its base under its own name.
    ///
    /// # Errors
    ///
    /// A [`LoraError`](crate::engine::LoraError) when a variant does not build, or
    /// when `model_name` has the shape `<model>:<adapter>` for a known
    /// model and no such adapter is registered. Otherwise the load error.
    pub async fn get_executor(&self, model_name: &str) -> Result<Arc<Executor<R>>> {
        if let Some(executor) = self.touch(model_name).await {
            return Ok(executor);
        }
        if self.pinned {
            return Err(crate::engine::ModelNotFound::new(model_name).into());
        }
        if let Some(info) = self.lora.get(model_name) {
            return Ok(self.load_variant(&info).await?);
        }
        if let Some(executor) = self.load_kev(model_name).await? {
            return Ok(executor);
        }
        if let Some(error) = self.unknown_variant(model_name).await {
            return Err(error.into());
        }
        self.load_model(model_name).await
    }

    /// Load the base model `model_name` off the async executor, once: a
    /// request that races a load of the same model waits for it and gets
    /// its executor. See [`Self::load_base`].
    ///
    /// A name for the weights of a loaded model returns that model. The name is
    /// checked before any eviction: its kind, its files, its config and its
    /// tokenizer. A refused name leaves every loaded model in place. A load that
    /// passes these checks and then fails has already evicted the models it made
    /// room for: bad tensor bytes, or a device that holds too little.
    ///
    /// # Errors
    ///
    /// [`ModelNotFound`](crate::engine::ModelNotFound) for no such model or no
    /// weights, [`ModelNotLoadable`](crate::engine::ModelNotLoadable) for another
    /// kind of model, a config or tokenizer that does not read, a load that
    /// fails, or a load that panics. A failed load registers nothing.
    pub(crate) async fn load_model(&self, model_name: &str) -> Result<Arc<Executor<R>>> {
        if let Some(executor) = self.served_copy(model_name).await {
            return Ok(executor);
        }
        let names = self.operator_names.clone();
        let model_dir = self.model_dir.clone();
        let target = model_name.to_string();
        run_blocking(move || check_before_load::<R>(&names, &model_dir, &target))
            .await
            .map_err(|e| anyhow!("model '{model_name}': the load check stopped: {e}"))??;
        let job = self.load_job();
        self.load_base(model_name, move |path| job.run(path)).await
    }

    /// This scheduler's load settings, owned for a load thread.
    fn load_job(&self) -> LoadJob<R> {
        LoadJob {
            device: self.device.clone(),
            num_ctx: self.num_ctx,
            chat_template: self.chat_template.clone(),
            vision: self.vision.clone(),
            inference_overlay: self.inference_overlay.clone(),
            tensor_parallel_size: self.tensor_parallel_size,
            dtype: self.dtype,
        }
    }

    /// The path the model name `model_name` resolves to, as a load resolves
    /// it. `None` when no such model exists.
    pub fn model_path(&self, model_name: &str) -> Option<PathBuf> {
        self.operator_names.resolve(&self.model_dir, model_name)
    }

    /// Set the keep-alive duration for a model.
    /// `None` means forever. `Some(Duration::ZERO)` means unload immediately,
    /// a base model's LoRA variants first. A [`Self::pinned`] scheduler
    /// keeps every model forever and ignores the call.
    pub async fn set_keep_alive(&self, model_name: &str, keep_alive: Option<Duration>) {
        if self.pinned {
            return;
        }
        let Some(key) = self.loaded_key(model_name).await else {
            return;
        };
        let removed = {
            let mut models = self.models.write().await;
            if let Some(entry) = models.get_mut(&key) {
                entry.keep_alive = keep_alive;
            }
            if keep_alive == Some(Duration::ZERO) {
                remove_with_variants(&mut models, &key)
            } else {
                Vec::new()
            }
        };
        if !removed.is_empty() {
            tracing::info!("Unloaded model '{}' (keep_alive=0)", model_name);
        }
        release(removed).await;
    }

    /// Evict models that have exceeded their keep-alive duration, with the
    /// LoRA variants of each evicted base. Called periodically by the
    /// reaper task.
    pub async fn evict_expired(&self) {
        let removed: Vec<ModelEntry<R>> = {
            let mut models = self.models.write().await;
            let order = eviction::expired(&residents(&models), Instant::now());
            order
                .iter()
                .filter_map(|name| {
                    tracing::info!("Auto-unloaded model '{}' (keep_alive expired)", name);
                    models.remove(name)
                })
                .collect()
        };
        release(removed).await;
    }

    /// Unload a specific model: a base model's LoRA variants first, then
    /// the model. A variant unloads alone and stays registered, so the next
    /// request for it builds it again. A [`Self::pinned`] scheduler unloads
    /// nothing and returns `false`.
    pub async fn unload(&self, model_name: &str) -> bool {
        if self.pinned {
            return false;
        }
        let Some(key) = self.loaded_key(model_name).await else {
            return false;
        };
        let removed = remove_with_variants(&mut *self.models.write().await, &key);
        let unloaded = !removed.is_empty();
        release(removed).await;
        unloaded
    }

    /// List loaded models, each under a name a client sends.
    pub async fn list_loaded(&self) -> Vec<LoadedModelInfo> {
        let models = self.models.read().await;
        models
            .iter()
            .map(|(name, entry)| LoadedModelInfo {
                name: self.operator_names.public_name(name),
                path: entry.path.clone(),
                last_accessed: entry.last_accessed,
            })
            .collect()
    }

    /// List available models in the model directory
    pub fn list_available(&self) -> Result<Vec<AvailableModelInfo>> {
        Ok(super::model_dir::list_available(&self.model_dir)?)
    }

    /// The models a client can request: [`Self::list_available`] with the start
    /// model, when it sits outside the model directory.
    pub fn list_served(&self) -> Result<Vec<AvailableModelInfo>> {
        let mut models = self.list_available()?;
        if let Some(start) = self.operator_names.unlisted_start(&self.model_dir, &models) {
            models.push(start);
        }
        Ok(models)
    }
}

/// Refuse `name` before any eviction.
///
/// # Errors
///
/// As [`Scheduler::load_model`]. A config or tokenizer that does not read keeps
/// its loader's error.
fn check_before_load<R: Runtime<DType = DType>>(
    names: &OperatorNames,
    model_dir: &Path,
    name: &str,
) -> Result<()> {
    let path = names
        .resolve(model_dir, name)
        .ok_or_else(|| ModelNotFound::new(name))?;
    match crate::loader::check_loadable(&path) {
        Ok(()) => {}
        Err(NotLoadable::NoWeights(_)) => return Err(ModelNotFound::new(name).into()),
        Err(refused) => {
            return Err(ModelNotLoadable {
                name: name.to_string(),
                reason: refused.to_string(),
            }
            .into())
        }
    }
    crate::loader::load_model_meta::<R>(&path)
        .map(|_| ())
        .with_context(|| format!("model '{name}' does not load"))
}

/// Information about a loaded model
#[derive(Debug, Clone)]
pub struct LoadedModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub last_accessed: Instant,
}
