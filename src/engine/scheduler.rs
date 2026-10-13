//! Model scheduler for lifecycle management
//!
//! Manages loading, unloading, and eviction of models.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::config::{DtypeRequest, InferenceOverlay, VisionOverrides, DEFAULT_NUM_CTX};
use tokio::sync::RwLock;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, UnaryOps,
};

use super::load::gate::LoadGates;
use super::load::job::LoadJob;
use super::load::registered::Models;
use super::lora::eviction::{self, Resident};
use super::lora::LoraRegistry;
pub use super::model_dir::AvailableModelInfo;
use crate::config::DecisionConfig;
use crate::engine::decision::DecisionRegistry;
use crate::engine::Executor;

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
}

/// Every loaded model, as eviction sees it.
pub(crate) fn residents<R: Runtime<DType = DType>>(models: &Models<R>) -> Vec<Resident<'_>> {
    models
        .iter()
        .map(|(name, entry)| Resident {
            name: name.as_str(),
            variant_of: entry.variant_of.as_deref(),
            last_accessed: entry.last_accessed,
            keep_alive: entry.keep_alive,
        })
        .collect()
}

/// Drop `name` and, for a base model, each of its variants first.
/// Returns whether `name` was loaded.
fn remove_with_variants<R: Runtime<DType = DType>>(models: &mut Models<R>, name: &str) -> bool {
    let order = eviction::unload_order(&residents(models), name);
    for victim in &order {
        models.remove(victim);
    }
    !order.is_empty()
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
        }
    }

    /// Detect, batch and bound System One decision models with `config`.
    pub fn with_decision(mut self, config: DecisionConfig) -> Self {
        self.decision_config = config;
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
    /// A [`LoraError`](super::LoraError) when a variant does not build, or
    /// when `model_name` has the shape `<model>:<adapter>` for a known
    /// model and no such adapter is registered. Otherwise the load error.
    pub async fn get_executor(&self, model_name: &str) -> Result<Arc<Executor<R>>> {
        if let Some(executor) = self.touch(model_name).await {
            return Ok(executor);
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
    /// # Errors
    ///
    /// No such model, a load that fails, or a load that panics. A failed
    /// load registers nothing.
    pub(crate) async fn load_model(&self, model_name: &str) -> Result<Arc<Executor<R>>> {
        let job = self.load_job();
        self.load_base(model_name, move |path| job.run(path)).await
    }

    /// This scheduler's load settings, owned for a load thread.
    fn load_job(&self) -> LoadJob<R> {
        LoadJob {
            model_dir: self.model_dir.clone(),
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
        super::model_dir::find_model_path(&self.model_dir, model_name)
    }

    /// Set the keep-alive duration for a model.
    /// `None` means forever. `Some(Duration::ZERO)` means unload immediately,
    /// a base model's LoRA variants first.
    pub async fn set_keep_alive(&self, model_name: &str, keep_alive: Option<Duration>) {
        let mut models = self.models.write().await;
        if let Some(entry) = models.get_mut(model_name) {
            entry.keep_alive = keep_alive;
        }
        if keep_alive == Some(Duration::ZERO) && remove_with_variants(&mut models, model_name) {
            tracing::info!("Unloaded model '{}' (keep_alive=0)", model_name);
        }
    }

    /// Evict models that have exceeded their keep-alive duration, with the
    /// LoRA variants of each evicted base. Called periodically by the
    /// reaper task.
    pub async fn evict_expired(&self) {
        let mut models = self.models.write().await;
        let order = eviction::expired(&residents(&models), Instant::now());
        for name in order {
            models.remove(&name);
            tracing::info!("Auto-unloaded model '{}' (keep_alive expired)", name);
        }
    }

    /// Unload a specific model: a base model's LoRA variants first, then
    /// the model. A variant unloads alone and stays registered, so the next
    /// request for it builds it again.
    pub async fn unload(&self, model_name: &str) -> bool {
        let mut models = self.models.write().await;
        remove_with_variants(&mut models, model_name)
    }

    /// List loaded models
    pub async fn list_loaded(&self) -> Vec<LoadedModelInfo> {
        let models = self.models.read().await;
        models
            .iter()
            .map(|(name, entry)| LoadedModelInfo {
                name: name.clone(),
                path: entry.path.clone(),
                last_accessed: entry.last_accessed,
            })
            .collect()
    }

    /// List available models in the model directory
    pub fn list_available(&self) -> Result<Vec<AvailableModelInfo>> {
        Ok(super::model_dir::list_available(&self.model_dir)?)
    }
}

/// Information about a loaded model
#[derive(Debug, Clone)]
pub struct LoadedModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub last_accessed: Instant,
}
