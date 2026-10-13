//! Model scheduler for lifecycle management
//!
//! Manages loading, unloading, and eviction of models.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};

use crate::config::{
    BlazrConfig, DtypeRequest, InferenceOverlay, VisionOverrides, DEFAULT_NUM_CTX,
};
use crate::model::chat_template::ChatTemplate;
use tokio::sync::RwLock;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, UnaryOps,
};

use super::lora::eviction::{self, Resident};
use super::lora::LoraRegistry;
pub use super::model_dir::AvailableModelInfo;
use crate::config::DecisionConfig;
use crate::engine::decision::DecisionRegistry;
use crate::engine::Executor;
use crate::loader;
use crate::tokenizer::checkpoint_tokenizer;

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
fn residents<R: Runtime<DType = DType>>(
    models: &HashMap<String, ModelEntry<R>>,
) -> Vec<Resident<'_>> {
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
fn remove_with_variants<R: Runtime<DType = DType>>(
    models: &mut HashMap<String, ModelEntry<R>>,
    name: &str,
) -> bool {
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
    /// Loaded models by name: base models and LoRA variants
    pub(crate) models: RwLock<HashMap<String, ModelEntry<R>>>,
    /// Maximum number of loaded models
    max_loaded: usize,
    /// Model directory for discovery
    model_dir: PathBuf,
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
    /// The registered LoRA adapter variants.
    pub(crate) lora: LoraRegistry,
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
            models: RwLock::new(HashMap::new()),
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
            lora: LoraRegistry::new(),
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
    /// from its base, loading the base first when needed.
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
        if let Some(error) = self.unknown_variant(model_name).await {
            return Err(error.into());
        }
        self.load_model(model_name).await
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

    /// Load a model by name
    pub(crate) async fn load_model(&self, model_name: &str) -> Result<Arc<Executor<R>>> {
        // Evict if necessary
        self.ensure_capacity().await?;

        // Find model path
        let model_path = self.find_model_path(model_name)?;

        // Tensor parallel degree: BLAZR_TP_SIZE, then the user config file,
        // then the model's own config.json, then 1.
        let tp_size = self.tensor_parallel_size.unwrap_or_else(|| {
            loader::model_config_inference(&model_path)
                .map_or(1, |inference| inference.tensor_parallel_size)
        });
        super::tensor_parallel::validate_tp_config(tp_size, None).map_err(|e| anyhow!("{e}"))?;

        let executor = if tp_size > 1 {
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            {
                use boostr::runtime::Device;
                // Rank r runs on GPU `self.device + r`, each as a thread of
                // this process. Followers load their shards while rank 0 does.
                let (tp_state, comm) = super::tensor_parallel::init_tp(
                    tp_size,
                    self.device.id(),
                    &model_path,
                    self.dtype,
                )
                .map_err(|e| anyhow!("{e}"))?;
                let (model, mut config) = loader::load_model_tp::<R, _>(
                    &model_path,
                    &self.device,
                    comm,
                    self.dtype.as_ref(),
                )?;
                if let Some(group) = &tp_state.group {
                    group.wait_loaded()?;
                }
                self.apply_settings(&mut config, tp_size)?;
                super::tensor_parallel::restrict_to_lockstep(&mut config.inference);
                let tokenizer = checkpoint_tokenizer(&model_path, config.vocab_size())?;
                let chat_template = ChatTemplate::detect(
                    &model_path,
                    config.model_type(),
                    self.chat_template.as_deref(),
                )?;
                Arc::new(
                    Executor::new_tensor_parallel(
                        model,
                        config,
                        tokenizer,
                        self.device.clone(),
                        self.num_ctx,
                        tp_state,
                    )?
                    .with_chat_template(chat_template)
                    .with_model_dir(self.model_dir.clone()),
                )
            }
            #[cfg(not(all(feature = "cuda", feature = "nccl")))]
            {
                return Err(anyhow::anyhow!(
                    "tensor_parallel_size {tp_size} needs the cuda and nccl features. Rebuild \
                     with --features cuda,nccl, or set the size to 1"
                ));
            }
        } else {
            // A GGUF carries its own vocabulary; a checkpoint directory is
            // matched to a bundled one by size.
            let source = loader::detect_model_source(&model_path)?;
            let (model, mut config, tokenizer) = match source.format {
                loader::ModelFormat::Gguf => loader::load_gguf_with_tokenizer_as::<R, _>(
                    &model_path,
                    &self.device,
                    self.dtype.as_ref(),
                )?,
                loader::ModelFormat::SafeTensors => {
                    let (model, config) = loader::load_model_as::<R, _>(
                        &model_path,
                        &self.device,
                        self.dtype.as_ref(),
                    )?;
                    let tokenizer = checkpoint_tokenizer(&model_path, config.vocab_size())?;
                    (model, config, tokenizer)
                }
            };
            self.apply_settings(&mut config, tp_size)?;
            let chat_template = ChatTemplate::detect(
                &model_path,
                config.model_type(),
                self.chat_template.as_deref(),
            )?;
            Arc::new(
                Executor::new(model, config, tokenizer, self.device.clone(), self.num_ctx)?
                    .with_chat_template(chat_template)
                    .with_model_dir(self.model_dir.clone())
                    .with_configured_vision_tower()?,
            )
        };

        metrics::counter!("blazr_scheduler_loads_total").increment(1);

        // Store in cache
        {
            let mut models = self.models.write().await;
            models.insert(
                model_name.to_string(),
                ModelEntry {
                    executor: Arc::clone(&executor),
                    last_accessed: Instant::now(),
                    path: model_path,
                    keep_alive: None,
                    variant_of: None,
                },
            );
        }

        Ok(executor)
    }

    /// Write the user config file's inference keys, then the command-line
    /// vision flags, over a freshly loaded model's config. Records the
    /// tensor parallel degree the model runs at.
    fn apply_settings(&self, config: &mut BlazrConfig, tp_size: usize) -> Result<()> {
        self.inference_overlay.apply(&mut config.inference)?;
        self.vision.apply(&mut config.inference);
        config.inference.tensor_parallel_size = tp_size;
        Ok(())
    }

    /// Find model path by name
    fn find_model_path(&self, model_name: &str) -> Result<PathBuf> {
        self.model_path(model_name)
            .ok_or_else(|| anyhow!("Model not found: {}", model_name))
    }

    /// The path the model name `model_name` resolves to, as a load resolves
    /// it. `None` when no such model exists.
    pub fn model_path(&self, model_name: &str) -> Option<PathBuf> {
        super::model_dir::find_model_path(&self.model_dir, model_name)
    }

    /// Ensure capacity for a new model. Only base models count against
    /// the limit. Evicting a base evicts its LoRA variants first.
    async fn ensure_capacity(&self) -> Result<()> {
        let mut models = self.models.write().await;
        loop {
            let order = {
                let residents = residents(&models);
                if eviction::base_count(&residents) < self.max_loaded {
                    break;
                }
                let Some(lru) = eviction::lru_base(&residents) else {
                    break;
                };
                eviction::unload_order(&residents, &lru)
            };
            for name in order {
                tracing::info!("Evicting model: {}", name);
                models.remove(&name);
            }
            metrics::counter!("blazr_scheduler_evictions_total").increment(1);
        }
        Ok(())
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
