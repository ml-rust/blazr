//! Model scheduler for lifecycle management
//!
//! Manages loading, unloading, and eviction of models.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};

use crate::model::chat_template::ChatTemplate;
use tokio::sync::RwLock;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, UnaryOps,
};

use crate::engine::Executor;
use crate::loader;
use crate::tokenizer::Tokenizer;

/// Parse an Ollama-style keep_alive string into a Duration.
///
/// Supported formats:
/// - `"0"` → `Some(Duration::ZERO)` (unload immediately)
/// - `"-1"` → `None` (keep forever)
/// - `"5m"` → 5 minutes
/// - `"1h"` → 1 hour
/// - `"30s"` → 30 seconds
/// - `"300"` → 300 seconds (bare number = seconds)
pub fn parse_keep_alive(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s == "-1" {
        return None; // forever
    }
    if s == "0" {
        return Some(Duration::ZERO);
    }
    if let Some(mins) = s.strip_suffix('m') {
        if let Ok(n) = mins.parse::<u64>() {
            return Some(Duration::from_secs(n * 60));
        }
    }
    if let Some(hours) = s.strip_suffix('h') {
        if let Ok(n) = hours.parse::<u64>() {
            return Some(Duration::from_secs(n * 3600));
        }
    }
    if let Some(secs) = s.strip_suffix('s') {
        if let Ok(n) = secs.parse::<u64>() {
            return Some(Duration::from_secs(n));
        }
    }
    // Bare number = seconds
    if let Ok(n) = s.parse::<u64>() {
        return Some(Duration::from_secs(n));
    }
    None // unparseable → default (forever)
}

/// Model entry in the scheduler
struct ModelEntry<R: Runtime<DType = DType>> {
    /// The executor wrapping the model
    executor: Arc<Executor<R>>,
    /// When the model was last accessed
    last_accessed: Instant,
    /// Model path for reloading
    path: PathBuf,
    /// Per-model keep-alive duration. None = forever (no auto-unload).
    keep_alive: Option<Duration>,
}

/// Default context size for scheduler-managed models
const DEFAULT_NUM_CTX: usize = 2048;

/// Model scheduler
///
/// Manages model lifecycle including:
/// - Loading models on demand
/// - Caching loaded models
/// - Evicting least-recently-used models when memory is tight
pub struct Scheduler<R: Runtime<DType = DType>> {
    /// Loaded models by name
    models: RwLock<HashMap<String, ModelEntry<R>>>,
    /// Maximum number of loaded models
    max_loaded: usize,
    /// Model directory for discovery
    model_dir: PathBuf,
    /// Device for loading
    device: R::Device,
    /// Default context size for KV cache
    num_ctx: usize,
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
        }
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

    /// Get an executor for a model, loading it if necessary
    pub async fn get_executor(&self, model_name: &str) -> Result<Arc<Executor<R>>> {
        // Check if already loaded
        {
            let mut models = self.models.write().await;
            if let Some(entry) = models.get_mut(model_name) {
                entry.last_accessed = Instant::now();
                return Ok(Arc::clone(&entry.executor));
            }
        }

        // Need to load the model
        self.load_model(model_name).await
    }

    /// Load a model by name
    async fn load_model(&self, model_name: &str) -> Result<Arc<Executor<R>>> {
        // Evict if necessary
        self.ensure_capacity().await?;

        // Find model path
        let model_path = self.find_model_path(model_name)?;

        // Load model — check for tensor parallelism from env or config
        let tp_size = std::env::var("BLAZR_TP_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1);

        let executor = if tp_size > 1 {
            #[cfg(feature = "cuda")]
            {
                let rank = std::env::var("BLAZR_TP_RANK")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let (tp_state, comm) = super::tensor_parallel::init_tp(tp_size, rank)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                let (model, config) =
                    loader::load_model_tp::<R, _>(&model_path, &self.device, comm)?;
                let tokenizer = Tokenizer::from_vocab_size(config.vocab_size())?;
                let chat_template = ChatTemplate::detect(&model_path, config.model_type());
                if let Some(ref moe_config) =
                    super::moe_offload::MoeOffloadConfig::from_inference_config(&config.inference)
                {
                    tracing::info!(
                        strategy = ?moe_config.strategy,
                        gpu_experts = moe_config.gpu_experts,
                        "MoE expert offloading configured"
                    );
                }
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
            #[cfg(not(feature = "cuda"))]
            {
                return Err(anyhow::anyhow!(
                    "Tensor parallelism requires the 'cuda' feature (BLAZR_TP_SIZE={})",
                    tp_size
                ));
            }
        } else {
            let (model, config) = loader::load_model::<R, _>(&model_path, &self.device)?;
            let tokenizer = Tokenizer::from_vocab_size(config.vocab_size())?;
            let chat_template = ChatTemplate::detect(&model_path, config.model_type());
            if let Some(ref moe_config) =
                super::moe_offload::MoeOffloadConfig::from_inference_config(&config.inference)
            {
                tracing::info!(
                    strategy = ?moe_config.strategy,
                    gpu_experts = moe_config.gpu_experts,
                    "MoE expert offloading configured"
                );
            }
            Arc::new(
                Executor::new(model, config, tokenizer, self.device.clone(), self.num_ctx)?
                    .with_chat_template(chat_template)
                    .with_model_dir(self.model_dir.clone()),
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
                },
            );
        }

        Ok(executor)
    }

    /// Find model path by name
    fn find_model_path(&self, model_name: &str) -> Result<PathBuf> {
        // Try direct path first
        let direct = PathBuf::from(model_name);
        if direct.exists() {
            return Ok(direct);
        }

        // Try in model directory
        let in_dir = self.model_dir.join(model_name);
        if in_dir.exists() {
            return Ok(in_dir);
        }

        // Try common patterns
        let patterns = [
            format!("{}.safetensors", model_name),
            format!("{}.gguf", model_name),
            format!("{}/model.safetensors", model_name),
        ];

        for pattern in &patterns {
            let candidate = self.model_dir.join(pattern);
            if candidate.exists() {
                return Ok(candidate);
            }
        }

        Err(anyhow!("Model not found: {}", model_name))
    }

    /// Ensure capacity for a new model
    async fn ensure_capacity(&self) -> Result<()> {
        let mut models = self.models.write().await;

        while models.len() >= self.max_loaded {
            // Find LRU model
            let lru_name = models
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
                .map(|(name, _)| name.clone());

            if let Some(name) = lru_name {
                tracing::info!("Evicting model: {}", name);
                models.remove(&name);
                metrics::counter!("blazr_scheduler_evictions_total").increment(1);
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Set the keep-alive duration for a model.
    /// `None` means forever. `Some(Duration::ZERO)` means unload immediately.
    pub async fn set_keep_alive(&self, model_name: &str, keep_alive: Option<Duration>) {
        let mut models = self.models.write().await;
        if let Some(entry) = models.get_mut(model_name) {
            entry.keep_alive = keep_alive;
        }
        // Immediate unload if Duration::ZERO
        if keep_alive == Some(Duration::ZERO) {
            models.remove(model_name);
            tracing::info!("Unloaded model '{}' (keep_alive=0)", model_name);
        }
    }

    /// Evict models that have exceeded their keep-alive duration.
    /// Called periodically by the reaper task.
    pub async fn evict_expired(&self) {
        let mut models = self.models.write().await;
        let mut to_remove = Vec::new();
        for (name, entry) in models.iter() {
            if let Some(ttl) = entry.keep_alive {
                if entry.last_accessed.elapsed() > ttl {
                    to_remove.push(name.clone());
                }
            }
        }
        for name in to_remove {
            models.remove(&name);
            tracing::info!("Auto-unloaded model '{}' (keep_alive expired)", name);
        }
    }

    /// Unload a specific model
    pub async fn unload(&self, model_name: &str) -> bool {
        let mut models = self.models.write().await;
        models.remove(model_name).is_some()
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
        let mut models = Vec::new();

        if !self.model_dir.exists() {
            return Ok(models);
        }

        for entry in std::fs::read_dir(&self.model_dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();

            if path.is_dir() {
                // Check for model files in directory
                if path.join("model.safetensors").exists()
                    || path.join("config.json").exists()
                    || std::fs::read_dir(&path)
                        .map(|entries| {
                            entries.filter_map(|e| e.ok()).any(|e| {
                                e.path()
                                    .extension()
                                    .map(|ext| ext == "gguf")
                                    .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false)
                {
                    models.push(AvailableModelInfo {
                        name: name.clone(),
                        path: path.clone(),
                        format: detect_format(&path),
                    });
                }
            } else if path
                .extension()
                .map(|e| e == "safetensors" || e == "gguf")
                .unwrap_or(false)
            {
                models.push(AvailableModelInfo {
                    name: name.clone(),
                    path: path.clone(),
                    format: detect_format(&path),
                });
            }
        }

        Ok(models)
    }
}

/// Information about a loaded model
#[derive(Debug, Clone)]
pub struct LoadedModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub last_accessed: Instant,
}

/// Information about an available model
#[derive(Debug, Clone)]
pub struct AvailableModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub format: String,
}

/// Detect model format from path
fn detect_format(path: &std::path::Path) -> String {
    if path.is_file() {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => "SafeTensors",
            Some("gguf") => "GGUF",
            _ => "Unknown",
        }
    } else if path.join("model.safetensors").exists() {
        "SafeTensors"
    } else if std::fs::read_dir(path)
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "gguf")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
    {
        "GGUF"
    } else {
        "Unknown"
    }
    .to_string()
}
