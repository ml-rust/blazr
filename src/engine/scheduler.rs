//! Model scheduler for lifecycle management
//!
//! Manages loading, unloading, and eviction of models.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use tokio::sync::RwLock;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, UnaryOps,
};

use crate::engine::Executor;
use crate::loader;
use crate::tokenizer::Tokenizer;

/// Model entry in the scheduler
struct ModelEntry<R: Runtime<DType = DType>> {
    /// The executor wrapping the model
    executor: Arc<Executor<R>>,
    /// When the model was last accessed
    last_accessed: Instant,
    /// Model path for reloading
    path: PathBuf,
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
        + ModelClient<R>
        + boostr::quant::DequantOps<R>,
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

        // Load model
        let (model, config) = loader::load_model::<R, _>(&model_path, &self.device)?;

        // Create tokenizer
        let tokenizer = Tokenizer::from_vocab_size(config.vocab_size())?;

        // Create executor with num_ctx for KV cache initial capacity
        let executor = Arc::new(Executor::new(
            model,
            config,
            tokenizer,
            self.device.clone(),
            self.num_ctx,
        )?);

        // Store in cache
        {
            let mut models = self.models.write().await;
            models.insert(
                model_name.to_string(),
                ModelEntry {
                    executor: Arc::clone(&executor),
                    last_accessed: Instant::now(),
                    path: model_path,
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
            } else {
                break;
            }
        }

        Ok(())
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
