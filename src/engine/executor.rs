//! Inference executor
//!
//! Runs inference on loaded models using boostr's LoadedModel.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::format::PeftAdapter;
use boostr::inference::memory::{BlockTable, CpuBlockAllocator};
use boostr::inference::prefix_cache::{PrefixCache, PrefixCacheConfig};
use boostr::model::LoadedModel;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};

use splintr::AnyTokenizer;

use crate::config::BlazrConfig;
use crate::model::chat_template::ChatTemplate;

use super::lora::LoraAdapterRegistry;
use super::vision_tower::{ImageTokenLimits, VisionTower};

/// `config` with `inference.dtype` set to the dtype `R` computes.
///
/// The loader already resolved it through `config::compute_dtype`, so a
/// loaded config passes unchanged. A config built by hand for a runtime
/// that computes in f32 only gets f32 here.
fn with_runtime_dtype<R: Runtime<DType = DType>>(mut config: BlazrConfig) -> Result<BlazrConfig> {
    let dtype = config.compute_dtype()?;
    let runs = crate::config::compute_dtype::<R>(None, Some(dtype));
    if runs != dtype {
        config.inference.dtype = crate::config::dtype_name(runs)?.to_string();
    }
    Ok(config)
}

/// Inference executor
///
/// Wraps a loaded model and provides text generation capabilities.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct Executor<R: Runtime<DType = DType>> {
    /// The loaded model
    pub(crate) model: Arc<LoadedModel<R>>,
    /// Model configuration
    pub(crate) config: BlazrConfig,
    /// Tokenizer — splintr's universal loaded-tokenizer handle, whatever its source
    pub(crate) tokenizer: AnyTokenizer,
    /// Device
    pub(crate) device: R::Device,
    /// Initial context size for KV cache (like Ollama's num_ctx)
    pub(crate) num_ctx: usize,
    /// Chat template for this model
    pub(crate) chat_template: ChatTemplate,
    /// Shared block allocator for paged attention (shared across all concurrent requests)
    pub(crate) shared_allocator: Option<Arc<std::sync::Mutex<CpuBlockAllocator>>>,
    /// Shared prefix cache for paged attention (persists across requests)
    pub(crate) prefix_cache: Option<std::sync::Mutex<PrefixCache<CpuBlockAllocator>>>,
    /// Model directory (for resolving relative draft model paths)
    pub(crate) model_dir: Option<PathBuf>,
    /// Lazily-loaded draft model for speculative decoding
    pub(crate) draft_model: std::sync::Mutex<Option<Arc<LoadedModel<R>>>>,
    /// LoRA adapter registry — holds all named adapters loaded for this model instance
    pub(crate) lora_registry: Arc<LoraAdapterRegistry<R>>,
    /// Vision tower for image inputs, loaded from `inference.mmproj`.
    pub(crate) vision_tower: Option<Arc<VisionTower<R>>>,
    /// Tensor parallelism state (CUDA only; present when tp_size > 1)
    #[cfg(feature = "cuda")]
    pub(crate) tp_state: Option<super::tensor_parallel::TensorParallelState>,
    /// GPU-accelerated prefix cache (CUDA only; opt-in via config)
    #[cfg(feature = "cuda")]
    pub(crate) gpu_prefix_cache:
        Option<std::sync::Mutex<boostr::inference::prefix_cache::GpuPrefixCache>>,
    /// Captured Llama decode graph and its caches, reused across requests.
    /// Held for a whole graph-mode request, so such requests run one at a time.
    #[cfg(feature = "cuda")]
    pub(crate) llama_graph_session:
        tokio::sync::Mutex<Option<super::graph_session::LlamaGraphSession>>,
    /// Captured `qwen35` decode graph and its caches, reused across requests.
    #[cfg(feature = "cuda")]
    pub(crate) qwen35_graph_session:
        tokio::sync::Mutex<Option<super::graph_session::Qwen35GraphSession>>,
    /// Multi-lane `qwen35` decode session: batched caches, one lane table and
    /// the decode driver. The lock is taken for one admit, step or release
    /// and dropped, never held across a request's token stream.
    #[cfg(feature = "cuda")]
    pub(crate) qwen35_lane_session:
        tokio::sync::Mutex<Option<super::lane_session::Qwen35LaneSession>>,
    /// Graph captures on this executor, across both arms.
    #[cfg(feature = "cuda")]
    pub(crate) graph_captures: std::sync::atomic::AtomicUsize,
}

impl<R: Runtime<DType = DType>> Executor<R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>,
{
    /// Create a new executor
    pub fn new(
        model: LoadedModel<R>,
        config: BlazrConfig,
        tokenizer: AnyTokenizer,
        device: R::Device,
        num_ctx: usize,
    ) -> Result<Self> {
        let config = with_runtime_dtype::<R>(config)?;
        if config.inference.paged_attention {
            if let Some(reason) = model.paged_attention_refusal() {
                return Err(anyhow!(reason));
            }
        }
        let chat_template = ChatTemplate::from_model_type(config.model_type());

        // Shared block allocator for paged attention (shared across all concurrent requests)
        let shared_allocator = if config.inference.paged_attention {
            let block_size = config.inference.block_size;
            let total_blocks = if config.inference.kv_pool_blocks > 0 {
                config.inference.kv_pool_blocks
            } else {
                // Auto-compute: enough for max_batch_size concurrent requests at max context
                let max_ctx = config.inference.max_context_len.unwrap_or(num_ctx);
                let blocks_per_seq = BlockTable::blocks_needed(max_ctx, block_size);
                let batch = config.inference.max_batch_size.max(1);
                // Pool = batch * per-seq blocks + 20% headroom for prefix cache + fragmentation
                let base = blocks_per_seq * batch;
                base + base / 5 + 64
            };
            tracing::info!(
                "Shared KV block pool: {} blocks × {} tokens/block = {} token slots",
                total_blocks,
                block_size,
                total_blocks * block_size
            );
            Some(Arc::new(std::sync::Mutex::new(CpuBlockAllocator::new(
                total_blocks,
                block_size,
            ))))
        } else {
            None
        };

        let prefix_cache = if config.inference.prefix_cache && config.inference.paged_attention {
            let block_size = config.inference.block_size;
            let max_cached = config.inference.max_cached_blocks;
            // Use the shared allocator for prefix cache
            let allocator = if let Some(ref shared) = shared_allocator {
                shared
                    .lock()
                    .map_err(|e| anyhow!("Block allocator lock poisoned: {e}"))?
                    .clone()
            } else {
                let total_blocks = max_cached + 1024;
                CpuBlockAllocator::new(total_blocks, block_size)
            };
            let cache_config = PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: max_cached,
                block_size,
                ..PrefixCacheConfig::default()
            };
            tracing::info!(
                "Prefix cache enabled: max_cached_blocks={}, block_size={}",
                max_cached,
                block_size
            );
            Some(std::sync::Mutex::new(PrefixCache::new(
                allocator,
                cache_config,
            )))
        } else {
            None
        };

        // GPU prefix cache (CUDA only, opt-in)
        #[cfg(feature = "cuda")]
        let gpu_prefix_cache = if config.inference.gpu_prefix_cache
            && config.inference.prefix_cache
            && config.inference.paged_attention
        {
            let capacity = config.inference.max_cached_blocks;
            let block_size = config.inference.block_size;
            let ram_tier = config.inference.gpu_prefix_cache_ram_tier;
            tracing::info!(
                "GPU prefix cache enabled: capacity={}, block_size={}, ram_tier={}",
                capacity,
                block_size,
                ram_tier,
            );
            {
                // Construct CudaDevice matching the executor's actual GPU via Device::id()
                use boostr::runtime::Device;
                let cuda_device = boostr::CudaDevice::new(device.id());
                Some(std::sync::Mutex::new(
                    boostr::inference::prefix_cache::GpuPrefixCache::new(
                        capacity,
                        block_size,
                        cuda_device,
                        ram_tier,
                    ),
                ))
            }
        } else {
            None
        };

        Ok(Self {
            model: Arc::new(model),
            config,
            tokenizer,
            device,
            num_ctx,
            chat_template,
            shared_allocator,
            prefix_cache,
            model_dir: None,
            draft_model: std::sync::Mutex::new(None),
            lora_registry: Arc::new(LoraAdapterRegistry::new()),
            vision_tower: None,
            #[cfg(feature = "cuda")]
            tp_state: None,
            #[cfg(feature = "cuda")]
            gpu_prefix_cache,
            #[cfg(feature = "cuda")]
            llama_graph_session: tokio::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            qwen35_graph_session: tokio::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            qwen35_lane_session: tokio::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            graph_captures: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Create a new executor for a tensor-parallel model.
    ///
    /// The `tp_state` keeps the NCCL communicator alive for the duration of
    /// the executor's lifetime. The model must already be loaded via
    /// `loader::load_model_tp` before calling this.
    #[cfg(feature = "cuda")]
    pub fn new_tensor_parallel(
        model: LoadedModel<R>,
        config: BlazrConfig,
        tokenizer: AnyTokenizer,
        device: R::Device,
        num_ctx: usize,
        tp_state: super::tensor_parallel::TensorParallelState,
    ) -> Result<Self> {
        let mut executor = Self::new(model, config, tokenizer, device, num_ctx)?;
        executor.tp_state = Some(tp_state);
        Ok(executor)
    }

    /// Replace the chat template. A Jinja template gets its special-token
    /// allow-list computed against this executor's tokenizer here.
    pub fn with_chat_template(mut self, mut template: ChatTemplate) -> Self {
        template.bind_special_tokens(&self.tokenizer);
        self.chat_template = template;
        self
    }

    /// Set the model directory (for resolving relative draft model paths)
    pub fn with_model_dir(mut self, dir: PathBuf) -> Self {
        self.model_dir = Some(dir);
        self
    }

    /// Load the vision tower `config.inference.mmproj` names, with the
    /// `image_min_tokens` / `image_max_tokens` limits, onto this executor's
    /// device. No `mmproj` set: returns `self` unchanged.
    ///
    /// # Errors
    ///
    /// The loaded model is not `qwen35`, or the tower does not load.
    pub fn with_configured_vision_tower(self) -> Result<Self> {
        let Some(mmproj) = self.config.inference.mmproj.clone() else {
            return Ok(self);
        };
        if !matches!(&*self.model, LoadedModel::Qwen35(_)) {
            return Err(anyhow!(
                "--mmproj needs a qwen35 model; the loaded model is {}",
                self.model.model_type()
            ));
        }
        let limits = ImageTokenLimits {
            min_tokens: self.config.inference.image_min_tokens,
            max_tokens: self.config.inference.image_max_tokens,
        };
        let tower = VisionTower::load(&mmproj, &self.device, &self.tokenizer, limits)?;
        tracing::info!(
            mmproj = %mmproj.display(),
            image_min_tokens = tower.config().image_min_tokens,
            image_max_tokens = tower.config().image_max_tokens,
            "vision tower loaded"
        );
        Ok(self.with_vision_tower(tower))
    }

    /// Attach a loaded vision tower. The tower must sit on this executor's
    /// device and carry the marker ids of its tokenizer.
    pub fn with_vision_tower(mut self, tower: VisionTower<R>) -> Self {
        self.vision_tower = Some(Arc::new(tower));
        self
    }

    /// The vision tower, when `mmproj` was configured.
    pub fn vision_tower(&self) -> Option<&Arc<VisionTower<R>>> {
        self.vision_tower.as_ref()
    }

    /// Get or lazily load the draft model for speculative decoding.
    /// The draft model is loaded from the path specified in the speculative config.
    pub(crate) fn get_or_load_draft_model(&self) -> Result<Arc<LoadedModel<R>>>
    where
        R::Client: boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
    {
        let mut guard = self
            .draft_model
            .lock()
            .map_err(|e| anyhow!("draft model lock: {e}"))?;

        if let Some(ref model) = *guard {
            return Ok(Arc::clone(model));
        }

        let spec_config = self
            .config
            .inference
            .speculative
            .as_ref()
            .ok_or_else(|| anyhow!("no speculative config"))?;

        let draft_path = PathBuf::from(&spec_config.draft_model);
        let draft_path = if draft_path.is_absolute() || draft_path.exists() {
            draft_path
        } else if let Some(ref model_dir) = self.model_dir {
            let in_dir = model_dir.join(&spec_config.draft_model);
            if in_dir.exists() {
                in_dir
            } else {
                return Err(anyhow!(
                    "Draft model not found: {} (searched {} and model_dir {})",
                    spec_config.draft_model,
                    draft_path.display(),
                    model_dir.display()
                ));
            }
        } else {
            return Err(anyhow!(
                "Draft model not found: {} (no model_dir set)",
                spec_config.draft_model
            ));
        };

        tracing::info!("Loading draft model from {:?}", draft_path);
        // The draft shares the target's KV cache dtype, so it loads at the
        // target's compute dtype.
        let request = crate::config::DtypeRequest {
            dtype: self.config.compute_dtype()?,
            source: crate::config::DtypeSource::Target,
        };
        let (model, _config) =
            crate::loader::load_model_as::<R, _>(&draft_path, &self.device, Some(&request))?;
        let model = Arc::new(model);
        *guard = Some(Arc::clone(&model));
        tracing::info!("Draft model loaded (vocab_size={})", model.vocab_size());
        Ok(model)
    }

    /// Get the chat template for this model
    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// Get model configuration
    pub fn config(&self) -> &BlazrConfig {
        &self.config
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &AnyTokenizer {
        &self.tokenizer
    }

    /// Get prefix cache statistics (if enabled)
    pub fn prefix_cache_stats(&self) -> Option<boostr::inference::PrefixCacheStats> {
        self.prefix_cache
            .as_ref()
            .and_then(|pc| pc.lock().ok().map(|c| c.stats()))
    }

    /// Get GPU prefix cache statistics (CUDA only).
    #[cfg(feature = "cuda")]
    pub fn gpu_prefix_cache_stats(&self) -> Option<boostr::inference::prefix_cache::GpuRadixStats> {
        self.gpu_prefix_cache
            .as_ref()
            .and_then(|pc| pc.lock().ok().map(|c| c.stats()))
    }

    /// Access the GPU prefix cache (CUDA only, for batch engine integration).
    #[cfg(feature = "cuda")]
    pub fn gpu_prefix_cache(
        &self,
    ) -> Option<&std::sync::Mutex<boostr::inference::prefix_cache::GpuPrefixCache>> {
        self.gpu_prefix_cache.as_ref()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// Get the device
    pub(crate) fn device(&self) -> &R::Device {
        &self.device
    }

    /// Get the loaded model as an Arc (for sharing with speculative/batch engines)
    pub(crate) fn model_arc(&self) -> &Arc<LoadedModel<R>> {
        &self.model
    }

    /// Get the loaded model
    pub(crate) fn model(&self) -> &LoadedModel<R> {
        &self.model
    }

    /// Get the shared block allocator (if paged attention enabled)
    pub(crate) fn shared_allocator(&self) -> Option<&Arc<std::sync::Mutex<CpuBlockAllocator>>> {
        self.shared_allocator.as_ref()
    }

    /// Get num_ctx
    #[cfg(feature = "cuda")]
    pub(crate) fn num_ctx(&self) -> usize {
        self.num_ctx
    }

    // ── LoRA adapter management ──────────────────────────────────────────────

    /// Load a LoRA adapter from disk and register it under `name`.
    ///
    /// `path` is a PEFT adapter directory holding `adapter_config.json` and
    /// `adapter_model.safetensors`, or the `.safetensors` file with
    /// `adapter_config.json` beside it. boostr's `PeftAdapter::load` reads it
    /// and refuses any adapter it cannot represent exactly.
    ///
    /// If an adapter with the same name was previously loaded it is replaced.
    pub fn load_lora(&self, path: &std::path::Path, name: &str) -> Result<()> {
        let adapter = PeftAdapter::<R>::load(path, &self.device)
            .map_err(|e| anyhow!("LoRA adapter '{name}' at {}: {e}", path.display()))?;
        tracing::info!(
            adapter = name,
            rank = adapter.config.r,
            alpha = adapter.config.lora_alpha,
            rslora = adapter.config.use_rslora,
            layers = adapter.layers.len(),
            "LoRA adapter registered"
        );
        self.lora_registry.insert(name, adapter);
        Ok(())
    }

    /// Unload a named LoRA adapter.  Returns `true` if the adapter existed.
    pub fn unload_lora(&self, name: &str) -> bool {
        let removed = self.lora_registry.remove(name);
        if removed {
            tracing::info!(adapter = name, "LoRA adapter unloaded");
        }
        removed
    }

    /// List names of all currently loaded LoRA adapters.
    pub fn list_loras(&self) -> Vec<String> {
        self.lora_registry.list()
    }

    /// Access the LoRA adapter registry (for use in generation paths).
    pub fn lora_registry(&self) -> &Arc<LoraAdapterRegistry<R>> {
        &self.lora_registry
    }
}
