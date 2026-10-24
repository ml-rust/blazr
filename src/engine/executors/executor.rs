//! Inference executor
//!
//! Runs inference on loaded models using boostr's LoadedModel.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
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

/// Every client trait the engine needs from a runtime.
pub(crate) trait EngineClient<R: Runtime<DType = DType>>:
    TensorOps<R>
    + ScalarOps<R>
    + ConvOps<R>
    + NormalizationOps<R>
    + UnaryOps<R>
    + ActivationOps<R>
    + BinaryOps<R>
    + TypeConversionOps<R>
    + SamplingOps<R>
    + boostr::GrammarDfaOps<R>
    + ModelClient<R>
    + boostr::RoPEPackedOps<R>
    + boostr::quant::DequantOps<R>
    + boostr::quant::QuantMatmulOps<R>
    + boostr::inference::speculative::SpeculativeClient<R>
{
}

impl<R, C> EngineClient<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::RoPEPackedOps<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>
        + boostr::inference::speculative::SpeculativeClient<R>,
{
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
    /// Tokenizer — splintr's universal loaded-tokenizer handle, whatever its
    /// source. Shared with the LoRA variants of this model.
    pub(crate) tokenizer: Arc<AnyTokenizer>,
    /// Device
    pub(crate) device: R::Device,
    /// Initial context size for KV cache (like Ollama's num_ctx)
    pub(crate) num_ctx: usize,
    /// Chat template for this model
    pub(crate) chat_template: ChatTemplate,
    /// Shared block allocator for paged attention (shared across all concurrent requests)
    pub(crate) shared_allocator: Option<Arc<std::sync::Mutex<CpuBlockAllocator>>>,
    /// The paged KV cache over every block of `shared_allocator`, created on first use.
    pub(crate) paged_pool: std::sync::OnceLock<LayeredPagedKvCache<R>>,
    /// Shared prefix cache for paged attention (persists across requests)
    pub(crate) prefix_cache: Option<std::sync::Mutex<PrefixCache<CpuBlockAllocator>>>,
    /// Vision tower for image inputs, loaded from `inference.mmproj`.
    pub(crate) vision_tower: Option<Arc<VisionTower<R>>>,
    /// The draft model of `drafter: model`, shared by every request.
    pub(crate) draft: Option<Arc<Executor<R>>>,
    /// The bytes of every model token id, built on the first constrained request.
    pub(crate) grammar_vocab: std::sync::OnceLock<boostr::inference::grammar::VocabBytes>,
    /// Tensor parallelism state (CUDA only; present when tp_size > 1)
    #[cfg(feature = "cuda")]
    pub(crate) tp_state: Option<crate::engine::tensor_parallel::TensorParallelState>,
    /// GPU-accelerated prefix cache (CUDA only; opt-in via config)
    #[cfg(feature = "cuda")]
    pub(crate) gpu_prefix_cache:
        Option<std::sync::Mutex<boostr::inference::prefix_cache::GpuPrefixCache>>,
    /// Captured Llama decode graph and its caches, reused across requests.
    /// Held for a whole graph-mode request, so such requests run one at a time.
    #[cfg(feature = "cuda")]
    pub(crate) llama_graph_session:
        tokio::sync::Mutex<Option<crate::engine::graphs::graph_session::LlamaGraphSession>>,
    /// Captured `qwen35` decode graph and its caches, reused across requests.
    #[cfg(feature = "cuda")]
    pub(crate) qwen35_graph_session:
        tokio::sync::Mutex<Option<crate::engine::graphs::graph_session::Qwen35GraphSession>>,
    /// Multi-lane `qwen35` decode session: batched caches, one lane table and
    /// the decode driver. The lock is taken for one admit, step or release
    /// and dropped, never held across a request's token stream.
    #[cfg(feature = "cuda")]
    pub(crate) qwen35_lane_session:
        tokio::sync::Mutex<Option<crate::engine::lane_session::Qwen35LaneSession>>,
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
        Self::build(model, config, Arc::new(tokenizer), device, num_ctx)
    }

    /// [`Self::new`] over a tokenizer another executor shares, such as the
    /// base executor of a LoRA variant.
    pub(crate) fn build(
        model: LoadedModel<R>,
        config: BlazrConfig,
        tokenizer: Arc<AnyTokenizer>,
        device: R::Device,
        num_ctx: usize,
    ) -> Result<Self> {
        let config = with_runtime_dtype::<R>(config)?;
        crate::engine::speculative::report_speculative(&model, &config.inference)?;
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
            paged_pool: std::sync::OnceLock::new(),
            prefix_cache,
            vision_tower: None,
            draft: None,
            grammar_vocab: std::sync::OnceLock::new(),
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
        tp_state: crate::engine::tensor_parallel::TensorParallelState,
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

    /// The ids that end generation, inside the model vocabulary.
    ///
    /// They come from the tokenizer, never from a hard-coded list.
    /// [`Self::decode_finish_reason`] ends on the same ids. The model can
    /// never emit an id past its vocabulary.
    pub(in crate::engine) fn eos_ids(&self) -> Vec<u32> {
        self.tokenizer()
            .eos_token_id()
            .filter(|&id| usize::try_from(id).is_ok_and(|at| at < self.vocab_size()))
            .into_iter()
            .collect()
    }

    /// Get the device
    pub(crate) fn device(&self) -> &R::Device {
        &self.device
    }

    /// Get the loaded model as an Arc (for sharing with batch engines)
    pub(crate) fn model_arc(&self) -> &Arc<LoadedModel<R>> {
        &self.model
    }

    /// The loaded model: the base model, or for a LoRA variant the model
    /// with its adapter attached.
    pub fn model(&self) -> &LoadedModel<R> {
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
}

#[cfg(test)]
mod tests {
    use boostr::model::{LoadedModel, UniversalConfig};
    use boostr::nn::{VarBuilder, VarMap};
    use boostr::{CpuDevice, CpuRuntime, DType, Tensor};

    use crate::config::BlazrConfig;
    use crate::engine::Executor;

    const MAMBA2_YAML: &str = "model_type: mamba2\nvocab_size: 16\nhidden_size: 8\nnum_layers: 1\n\
        max_seq_len: 16\ntie_word_embeddings: true\nssm:\n  variant: mamba2\n  num_heads: 2\n  \
        head_dim: 8\n  state_size: 4\n  chunk_size: 4\n  n_groups: 1\n  conv_kernel: 4\n  expand: 2\n";

    /// A one-layer Mamba2 with all-zero weights.
    fn tiny_mamba2(paged: bool, prefix_cache: bool) -> anyhow::Result<Executor<CpuRuntime>> {
        let device = CpuDevice::new();
        let config: UniversalConfig = serde_saphyr::from_str(MAMBA2_YAML)?;
        let mut map = VarMap::<CpuRuntime>::new();
        for (name, shape) in [
            ("model.embed_tokens.weight", vec![16, 8]),
            ("model.layers.0.input_layernorm.weight", vec![8]),
            ("model.layers.0.mixer.in_proj.weight", vec![42, 8]),
            ("model.layers.0.mixer.conv1d.weight", vec![24, 1, 4]),
            ("model.layers.0.mixer.out_proj.weight", vec![8, 16]),
            ("model.layers.0.mixer.A_log", vec![2]),
            ("model.layers.0.mixer.dt_bias", vec![2]),
            ("model.layers.0.mixer.D", vec![2]),
            ("model.norm.weight", vec![8]),
        ] {
            map.insert(
                name.to_string(),
                Tensor::<CpuRuntime>::zeros(&shape, DType::F32, &device)?,
            );
        }
        let mut vb = VarBuilder::new(&mut map, &device);
        let model = LoadedModel::<CpuRuntime>::load(&config, &mut vb)?;
        let mut blazr_config = BlazrConfig::from_universal_with_dtype(config, "f32");
        blazr_config.inference.paged_attention = paged;
        blazr_config.inference.prefix_cache = prefix_cache;
        let tokenizer = crate::tokenizer::from_pretrained("llama3")?;
        Executor::new(model, blazr_config, tokenizer, device, 16)
    }

    #[test]
    fn paged_attention_is_an_error_for_an_ssm_model() {
        let err = tiny_mamba2(true, false)
            .err()
            .expect("paged must be refused");
        let text = err.to_string();
        assert!(text.contains("'mamba2'"), "{text}");
        assert!(text.contains("turn paged attention off"), "{text}");
        let err = tiny_mamba2(true, true)
            .err()
            .expect("prefix cache must be refused");
        assert!(err.to_string().contains("'mamba2'"), "{err}");
    }

    #[test]
    fn an_ssm_model_builds_with_paged_attention_off() {
        assert!(tiny_mamba2(false, false).is_ok());
    }
}
