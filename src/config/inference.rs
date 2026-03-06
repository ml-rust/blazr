//! Inference configuration settings

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Device configuration for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeviceConfig {
    /// Simple device string (e.g., "cuda:0", "cpu")
    Simple(String),
    /// Detailed device configuration
    Detailed {
        /// Device type: "cuda", "cpu"
        device_type: String,
        /// Device ID (for multi-GPU)
        #[serde(default)]
        device_id: usize,
    },
}

impl Default for DeviceConfig {
    fn default() -> Self {
        DeviceConfig::Simple("cuda:0".to_string())
    }
}

impl DeviceConfig {
    /// Get device type ("cuda" or "cpu")
    pub fn device_type(&self) -> &str {
        match self {
            DeviceConfig::Simple(s) => {
                if s.starts_with("cuda") {
                    "cuda"
                } else {
                    "cpu"
                }
            }
            DeviceConfig::Detailed { device_type, .. } => device_type,
        }
    }

    /// Get device ID (for multi-GPU)
    pub fn device_id(&self) -> usize {
        match self {
            DeviceConfig::Simple(s) => s
                .strip_prefix("cuda:")
                .and_then(|id| id.parse().ok())
                .unwrap_or(0),
            DeviceConfig::Detailed { device_id, .. } => *device_id,
        }
    }

    /// Check if using CUDA
    pub fn is_cuda(&self) -> bool {
        self.device_type() == "cuda"
    }
}

/// Inference-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Path to model weights (SafeTensors, GGUF, etc.)
    #[serde(default)]
    pub weights_path: Option<PathBuf>,

    /// Device configuration
    #[serde(default)]
    pub device: DeviceConfig,

    /// Maximum context length (overrides model default if smaller)
    #[serde(default)]
    pub max_context_len: Option<usize>,

    /// Data type for inference (f32, f16, bf16)
    #[serde(default = "default_dtype")]
    pub dtype: String,

    /// Enable flash attention if available
    #[serde(default = "default_true")]
    pub flash_attention: bool,

    /// Enable KV cache
    #[serde(default = "default_true")]
    pub kv_cache: bool,

    /// Maximum batch size for batched inference
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Enable paged attention (vLLM-style block-based KV cache)
    #[serde(default)]
    pub paged_attention: bool,

    /// Block size for paged attention (tokens per block)
    #[serde(default = "default_block_size")]
    pub block_size: usize,

    /// Number of KV cache blocks to pre-allocate for paged attention.
    /// If 0, auto-computed from max_context_len.
    #[serde(default)]
    pub num_blocks: usize,

    /// Enable prefix caching for paged attention.
    ///
    /// When enabled, reuses KV cache blocks for repeated prompt prefixes across requests.
    /// Requires `paged_attention = true`. Uses boostr's PrefixCache with FNV block hashing.
    #[serde(default)]
    pub prefix_cache: bool,

    /// Maximum number of cached prefix blocks to retain.
    /// Only used when `prefix_cache = true`.
    #[serde(default = "default_max_cached_blocks")]
    pub max_cached_blocks: usize,

    /// Use GPU-accelerated prefix cache lookup (requires CUDA).
    ///
    /// When enabled, mirrors the prefix cache hash table to GPU memory and uses a
    /// CUDA kernel for batch lookups, eliminating the CPU scheduling bottleneck.
    /// Falls back to CPU lookup when CUDA is unavailable.
    /// Includes a two-tier cache: VRAM (hot) → RAM (warm) with automatic demotion.
    #[serde(default)]
    pub gpu_prefix_cache: bool,

    /// Maximum entries in the RAM (warm) tier of the GPU prefix cache.
    /// Evicted GPU entries are demoted here before permanent eviction.
    /// Only used when `gpu_prefix_cache = true`. Default: 5000.
    #[serde(default = "default_ram_tier_capacity")]
    pub gpu_prefix_cache_ram_tier: usize,

    /// Chunk size for prefill (0 = no chunking, process entire prompt at once).
    /// When set, long prompts are split into chunks of this size, interleaving
    /// with decode steps between chunks to reduce TTFT for concurrent requests.
    #[serde(default)]
    pub prefill_chunk_size: usize,

    /// Number of KV cache blocks in the shared pool.
    /// Used by paged attention for all concurrent requests.
    /// If 0, auto-computed from max_context_len and max_batch_size.
    #[serde(default)]
    pub kv_pool_blocks: usize,

    /// Speculative decoding configuration.
    /// When set, uses a draft model to speculate tokens and verifies with the main model.
    #[serde(default)]
    pub speculative: Option<SpeculativeDecodingConfig>,

    /// Tensor parallelism degree (default: 1 = single GPU).
    /// When > 1, shards model across multiple GPUs using NCCL all-reduce.
    /// Requires CUDA and NCCL support.
    #[serde(default = "default_tensor_parallel")]
    pub tensor_parallel_size: usize,

    /// MoE expert offloading strategy.
    /// Only used for Mixture-of-Experts models (DeepSeek-V2/V3, Mixtral).
    /// Values: "auto" (default), "gpu", "cpu", "hybrid"
    #[serde(default)]
    pub moe_offload: Option<String>,

    /// Number of MoE experts to keep on GPU in hybrid offloading mode.
    /// If 0, auto-computed from available VRAM.
    #[serde(default)]
    pub moe_gpu_experts: usize,

    /// Enable graph capture for greedy decode (backend-agnostic).
    ///
    /// After prompt prefill, captures the single-token decode forward pass as a
    /// compute graph. Subsequent decode steps replay it with minimal dispatch overhead.
    /// Requires a stable KV cache pre-allocated at full capacity.
    /// Incompatible with paged_attention. Falls back gracefully if the backend
    /// does not support graph capture (`Runtime::supports_graph_capture()`).
    #[serde(default)]
    pub graphs: bool,
}

fn default_dtype() -> String {
    "f16".to_string()
}

fn default_true() -> bool {
    true
}

fn default_max_batch_size() -> usize {
    1
}

fn default_block_size() -> usize {
    16
}

fn default_max_cached_blocks() -> usize {
    10000
}

/// Speculative decoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeDecodingConfig {
    /// Path to the draft model (smaller, faster model for speculation)
    pub draft_model: String,
    /// Number of tokens to speculate per iteration (default: 5)
    #[serde(default = "default_spec_tokens")]
    pub num_speculative_tokens: usize,
    /// Enable adaptive depth (adjust speculation depth based on acceptance rate)
    #[serde(default)]
    pub adaptive_depth: bool,
}

fn default_ram_tier_capacity() -> usize {
    5000
}

fn default_tensor_parallel() -> usize {
    1
}

fn default_spec_tokens() -> usize {
    5
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            weights_path: None,
            device: DeviceConfig::default(),
            max_context_len: None,
            dtype: default_dtype(),
            flash_attention: true,
            kv_cache: true,
            max_batch_size: 1,
            paged_attention: false,
            block_size: default_block_size(),
            num_blocks: 0,
            prefix_cache: false,
            max_cached_blocks: default_max_cached_blocks(),
            gpu_prefix_cache: false,
            gpu_prefix_cache_ram_tier: default_ram_tier_capacity(),
            prefill_chunk_size: 0,
            kv_pool_blocks: 0,
            speculative: None,
            tensor_parallel_size: 1,
            moe_offload: None,
            moe_gpu_experts: 0,
            graphs: false,
        }
    }
}
