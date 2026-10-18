//! Inference configuration settings

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

/// Inference-specific configuration.
///
/// A model's `config.json` sets these per model. The `inference` section of
/// `~/.blazr/config.yaml` replaces each key it sets, for every model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Maximum context length. Replaces the model maximum sequence length.
    #[serde(default)]
    pub max_context_len: Option<usize>,

    /// The compute dtype (f32, bf16, f16) of the loaded model. The loader
    /// resolves it before any weight loads, see `config::dtype`. Weights,
    /// activations and the KV cache all take it. A model `config.json`
    /// value here is ignored. The user file key `inference.dtype` is a
    /// request the loader reads, never a write over this field.
    #[serde(skip, default = "default_dtype")]
    pub dtype: String,

    /// Maximum batch size for batched inference
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Enable paged attention (vLLM-style block-based KV cache)
    #[serde(default)]
    pub paged_attention: bool,

    /// Block size for paged attention (tokens per block)
    #[serde(default = "default_block_size")]
    pub block_size: usize,

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
    /// Also drives the lane path's chunked prefill, letting a long prompt
    /// interleave with lanes that are already decoding. Chunking changes the
    /// shapes the prefill runs at, so a non-zero value makes the prefill
    /// arithmetic differ in the last bits from a one-shot prefill of the
    /// same prompt. Neither path is bit-identical to the other.
    #[serde(default)]
    pub prefill_chunk_size: usize,

    /// Number of KV cache blocks in the shared pool.
    /// Used by paged attention for all concurrent requests.
    /// If 0, auto-computed from max_context_len and max_batch_size.
    #[serde(default)]
    pub kv_pool_blocks: usize,

    /// Speculative decoding config. Blazr parses it and ignores it: requests
    /// decode through the sampler. [`InferenceConfig::warn_speculative_ignored`]
    /// logs the ignored key once.
    #[serde(default)]
    pub speculative: Option<SpeculativeDecodingConfig>,

    /// Tensor parallelism degree. 1 runs on one GPU.
    /// Above 1, `blazr serve` shards a SafeTensors model across GPUs with
    /// NCCL all-reduce. `BLAZR_TP_SIZE` overrides it. Needs CUDA and NCCL.
    #[serde(default = "default_one")]
    pub tensor_parallel_size: usize,

    /// Data-parallel replicas of the start model for `blazr serve`.
    ///
    /// Above 1, `blazr serve` loads one copy per device: CUDA devices from
    /// the `device` index up on a CUDA build, CPU copies on a CPU build.
    /// `--data-parallel` overrides it. Only the user config file sets it. A
    /// model's `config.json` value is ignored.
    #[serde(default = "default_one")]
    pub data_parallel: usize,

    /// CUDA graph capture for greedy decode in `blazr run` and `generate`.
    ///
    /// After prefill, the single-token decode step is captured once and
    /// replayed per token. A run that samples or applies a penalty decodes
    /// eagerly. `--graphs` sets it from the command line.
    #[serde(default)]
    pub graphs: bool,

    /// Qwen3-VL mmproj GGUF loaded beside a `qwen35` model for image inputs.
    #[serde(default)]
    pub mmproj: Option<PathBuf>,

    /// Fewest output tokens an image resizes to. `None` keeps the mmproj value.
    #[serde(default)]
    pub image_min_tokens: Option<usize>,

    /// Most output tokens an image resizes to. `None` keeps the mmproj value.
    #[serde(default)]
    pub image_max_tokens: Option<usize>,
    /// System One decision models: detection, batching and limits.
    #[serde(default)]
    pub decision: super::decision::DecisionConfig,
    /// `blazr serve`: the start model's LoRA adapters, `<dir>[=<name>]`. `--lora` replaces it.
    #[serde(default)]
    pub lora: Vec<String>,
}

fn default_dtype() -> String {
    "f16".to_string()
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

/// Speculative decoding config. Parsed so existing config files load. Blazr does not read its fields.
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

/// Set once the ignored-`speculative` warning has been logged.
static SPECULATIVE_WARNED: AtomicBool = AtomicBool::new(false);

/// Log the ignored-`speculative` warning the first time `flag` is unset.
/// Return `true` when this call logged it.
fn warn_speculative_once(
    flag: &AtomicBool,
    speculative: Option<&SpeculativeDecodingConfig>,
) -> bool {
    if speculative.is_none() || flag.swap(true, Ordering::Relaxed) {
        return false;
    }
    tracing::warn!(
        "inference.speculative is ignored: speculative decoding is disabled until its \
         implementation is rewritten. Requests decode with the normal sampler"
    );
    true
}

impl InferenceConfig {
    /// Log once per process that `inference.speculative` is ignored.
    /// Return `true` when this call logged the warning.
    pub fn warn_speculative_ignored(&self) -> bool {
        warn_speculative_once(&SPECULATIVE_WARNED, self.speculative.as_ref())
    }
}

fn default_ram_tier_capacity() -> usize {
    5000
}

fn default_one() -> usize {
    1
}

fn default_spec_tokens() -> usize {
    5
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_context_len: None,
            dtype: default_dtype(),
            max_batch_size: default_max_batch_size(),
            paged_attention: false,
            block_size: default_block_size(),
            prefix_cache: false,
            max_cached_blocks: default_max_cached_blocks(),
            gpu_prefix_cache: false,
            gpu_prefix_cache_ram_tier: default_ram_tier_capacity(),
            prefill_chunk_size: 0,
            kv_pool_blocks: 0,
            speculative: None,
            tensor_parallel_size: default_one(),
            data_parallel: default_one(),
            graphs: false,
            mmproj: None,
            image_min_tokens: None,
            image_max_tokens: None,
            decision: Default::default(),
            lora: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> SpeculativeDecodingConfig {
        SpeculativeDecodingConfig {
            draft_model: "d".to_string(),
            num_speculative_tokens: 5,
            adaptive_depth: false,
        }
    }

    #[test]
    fn a_speculative_section_parses() -> Result<(), serde_saphyr::Error> {
        let cfg: InferenceConfig =
            serde_saphyr::from_str("speculative:\n  draft_model: tiny\n  adaptive_depth: true\n")?;
        let spec = cfg.speculative;
        assert_eq!(spec.as_ref().map(|s| s.draft_model.as_str()), Some("tiny"));
        assert_eq!(spec.as_ref().map(|s| s.num_speculative_tokens), Some(5));
        assert_eq!(spec.map(|s| s.adaptive_depth), Some(true));
        Ok(())
    }

    #[test]
    fn the_warning_logs_once() {
        let flag = AtomicBool::new(false);
        assert!(!warn_speculative_once(&flag, None), "unset logs nothing");
        assert!(warn_speculative_once(&flag, Some(&spec())));
        assert!(!warn_speculative_once(&flag, Some(&spec())));
        assert!(!warn_speculative_once(&flag, Some(&spec())));
    }
}
