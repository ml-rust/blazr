//! Argument groups shared by several commands.

use clap::Parser;

/// Shared sampling parameters for text generation
#[derive(Parser, Debug, Clone)]
pub struct SamplingArgs {
    /// Maximum tokens to generate
    #[arg(long, default_value = "2048")]
    pub max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Top-p nucleus sampling
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value = "0")]
    pub top_k: usize,

    /// Min-p sampling threshold (filters tokens with prob < min_p * max_prob)
    #[arg(long, default_value = "0.05")]
    pub min_p: f32,

    /// Repetition penalty (1.0 = no penalty)
    #[arg(long, default_value = "1.1")]
    pub repeat_penalty: f32,

    /// Number of recent tokens for repetition penalty window
    #[arg(long, default_value = "64")]
    pub repeat_last_n: usize,
}

/// Vision tower flags shared by `run`, `generate`, `chat` and `serve`.
#[derive(Parser, Debug, Clone, Default)]
pub struct VisionArgs {
    /// Qwen3-VL mmproj GGUF to load beside a qwen35 model for image inputs
    #[arg(long, value_name = "PATH")]
    pub mmproj: Option<std::path::PathBuf>,

    /// Fewest tokens an image resizes to (default: the mmproj value)
    #[arg(long, value_name = "N")]
    pub image_min_tokens: Option<usize>,

    /// Most tokens an image resizes to (default: the mmproj value)
    #[arg(long, value_name = "N")]
    pub image_max_tokens: Option<usize>,
}

impl VisionArgs {
    /// The set flags as config overrides.
    pub fn into_overrides(self) -> crate::config::VisionOverrides {
        crate::config::VisionOverrides {
            mmproj: self.mmproj,
            image_min_tokens: self.image_min_tokens,
            image_max_tokens: self.image_max_tokens,
        }
    }
}

/// Shared runtime/device parameters
#[derive(Parser, Debug, Clone)]
pub struct RuntimeArgs {
    /// Number of layers to load on GPU (default: auto-detect based on VRAM)
    /// Use -1 for all layers, 0 for auto-detect
    #[arg(long, default_value = "0", allow_hyphen_values = true)]
    pub gpu_layers: i32,

    /// Use CPU instead of GPU
    #[arg(long)]
    pub cpu: bool,

    /// Context window size (like Ollama's num_ctx). Default: 2048.
    /// This controls the initial KV cache allocation. The cache will grow
    /// dynamically if needed (up to the model's max_position_embeddings).
    #[arg(long, default_value = "2048")]
    pub num_ctx: usize,

    /// Use paged attention (vLLM-style block-based KV cache)
    #[arg(long)]
    pub paged_attention: bool,

    /// Use compute graph capture for greedy decode (requires --temperature 0)
    #[arg(long)]
    pub graphs: bool,

    #[command(flatten)]
    pub vision: VisionArgs,
}

impl SamplingArgs {
    /// Convert to a GenerationConfig
    pub fn into_gen_config(self) -> crate::config::GenerationConfig {
        crate::config::GenerationConfig {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            ..Default::default()
        }
    }
}
