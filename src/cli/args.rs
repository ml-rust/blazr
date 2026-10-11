//! Argument groups shared by several commands.

use clap::Parser;

/// Shared sampling parameters for text generation. An unset flag takes the
/// `generation` section of `~/.blazr/config.yaml`, then the built-in default.
#[derive(Parser, Debug, Clone)]
pub struct SamplingArgs {
    /// Maximum tokens to generate [default: generation.max_tokens, else 2048]
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Sampling temperature, 0 = greedy [default: generation.temperature, else 1.0]
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Top-p nucleus sampling [default: generation.top_p, else 1.0]
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Top-k sampling, 0 = off [default: generation.top_k, else 0]
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Min-p threshold: drops tokens with prob < min_p * max_prob
    /// [default: generation.min_p, else 0.05]
    #[arg(long)]
    pub min_p: Option<f32>,

    /// Repetition penalty, 1.0 = off [default: generation.repeat_penalty, else 1.1]
    #[arg(long)]
    pub repeat_penalty: Option<f32>,

    /// Recent tokens the repetition penalty covers
    /// [default: generation.repeat_last_n, else 64]
    #[arg(long)]
    pub repeat_last_n: Option<usize>,
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

    /// Use CPU instead of GPU [default: the `device` key, else GPU 0 of the build]
    #[arg(long)]
    pub cpu: bool,

    /// Initial KV cache size, like Ollama's num_ctx. The cache grows as
    /// needed, up to the model's maximum. [default: num_ctx, else 2048]
    #[arg(long)]
    pub num_ctx: Option<usize>,

    /// Use paged attention (vLLM-style block-based KV cache)
    /// [default: inference.paged_attention]
    #[arg(long)]
    pub paged_attention: bool,

    /// Use compute graph capture for greedy decode (requires --temperature 0)
    /// [default: inference.graphs]
    #[arg(long)]
    pub graphs: bool,

    #[command(flatten)]
    pub vision: VisionArgs,

    #[command(flatten)]
    pub dtype: DtypeArgs,
}

/// The compute dtype flag of `run`, `generate`, `chat`, `bench` and `serve`.
#[derive(Parser, Debug, Clone, Default)]
pub struct DtypeArgs {
    /// Compute dtype of the weights, activations and KV cache. wgpu runs
    /// f32 whatever this says. GGUF runs f32, AWQ and GPTQ run f16.
    /// [default: inference.dtype, else the model torch_dtype, else f32]
    #[arg(long, value_name = "f32|bf16|f16", value_parser = parse_dtype_flag)]
    pub dtype: Option<boostr::DType>,
}

/// Parse `--dtype`. The error names the flag and the accepted values.
fn parse_dtype_flag(value: &str) -> Result<boostr::DType, String> {
    crate::config::parse_dtype_for(value, "--dtype").map_err(|e| e.to_string())
}

impl SamplingArgs {
    /// The given flags. `verbose_prompt` is the `-v` or `--verbose-prompt` switch.
    pub fn into_flags(self, verbose_prompt: bool) -> crate::config::GenerationFlags {
        crate::config::GenerationFlags {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            verbose_prompt,
        }
    }
}
