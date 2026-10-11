use clap::{Parser, Subcommand};

use super::args::{RuntimeArgs, SamplingArgs, VisionArgs};

/// The optional features this binary is built with, as `--version` lists
/// them after the version: `0.1.1 (audio, cuda)`.
const FEATURES: [(&str, bool); 4] = [
    ("audio", cfg!(feature = "audio")),
    ("cuda", cfg!(feature = "cuda")),
    ("nccl", cfg!(feature = "nccl")),
    ("distributed", cfg!(feature = "distributed")),
];

/// The `--version` text: the crate version, then the built features.
static VERSION: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let built: Vec<&str> = FEATURES
        .iter()
        .filter(|(_, on)| *on)
        .map(|(name, _)| *name)
        .collect();
    if built.is_empty() {
        env!("CARGO_PKG_VERSION").to_string()
    } else {
        format!("{} ({})", env!("CARGO_PKG_VERSION"), built.join(", "))
    }
});

/// Blazr - Blazing-fast inference server for LLMs
#[derive(Parser)]
#[command(name = "blazr")]
#[command(version = VERSION.as_str(), about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run interactive generation with a model
    Run {
        /// Model name or path [default: default_model]
        model: Option<String>,

        /// Optional initial prompt
        #[arg(long, short)]
        prompt: Option<String>,

        /// Chat template: a built-in name (chatml, llama3, mistral, phi3,
        /// gemma, deepseek, generic) or a Jinja template file. Default: the
        /// model's own template, then the built-in for its model type.
        #[arg(long, value_name = "NAME|FILE")]
        chat_template: Option<String>,

        /// Show debug info (token IDs, timing breakdown)
        #[arg(long, short)]
        verbose: bool,

        #[command(flatten)]
        sampling: SamplingArgs,

        #[command(flatten)]
        runtime: RuntimeArgs,
    },

    /// Start inference server
    Serve {
        /// Model to load at start. Others load on demand. [default: default_model]
        #[arg(long, short)]
        model: Option<String>,

        /// Chat template for every served model: a built-in name (chatml,
        /// llama3, mistral, phi3, gemma, deepseek, generic) or a Jinja
        /// template file. Default: each model's own template, then the
        /// built-in for its model type.
        #[arg(long, value_name = "NAME|FILE")]
        chat_template: Option<String>,

        /// Port to listen on [default: server.port, else 8080]
        #[arg(long)]
        port: Option<u16>,

        /// Host to bind to [default: server.host, else 0.0.0.0]
        #[arg(long)]
        host: Option<String>,

        /// Copies of the start model, one per device: CUDA devices from the
        /// `device` index up, or CPU copies on a CPU build. Requests for it
        /// go to the copy with a warm prefix cache, else the least loaded.
        /// Above 1, needs a start model and tensor parallelism at 1.
        /// [default: inference.data_parallel, else 1]
        #[arg(long, value_name = "N")]
        data_parallel: Option<usize>,

        /// API key for authentication [default: $BLAZR_API_KEY]
        #[arg(long)]
        api_key: Option<String>,

        /// File containing API keys (one per line)
        #[arg(long)]
        api_key_file: Option<std::path::PathBuf>,

        /// TLS certificate file (PEM format). With a key, enables HTTPS.
        /// [default: server.tls_cert]
        #[arg(long)]
        tls_cert: Option<std::path::PathBuf>,

        /// TLS private key file (PEM format) [default: server.tls_key]
        #[arg(long)]
        tls_key: Option<std::path::PathBuf>,

        /// Register a standalone vision embedder (SigLIP/CLIP) for `/v1/embeddings`
        /// image inputs. Format: `name=/path/to/model_dir` where the directory
        /// contains `config.json` and `model.safetensors`. May be repeated.
        #[arg(long = "vision-model", value_name = "NAME=PATH")]
        vision_models: Vec<String>,

        /// Register a Whisper ASR model for `/v1/audio/transcriptions`.
        /// Format: `name=/path/to/whisper_dir` where the directory contains
        /// `config.json`, `tokenizer.json`, and `model.safetensors`. May be
        /// repeated.
        #[cfg(feature = "audio")]
        #[arg(long = "asr-model", value_name = "NAME=PATH")]
        asr_models: Vec<String>,

        /// Register a TTS model for `/v1/audio/speech`. Format:
        /// `name=/path/to/tts_dir`. The directory's `config.json` picks the
        /// engine: a VoxCPM2 config (with `dit_config`) loads the voice-cloning
        /// engine, whose voices are the recordings under `voices/` — that
        /// directory is optional, and a bundle without it still renders
        /// zero-shot; a Kokoro config loads Kokoro. A VoxCPM2 directory's
        /// optional `adapters/<one>.safetensors` folds a LoRA adapter into that
        /// bundle's weights at load time — one adapted model per bundle, no
        /// per-request switching. Serving several adapters means repeating
        /// this flag with several `NAME=DIR` bundles, each with its own
        /// `adapters/` directory. `voice` on a synthesis request omitted means
        /// zero-shot. May be repeated.
        #[cfg(feature = "audio")]
        #[arg(long = "tts-model", value_name = "NAME=PATH")]
        tts_models: Vec<String>,

        /// Requests one VoxCPM2 batch renders together (rounded down to 1,
        /// 2, 4, or 8). Kokoro is unaffected. Every slot steps on every
        /// iteration, a finished one included, so a wider batch raises
        /// throughput and raises each request's own latency.
        #[cfg(feature = "audio")]
        #[arg(long = "tts-max-batch", value_name = "N", default_value_t = 8)]
        tts_max_batch: usize,

        /// Milliseconds a VoxCPM2 batch waits after its first request
        /// before starting, so late arrivals still join it instead of
        /// opening the next batch. Kokoro is unaffected.
        #[cfg(feature = "audio")]
        #[arg(long = "tts-batch-window-ms", value_name = "MS", default_value_t = 20)]
        tts_batch_window_ms: u64,

        /// Stop keeping a VoxCPM2 engine's batch open across steps.
        ///
        /// By default the batch is persistent, so a request that arrives
        /// mid-generation joins a free slot on the next step instead of
        /// waiting for the running batch to drain. It costs an idle
        /// deployment nothing: the batch steps only the slots actually
        /// seated, and its row allocation opens at one and grows only as far
        /// as concurrency reaches. What a seated row does hold is
        /// --tts-continuous-capacity cache positions for the engine's life.
        /// Pass this where that is worth more than mid-flight admission.
        /// Kokoro is unaffected.
        #[cfg(feature = "audio")]
        #[arg(long = "tts-no-continuous-batching")]
        tts_no_continuous_batching: bool,

        /// Cache positions held per seated batch slot, unless
        /// --tts-no-continuous-batching is passed. Not a per-request limit:
        /// held for two caches (K and V) per layer, for the engine's whole
        /// life, once per row the batch has grown to. Kokoro is unaffected.
        #[cfg(feature = "audio")]
        #[arg(
            long = "tts-continuous-capacity",
            value_name = "N",
            default_value_t = 1024
        )]
        tts_continuous_capacity: usize,

        /// Directory of Kokoro voice files (`.safetensors`, `.pt`, or
        /// `.pth`), searched before the model directory's `voices/`.
        /// [default: $BLAZR_VOICE_DIR]. A bare `voice` id in
        /// `/v1/audio/speech` (e.g. `af_alloy`) is looked up here; a path
        /// is read as given.
        #[arg(long = "voice-dir", value_name = "DIR")]
        voice_dir: Option<std::path::PathBuf>,

        /// Default reasoning split for chat responses: `deepseek` returns the
        /// think block as `reasoning_content` and the answer as `content`,
        /// `none` returns raw output. A request's `reasoning_format` overrides.
        /// [default: server.reasoning_format, else deepseek]
        #[arg(long, value_enum)]
        reasoning_format: Option<crate::config::ReasoningFormat>,

        #[command(flatten)]
        vision: VisionArgs,
    },

    /// List available models
    List {
        /// Show detailed information
        #[arg(long, short)]
        verbose: bool,
    },

    /// Show model information
    Info {
        /// Model name or path
        model: String,
    },

    /// Pull model from HuggingFace Hub
    Pull {
        /// Repository ID (e.g., "TheBloke/Mistral-7B-v0.1-GGUF")
        repo: String,

        /// Specific file to download (e.g., "mistral-7b-v0.1.Q4_K_M.gguf")
        #[arg(long)]
        file: Option<String>,

        /// Output directory [default: $BLAZR_MODEL_DIR, then model_dir, then ./models]
        #[arg(long, short)]
        output: Option<std::path::PathBuf>,
    },

    /// Generate text (non-interactive)
    Generate {
        /// Model name or path [default: default_model]
        #[arg(long, short)]
        model: Option<String>,

        /// Prompt text
        #[arg(long, short)]
        prompt: String,

        #[command(flatten)]
        sampling: SamplingArgs,

        #[command(flatten)]
        runtime: RuntimeArgs,

        /// Print prompt tokens before generation (like llama-cli --verbose-prompt)
        #[arg(long)]
        verbose_prompt: bool,

        /// Image file to place before the prompt; needs --mmproj. May be repeated.
        #[arg(long = "image", value_name = "PATH")]
        images: Vec<std::path::PathBuf>,
    },

    /// Interactive multi-turn chat with a model
    Chat {
        /// Model name or path [default: default_model]
        model: Option<String>,

        /// System prompt
        #[arg(long, short)]
        system: Option<String>,

        /// Chat template: a built-in name (chatml, llama3, mistral, phi3,
        /// gemma, deepseek, generic) or a Jinja template file. Default: the
        /// model's own template, then the built-in for its model type.
        #[arg(long, value_name = "NAME|FILE")]
        chat_template: Option<String>,

        /// Maximum tokens per turn [default: generation.max_tokens, else 2048]
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Sampling temperature, 0 = greedy [default: generation.temperature, else 1.0]
        #[arg(long)]
        temperature: Option<f32>,

        /// Top-p nucleus sampling [default: generation.top_p, else 1.0]
        #[arg(long)]
        top_p: Option<f32>,

        /// Initial KV cache size [default: num_ctx, else 2048]
        #[arg(long)]
        num_ctx: Option<usize>,

        /// Show debug info (token IDs, timing breakdown)
        #[arg(long, short)]
        verbose: bool,

        #[command(flatten)]
        vision: VisionArgs,
    },

    /// Benchmark model performance (tok/s, TTFT, ITL)
    Bench {
        /// Model name or path [default: default_model]
        model: Option<String>,

        /// Initial KV cache size [default: num_ctx, else 2048]
        #[arg(long)]
        num_ctx: Option<usize>,

        /// Number of tokens to generate per run
        #[arg(long)]
        decode_tokens: Option<usize>,

        /// Number of measurement runs
        #[arg(long)]
        runs: Option<usize>,

        /// Export results to JSON file
        #[arg(long)]
        json: Option<std::path::PathBuf>,

        /// Concurrency levels to sweep (e.g., --concurrency 1 2 4 8)
        #[arg(long, num_args = 1..)]
        concurrency: Option<Vec<usize>>,
    },

    /// Show running models on a blazr server
    Ps {
        /// Server URL (default: http://localhost:8080)
        #[arg(long, default_value = "http://localhost:8080")]
        server: String,
    },

    /// Convert model format (delegates to compressr)
    Convert {
        /// Input model path
        input: String,

        /// Output path (default: auto-generated)
        #[arg(long, short)]
        output: Option<String>,

        /// Target format (gguf, awq, gptq, fp8, safetensors)
        #[arg(long, short, default_value = "gguf")]
        format: String,

        /// Quantization method (e.g., q4_k_m, q8_0, q5_k_s)
        #[arg(long, short)]
        quantization: Option<String>,

        /// Debug-level compressr logs (a `RUST_LOG` already set wins)
        #[arg(long, short)]
        verbose: bool,
    },

    /// Join or start a distributed inference swarm
    #[cfg(feature = "distributed")]
    Swarm {
        /// Role: "leader" or "worker"
        #[arg(long, default_value = "worker")]
        role: String,

        /// Shared secret token for swarm membership
        #[arg(long, env = "BLAZR_SWARM_TOKEN")]
        token: String,

        /// Model to serve (leader only) [default: default_model]
        #[arg(long, short)]
        model: Option<String>,

        /// Leader address to connect to (worker only, e.g., "192.168.1.10:9090")
        #[arg(long)]
        leader: Option<String>,

        /// Port for nexar transport
        #[arg(long, default_value = "9090")]
        swarm_port: u16,

        /// Port for HTTP API (leader only) [default: server.port, else 8080]
        #[arg(long)]
        port: Option<u16>,

        /// Host for HTTP API (leader only) [default: server.host, else 0.0.0.0]
        #[arg(long)]
        host: Option<String>,

        /// API key for the leader's HTTP API [default: $BLAZR_API_KEY]
        #[arg(long)]
        api_key: Option<String>,

        /// Enable mDNS auto-discovery on LAN
        #[arg(long)]
        mdns: bool,
    },

    /// Run a disaggregated prefill/decode router or worker
    #[cfg(feature = "distributed")]
    Disagg {
        /// Role: "router", "prefill", or "decode"
        #[arg(long, default_value = "router")]
        role: String,

        /// Model to load (prefill and decode roles) [default: default_model]
        #[arg(long, short)]
        model: Option<String>,

        /// Address for this node to listen on
        #[arg(long, default_value = "0.0.0.0:9190")]
        listen_addr: String,

        /// Router address to connect to (required for prefill and decode roles)
        #[arg(long)]
        router_addr: Option<String>,

        /// Prefill worker addresses (router role only, e.g., "192.168.1.2:9191")
        #[arg(long, num_args = 1..)]
        prefill: Option<Vec<String>>,

        /// Decode worker addresses (router role only, e.g., "192.168.1.3:9192")
        #[arg(long, num_args = 1..)]
        decode: Option<Vec<String>>,

        /// HTTP API port (router role only) [default: server.port, else 8080]
        #[arg(long)]
        port: Option<u16>,

        /// HTTP API host (router role only) [default: server.host, else 0.0.0.0]
        #[arg(long)]
        host: Option<String>,
    },

    /// Narration from a script: the voiceover WAV, its script, its phrase
    /// spans, and every phrase heard back with Whisper. Exits 1 when a
    /// problem remains or the run fails, 2 on a bad input, 3 on a model
    /// that blazr does not run.
    #[cfg(feature = "audio")]
    Speak(super::narration::SpeakArgs),

    /// Word timings for narration from a local Whisper model: one words
    /// file per narration, and the video timeline from a manifest. Exits 1
    /// when the run fails, 2 on a bad input.
    #[cfg(feature = "audio")]
    Transcribe(super::narration::TranscribeArgs),

    /// The motion-video skill's developer sound review: sharpness, hiss,
    /// tail and true peak of every rendered sound, and a CLAP ranking of the
    /// base variants against their `says` texts (`--no-model` skips it).
    /// Exits 1 when a sound fails a limit or the run fails, 2 with no
    /// directory.
    #[cfg(feature = "audio")]
    SoundReview(super::narration::SoundReviewArgs),

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_names_each_built_feature() {
        assert!(VERSION.starts_with(env!("CARGO_PKG_VERSION")));
        assert_eq!(VERSION.contains("(audio"), cfg!(feature = "audio"));
        assert_eq!(VERSION.contains("cuda"), cfg!(feature = "cuda"));
        assert_eq!(VERSION.contains('('), FEATURES.iter().any(|(_, on)| *on));
    }
}
