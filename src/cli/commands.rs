use clap::{Parser, Subcommand};

use super::args::{RuntimeArgs, SamplingArgs, VisionArgs};

/// Blazr - Blazing-fast inference server for LLMs
#[derive(Parser)]
#[command(name = "blazr")]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Run interactive generation with a model
    Run {
        /// Model name or path
        model: String,

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
        /// Model name or path (optional, can load on-demand)
        #[arg(long, short)]
        model: Option<String>,

        /// Chat template for every served model: a built-in name (chatml,
        /// llama3, mistral, phi3, gemma, deepseek, generic) or a Jinja
        /// template file. Default: each model's own template, then the
        /// built-in for its model type.
        #[arg(long, value_name = "NAME|FILE")]
        chat_template: Option<String>,

        /// Port to listen on
        #[arg(long, default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// API key for authentication (also reads BLAZR_API_KEY env var)
        #[arg(long, env = "BLAZR_API_KEY")]
        api_key: Option<String>,

        /// File containing API keys (one per line)
        #[arg(long)]
        api_key_file: Option<std::path::PathBuf>,

        /// TLS certificate file (PEM format) — enables HTTPS
        #[arg(long)]
        tls_cert: Option<std::path::PathBuf>,

        /// TLS private key file (PEM format)
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

        /// Keep a VoxCPM2 engine's batch open across steps and admit a
        /// request that arrives mid-generation into a free slot on the
        /// next step, instead of making it wait for the batch to drain.
        /// Lowers latency for overlapping requests at a standing cost:
        /// every reserved slot steps every iteration, occupied or not,
        /// for the engine's whole life. Off by default. Kokoro is
        /// unaffected.
        #[cfg(feature = "audio")]
        #[arg(long = "tts-continuous-batching")]
        tts_continuous_batching: bool,

        /// Cache positions reserved per batch slot when
        /// --tts-continuous-batching is on; ignored otherwise. A memory
        /// reservation, not a per-request limit: held for two caches (K
        /// and V) per layer, sized at this value times --tts-max-batch,
        /// for the engine's whole life. Kokoro is unaffected.
        #[cfg(feature = "audio")]
        #[arg(long = "tts-continuous-capacity", value_name = "N", default_value_t = 1024)]
        tts_continuous_capacity: usize,

        /// Directory containing TTS voice files (`.safetensors`, `.pt`, or
        /// `.pth`). Overrides the bundled `assets/kokoro_voices/` that ships
        /// with blazr. Also overridable via `$BLAZR_VOICE_DIR`. The `--voice`
        /// CLI flags on synthesis requests and the OpenAI-compatible `voice`
        /// field in `/v1/audio/speech` resolve against this directory. Bare
        /// voice IDs (e.g. `af_alloy`) are looked up here; absolute paths
        /// bypass the resolver.
        #[arg(long = "voice-dir", value_name = "DIR")]
        voice_dir: Option<std::path::PathBuf>,

        /// Default reasoning split for chat responses: `deepseek` returns the
        /// think block as `reasoning_content` and the answer as `content`,
        /// `none` returns raw output. A request's `reasoning_format` overrides.
        #[arg(long, value_enum, default_value = "deepseek")]
        reasoning_format: crate::config::ReasoningFormat,

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

        /// Output directory
        #[arg(long, short)]
        output: Option<std::path::PathBuf>,
    },

    /// Generate text (non-interactive)
    Generate {
        /// Model name or path
        #[arg(long, short)]
        model: String,

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
        /// Model name or path
        model: String,

        /// System prompt
        #[arg(long, short)]
        system: Option<String>,

        /// Chat template: a built-in name (chatml, llama3, mistral, phi3,
        /// gemma, deepseek, generic) or a Jinja template file. Default: the
        /// model's own template, then the built-in for its model type.
        #[arg(long, value_name = "NAME|FILE")]
        chat_template: Option<String>,

        /// Maximum tokens to generate per turn
        #[arg(long, default_value = "2048")]
        max_tokens: usize,

        /// Sampling temperature (0 = greedy)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Top-p nucleus sampling
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Context window size
        #[arg(long, default_value = "4096")]
        num_ctx: usize,

        /// Show debug info (token IDs, timing breakdown)
        #[arg(long, short)]
        verbose: bool,

        #[command(flatten)]
        vision: VisionArgs,
    },

    /// Benchmark model performance (tok/s, TTFT, ITL)
    Bench {
        /// Model name or path
        model: String,

        /// Context window size
        #[arg(long, default_value = "2048")]
        num_ctx: usize,

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

        /// Verbose output
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

        /// Model to serve (leader only)
        #[arg(long, short)]
        model: Option<String>,

        /// Leader address to connect to (worker only, e.g., "192.168.1.10:9090")
        #[arg(long)]
        leader: Option<String>,

        /// Port for nexar transport
        #[arg(long, default_value = "9090")]
        swarm_port: u16,

        /// Port for HTTP API (leader only)
        #[arg(long, default_value = "8080")]
        port: u16,

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

        /// Model to load (required for prefill and decode roles)
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

        /// HTTP API port (router role only)
        #[arg(long, default_value = "8080")]
        port: u16,
    },

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },

    /// Decode a file with a model (for testing)
    #[command(hide = true)]
    Decode {
        /// Model name or path
        #[arg(long, short)]
        model: String,

        /// Input file
        input: std::path::PathBuf,
    },
}
