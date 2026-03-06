use clap::{Parser, Subcommand};

/// Blazr - Blazing-fast inference server for LLMs
#[derive(Parser)]
#[command(name = "blazr")]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

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

#[derive(Subcommand)]
pub enum Commands {
    /// Run interactive generation with a model
    Run {
        /// Model name or path
        model: String,

        /// Optional initial prompt
        #[arg(long, short)]
        prompt: Option<String>,

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
    },

    /// Interactive multi-turn chat with a model
    Chat {
        /// Model name or path
        model: String,

        /// System prompt
        #[arg(long, short)]
        system: Option<String>,

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
