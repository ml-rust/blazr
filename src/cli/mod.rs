//! CLI commands
//!
//! Provides ollama-like CLI interface for blazr.

mod info;
mod list;
mod pull;
mod run;
mod serve;

pub use info::info;
pub use list::list;
pub use pull::pull;
pub use run::run;
pub use serve::serve;

use clap::{Parser, Subcommand};

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

        /// Maximum tokens to generate
        #[arg(long, default_value = "2048")]
        max_tokens: usize,

        /// Sampling temperature (0 = greedy)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Top-p nucleus sampling
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Number of layers to load on GPU (default: auto-detect based on VRAM)
        /// Use -1 for all layers, 0 for auto-detect
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        gpu_layers: i32,

        /// Use CPU instead of GPU
        #[arg(long)]
        cpu: bool,

        /// Context window size (like Ollama's num_ctx). Default: 2048.
        /// This controls the initial KV cache allocation. The cache will grow
        /// dynamically if needed (up to the model's max_position_embeddings).
        #[arg(long, default_value = "2048")]
        num_ctx: usize,
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

        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Sampling temperature (0 = greedy)
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Number of layers to load on GPU (default: auto-detect based on VRAM)
        /// Use -1 for all layers, 0 for auto-detect
        #[arg(long, default_value = "0", allow_hyphen_values = true)]
        gpu_layers: i32,

        /// Use CPU instead of GPU
        #[arg(long)]
        cpu: bool,

        /// Context window size (like Ollama's num_ctx). Default: 2048.
        /// This controls the initial KV cache allocation. The cache will grow
        /// dynamically if needed (up to the model's max_position_embeddings).
        #[arg(long, default_value = "2048")]
        num_ctx: usize,
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
