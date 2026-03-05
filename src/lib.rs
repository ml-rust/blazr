//! Blazr - Blazing-fast inference server for LLMs
//!
//! Blazr is a thin wrapper around boostr's model infrastructure,
//! providing CLI and HTTP interfaces for model inference.
//!
//! # Architecture
//!
//! Blazr follows the thin-layer design principle:
//! - **boostr**: All tensor ops, model architectures, quantization, kernels
//! - **blazr**: CLI, HTTP server, model management, streaming
//!
//! # Supported Formats
//!
//! - SafeTensors (HuggingFace standard)
//! - GGUF (llama.cpp format, quantized models)
//!
//! # Example
//!
//! ```bash
//! # Interactive generation
//! blazr run mistral-7b --prompt "Hello"
//!
//! # Start server
//! blazr serve --model mistral-7b --port 8080
//!
//! # List available models
//! blazr list
//! ```

pub mod cli;
pub mod config;
pub mod engine;
pub mod loader;
pub mod model;
pub mod server;
pub mod tokenizer;

// Re-export key types
pub use config::{BlazrConfig, GenerationConfig, ServerConfig};
pub use engine::{Executor, Scheduler};
pub use loader::{load_model, ModelFormat, ModelSource};
