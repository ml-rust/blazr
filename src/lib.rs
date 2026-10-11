//! Blazr - Blazing-fast inference server for LLMs
//!
//! Blazr is a thin layer over boostr's model infrastructure. The library is
//! the inference engine. The CLI and the HTTP server are front ends over it.
//!
//! # Architecture
//!
//! Blazr follows the thin-layer design principle:
//! - **boostr**: All tensor ops, model architectures, quantization, kernels
//! - **blazr engine**: model loading, tokenizer, chat templates, scheduling,
//!   sampling, grammar, streaming generation, embeddings
//! - **blazr front ends**: CLI (`cli`), HTTP server (`server`)
//!
//! # Cargo features
//!
//! | Feature       | Adds                                                      |
//! | ------------- | --------------------------------------------------------- |
//! | `f16`         | Half-precision weights (default)                          |
//! | `cli`         | The [`cli`] module: parsing, REPL, progress (default)     |
//! | `server`      | The `server` module, TLS, Prometheus, `serve` (default)   |
//! | `hub`         | The Hugging Face hub client, implied by `cli` and `server` |
//! | `audio`       | TTS and ASR routes and commands, implies `server`, `cli`  |
//! | `cuda`        | CUDA runtime                                              |
//! | `wgpu`        | WebGPU runtime: Metal, Vulkan, DX12, browser              |
//! | `nccl`        | Multi-GPU through NCCL                                    |
//! | `capi`        | The `capi` module: C ABI, header `include/blazr.h`        |
//! | `distributed` | The `distributed` module (nexar)                          |
//!
//! `--no-default-features --features f16` builds the engine alone. That
//! build has no HTTP stack and no network client. It targets phone,
//! browser and embedded hosts.
//!
//! # Embedding the engine
//!
//! [`Scheduler`] is the entry point. It finds models in a directory,
//! loads them on first use, and returns a shared [`Executor`].
//! [`Executor::generate`] streams [`engine::GeneratedToken`]s for prompt ids.
//! Dropping the stream cancels generation before the next token.
//! [`Scheduler::unload`] removes a model. Its memory frees when the last
//! [`Executor`] handle drops.
//!
//! ```no_run
//! use std::pin::pin;
//! use std::sync::atomic::{AtomicBool, Ordering};
//! use std::sync::Arc;
//!
//! use blazr::boostr::{CpuDevice, CpuRuntime};
//! use blazr::tokenizer::Tokenize;
//! use blazr::{GenerationConfig, Scheduler};
//! use futures::StreamExt;
//!
//! async fn complete(cancel: Arc<AtomicBool>, on_token: impl Fn(&str)) -> anyhow::Result<()> {
//!     // Load: `llama-3.2-1b` is a directory or GGUF file under `./models`.
//!     let scheduler = Scheduler::<CpuRuntime>::new("./models".into(), CpuDevice::new());
//!     let executor = scheduler.get_executor("llama-3.2-1b").await?;
//!
//!     // Generate: each token goes to the callback as it arrives.
//!     let prompt = executor.tokenizer().encode("The capital of France is");
//!     let config = GenerationConfig {
//!         max_tokens: 64,
//!         ..Default::default()
//!     };
//!     {
//!         let mut tokens = pin!(executor.generate(&prompt, &config));
//!         while let Some(token) = tokens.next().await {
//!             // Cancel: leaving the loop drops the stream and stops decoding.
//!             if cancel.load(Ordering::Relaxed) {
//!                 break;
//!             }
//!             on_token(&token?.text);
//!         }
//!     }
//!
//!     // Unload: the weights free when `executor` drops too.
//!     drop(executor);
//!     scheduler.unload("llama-3.2-1b").await;
//!     Ok(())
//! }
//! ```
//!
//! [`Executor::get_embeddings`] returns per-token hidden states.
//! [`engine::pooling`] reduces them to one vector.
//!
//! # Command line
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

#[cfg(feature = "capi")]
pub mod capi;
#[cfg(feature = "cli")]
pub mod cli;
pub mod config;
#[cfg(feature = "distributed")]
pub mod distributed;
pub mod engine;
pub mod loader;
pub mod model;
pub mod runtime;
#[cfg(feature = "server")]
pub mod server;
pub mod telemetry;
#[cfg(feature = "server")]
pub mod tls;
pub mod tokenizer;

/// The boostr release this engine runs on, for runtime and device types.
pub use boostr;

// Re-export key types
pub use config::{BlazrConfig, FileConfig, GenerationConfig, ServerConfig};
pub use engine::{BatchEngine, Executor, RequestScheduler, Scheduler};
pub use loader::{load_model, ModelFormat, ModelSource};
pub use model::{ChatMessage, ChatTemplate};
