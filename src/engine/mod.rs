//! Core inference engine
//!
//! This module provides the inference execution pipeline:
//! - Executor: Runs inference on a loaded model
//! - Scheduler: Manages model lifecycle (load/unload/eviction)
//! - Sampling: Token sampling strategies (temperature, mirostat, DRY, typical)
//! - Types: Shared result types (FinishReason, GeneratedToken, etc.)

pub mod batch_decode;
pub mod batch_engine;
pub mod bench_config;
pub mod cache_router;
mod cuda_graphs;
mod cuda_graphs_batched;
mod cuda_graphs_qwen35;
pub mod data_parallel;
pub(crate) mod executor;
mod executor_cache;
mod executor_embed;
mod executor_generate;
mod executor_generate_qwen35;
mod executor_multimodal;
mod generate_text;
pub mod grammar;
pub(crate) mod grammar_json;
pub(crate) mod grammar_parser;
mod graph_session;
pub mod lora;
pub(crate) mod mirostat;
pub mod moe_offload;
pub mod moe_offload_types;
pub mod request_scheduler;
pub(crate) mod sampling;
mod scheduler;
pub mod slots;
pub mod speculative;
pub mod tensor_parallel;
mod types;
mod unsupported;
mod warmup;

pub use batch_engine::BatchEngine;
pub use data_parallel::DataParallelGroup;
pub use executor::Executor;
#[cfg(feature = "cuda")]
pub use graph_session::{LlamaGraphSession, Qwen35GraphSession};
pub use lora::{load_lora_adapter, LoraAdapter, LoraAdapterRegistry};
pub use request_scheduler::RequestScheduler;
pub use scheduler::{parse_keep_alive, AvailableModelInfo, Scheduler};
pub use slots::SlotManager;
pub use types::{FinishReason, GeneratedToken, GenerationResult, TokenLogprob};
pub(crate) use unsupported::qwen35_unsupported;
