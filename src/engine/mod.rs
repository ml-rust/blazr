//! Core inference engine
//!
//! This module provides the inference execution pipeline:
//! - Executor: Runs inference on a loaded model
//! - Scheduler: Manages model lifecycle (load/unload/eviction)
//! - Sampling: Token sampling strategies (temperature, mirostat, DRY, typical)
//! - Types: Shared result types (FinishReason, GeneratedToken, etc.)

pub mod bench_config;
pub mod decision;
mod decoding;
mod executors;
mod graphs;
pub(crate) mod host_read;
mod lanes;
mod lifecycle;
pub(crate) mod load;
pub mod lora;
mod parallel;
mod requests;
mod speculative;
#[cfg(test)]
pub(crate) mod test_support;

pub use decoding::*;
pub use executors::*;
#[cfg(feature = "cuda")]
pub use graphs::graph_session::{LlamaGraphSession, Qwen35GraphSession};
pub use lanes::*;
pub use lifecycle::*;
pub use lora::{AdapterSpec, LoraError, VariantInfo};
pub use parallel::*;
pub use requests::*;
