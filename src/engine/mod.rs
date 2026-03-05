//! Core inference engine
//!
//! This module provides the inference execution pipeline:
//! - Executor: Runs inference on a loaded model
//! - Scheduler: Manages model lifecycle (load/unload/eviction)
//! - Sampling: Token sampling strategies (temperature, mirostat, DRY, typical)
//! - Types: Shared result types (FinishReason, GeneratedToken, etc.)

mod cuda_graphs;
pub(crate) mod executor;
pub(crate) mod sampling;
mod scheduler;
pub mod slots;
mod types;

pub use executor::Executor;
pub use scheduler::{parse_keep_alive, Scheduler};
pub use slots::SlotManager;
pub use types::{FinishReason, GeneratedToken, GenerationResult, TokenLogprob};
