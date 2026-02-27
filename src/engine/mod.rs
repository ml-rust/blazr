//! Core inference engine
//!
//! This module provides the inference execution pipeline:
//! - Executor: Runs inference on a loaded model
//! - Scheduler: Manages model lifecycle (load/unload/eviction)
//! - Batch: Request batching (future)

mod executor;
mod scheduler;

pub use executor::{Executor, GeneratedToken};
pub use scheduler::Scheduler;
