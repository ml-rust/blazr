//! Request scheduling: continuous batching, slots, queue depth, SLO policy.

pub mod batch_decode;
mod batch_emit;
pub mod batch_engine;
mod batch_engine_open;
mod batch_prefill;
mod batch_run;
#[cfg(test)]
pub(crate) mod batch_test_rig;
mod cancel_marks;
mod engine_thread;
pub(in crate::engine) mod kv_recompute;
pub mod latency_window;
pub(super) mod queue_depth;
mod request_cancel;
pub mod request_registry;
pub mod request_scheduler;
pub(in crate::engine) mod seq_sampler;
pub mod slo_policy;
pub mod slots;
mod work_hold;

pub use batch_engine::BatchEngine;
pub use engine_thread::{EngineThread, FailRequests, ThreadedScheduler};
pub use queue_depth::{QueueDepth, Waiting};
pub use request_registry::{CancelRequest, GenerationScheduler, RequestHandle, RequestRegistry};
pub use request_scheduler::RequestScheduler;
pub use {slo_policy::SloBatchPolicy, slots::SlotManager, work_hold::WorkHold};
