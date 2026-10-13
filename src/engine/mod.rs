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
pub mod decision;
pub(crate) mod executor;
mod executor_cache;
mod executor_embed;
mod executor_generate;
mod executor_generate_qwen35;
pub mod executor_load;
mod executor_multimodal;
mod executor_qwen35_vision;
mod generate_text;
pub mod grammar;
pub(crate) mod grammar_json;
pub(crate) mod grammar_parser;
mod graph_session;
mod keep_alive;
mod lane_api;
mod lane_engine;
mod lane_prefill;
mod lane_scheduler;
mod lane_session;
pub mod latency_window;
pub mod lora;
pub(crate) mod mirostat;
mod model_dir;
pub mod pooling;
mod queue_depth;
pub mod request_registry;
pub mod request_scheduler;
pub(crate) mod sampling;
mod scheduler;
pub mod slo_policy;
pub mod slots;
pub mod speculative;
pub mod tensor_parallel;
#[cfg(all(feature = "cuda", feature = "nccl"))]
pub mod tp_group;
mod types;
mod unsupported;
pub mod vision_tower;
mod warmup;

#[cfg(feature = "cuda")]
pub use boostr::inference::lane_decode::LaneDecodeMode;

pub use batch_engine::BatchEngine;
pub use cache_router::{CacheAwareRouter, RouteReason};
pub use data_parallel::{route_through, DataParallelGroup, ReplicaLease};
pub use executor::Executor;
pub use executor_load::{load_executor, load_executor_from_bytes, LoadOptions, ModelBytes};
#[cfg(feature = "cuda")]
pub use graph_session::{LlamaGraphSession, Qwen35GraphSession};
pub use keep_alive::parse_keep_alive;
#[cfg(feature = "cuda")]
pub use lane_engine::LaneEngine;
pub use lane_scheduler::LaneScheduler;
#[cfg(feature = "cuda")]
pub use lane_session::{LaneAdmitted, LaneToken, Qwen35LaneSession};
pub use lora::LoraAdapterRegistry;
pub use queue_depth::{QueueDepth, Waiting};
pub use request_registry::{GenerationScheduler, RequestHandle, RequestRegistry};
pub use request_scheduler::RequestScheduler;
pub use scheduler::{AvailableModelInfo, Scheduler};
pub use slo_policy::SloBatchPolicy;
pub use slots::SlotManager;
pub use types::{FinishReason, GeneratedToken, GenerationResult, TokenLogprob};
pub(crate) use unsupported::qwen35_unsupported;
pub use vision_tower::{ImageTokenLimits, VisionTower};
