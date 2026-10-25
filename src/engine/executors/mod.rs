//! The `Executor`: generate, embed, multimodal, cache, warmup, and load paths.

pub(crate) mod blocking_stream;
pub(crate) mod decode_output;
#[cfg(not(target_family = "wasm"))]
mod decode_thread;
pub(crate) mod executor;
pub(crate) mod executor_access;
pub(super) mod executor_cache;
pub(super) mod executor_embed;
pub(super) mod executor_generate;
pub(super) mod executor_generate_qwen35;
pub mod executor_load;
pub(super) mod executor_multimodal;
pub(super) mod executor_pools;
pub(super) mod executor_qwen35_vision;
pub(super) mod finish_reason;
pub(super) mod generate_text;
pub(super) mod paged_pool;
pub mod pooling;
#[cfg(any(test, all(feature = "distributed", feature = "server")))]
pub(super) mod row_sampler;
pub(super) mod sampler_grammar;
pub(super) mod sampler_state;
pub(super) mod sampler_step;
pub(super) mod types;
pub(super) mod unsupported;
pub mod vision_tower;
pub(super) mod warmup;

pub use executor::Executor;
pub use executor_access::NoWeights;
pub use executor_load::{load_executor, load_executor_from_bytes, LoadOptions, ModelBytes};
#[cfg(all(feature = "distributed", feature = "server"))]
pub(crate) use row_sampler::RowSampler;
pub use types::{FinishReason, GeneratedToken, GenerationResult, TokenLogprob};
pub(crate) use unsupported::qwen35_unsupported;
pub use vision_tower::{ImageTokenLimits, VisionTower};
