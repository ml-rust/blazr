//! The `Executor`: generate, embed, multimodal, cache, warmup, and load paths.

pub(crate) mod blocking_stream;
pub(crate) mod decode_stream;
pub(crate) mod executor;
pub(crate) mod executor_access;
pub(super) mod executor_cache;
pub(super) mod executor_embed;
pub mod executor_load;
pub(super) mod executor_multimodal;
pub(super) mod executor_pools;
pub(super) mod executor_qwen35_vision;
pub(super) mod finish_reason;
pub(super) mod generation;
pub(super) mod paged_pool;
pub mod pooling;
pub(super) mod sampler;
pub(super) mod types;
pub(super) mod unsupported;
pub mod vision_tower;
pub(super) mod warmup;

pub use executor::Executor;
pub use executor_access::NoWeights;
pub use executor_load::{load_executor, load_executor_from_bytes, LoadOptions, ModelBytes};
#[cfg(all(feature = "distributed", feature = "server"))]
pub(crate) use sampler::row_sampler::RowSampler;
pub use types::{FinishReason, GeneratedToken, GenerationResult, PrefillCancel, TokenLogprob};
pub(crate) use unsupported::qwen35_unsupported;
pub use vision_tower::{ImageTokenLimits, VisionTower};
