//! The `Executor`: generate, embed, multimodal, cache, warmup, and load paths.

pub(crate) mod executor;
pub(super) mod executor_cache;
pub(super) mod executor_embed;
pub(super) mod executor_generate;
pub(super) mod executor_generate_qwen35;
pub mod executor_load;
pub(super) mod executor_multimodal;
pub(super) mod executor_qwen35_vision;
pub(super) mod generate_text;
pub mod pooling;
pub(super) mod types;
pub(super) mod unsupported;
pub mod vision_tower;
pub(super) mod warmup;

pub use executor::Executor;
pub use executor_load::{load_executor, load_executor_from_bytes, LoadOptions, ModelBytes};
pub use types::{FinishReason, GeneratedToken, GenerationResult, TokenLogprob};
pub(crate) use unsupported::qwen35_unsupported;
pub use vision_tower::{ImageTokenLimits, VisionTower};
