//! HTTP server for inference
//!
//! Provides OpenAI-compatible REST API.

mod anthropic;
mod anthropic_media;
mod anthropic_stream;
mod anthropic_types;
#[cfg(feature = "audio")]
mod audio;
#[cfg(feature = "audio")]
mod audio_decode;
#[cfg(feature = "audio")]
mod audio_transcribe;
mod chat;
mod chat_types;
mod chat_vision;
mod completions;
mod concurrency;
mod config_watch;
pub mod decision;
mod embeddings;
mod encoding;
mod gen_types;
mod generation;
pub mod handlers;
mod infill;
mod lora;
mod management;
pub mod metrics;
pub mod multimodal;
mod multimodal_audio;
mod reasoning;
mod rerank;
mod responses;
mod responses_media;
mod responses_types;
mod routes;
mod scheduled;
mod startup;
mod streaming;
mod tls;
mod tool_calls;
mod tools;

pub use crate::telemetry::slo;
#[cfg(test)]
pub(crate) use crate::telemetry::test_recorder;
pub use handlers::AppState;
pub use reasoning::{
    prompt_opens_reasoning, split_reasoning, split_reasoning_for_format, ReasoningParser,
    ReasoningTags,
};
pub use routes::api_routes;
pub use startup::{start, start_with_batch, LiveSettings};
pub use streaming::{create_chat_stream, create_completion_stream, StreamToken};
