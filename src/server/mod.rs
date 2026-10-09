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
mod config_watch;
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
mod pooling;
mod reasoning;
mod rerank;
mod responses;
mod responses_media;
mod responses_types;
mod routes;
pub mod slo;
mod startup;
mod streaming;
mod tls;
mod tool_calls;
mod tools;

pub use handlers::AppState;
pub use reasoning::{
    prompt_opens_reasoning, split_reasoning, split_reasoning_for_format, ReasoningParser,
    ReasoningTags,
};
pub use routes::api_routes;
pub use startup::{start, start_with_batch};
pub use streaming::{create_chat_stream, create_completion_stream, StreamToken};
