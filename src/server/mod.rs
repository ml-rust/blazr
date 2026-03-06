//! HTTP server for inference
//!
//! Provides OpenAI-compatible REST API.

mod chat;
mod chat_types;
mod completions;
mod config_watch;
mod gen_types;
mod generation;
mod handlers;
mod infill;
mod lora;
mod management;
pub mod metrics;
mod routes;
mod startup;
mod streaming;
mod tools;

pub use handlers::AppState;
pub use routes::api_routes;
pub use startup::{start, start_with_batch};
pub use streaming::{create_chat_stream, create_completion_stream, StreamToken};
