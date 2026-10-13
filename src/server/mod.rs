//! HTTP server for inference
//!
//! Provides OpenAI-compatible REST API.

mod anthropic_api;
#[cfg(feature = "audio")]
mod audio;
mod chat_api;
pub mod decision;
mod gen_core;
pub mod handlers;
mod management;
mod media;
mod responses_api;
mod routes;
mod serving;
mod text_api;

pub use crate::telemetry::slo;
#[cfg(test)]
pub(crate) use crate::telemetry::test_recorder;
pub(in crate::server) use anthropic_api::*;
pub(in crate::server) use chat_api::*;
pub use gen_core::*;
pub use handlers::AppState;
pub use media::*;
pub(in crate::server) use responses_api::*;
pub use routes::api_routes;
pub use serving::*;
pub(in crate::server) use text_api::*;
