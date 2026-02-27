//! HTTP server for inference
//!
//! Provides OpenAI-compatible REST API.

mod handlers;
mod routes;
mod streaming;

use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::config::ServerConfig;
use crate::engine::Scheduler;

pub use handlers::AppState;
pub use routes::api_routes;
pub use streaming::{create_chat_stream, create_completion_stream};

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

/// Start the HTTP inference server
pub async fn start(scheduler: Arc<Scheduler<ServerRuntime>>, config: ServerConfig) -> Result<()> {
    let state = Arc::new(AppState::new(scheduler));

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .merge(api_routes())
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = config.addr();
    let listener = TcpListener::bind(&addr).await?;

    tracing::info!("Server listening on http://{}", addr);
    tracing::info!("API endpoints:");
    tracing::info!("  GET  /health - Health check");
    tracing::info!("  GET  /v1/models - List models");
    tracing::info!("  POST /v1/completions - Text completion");
    tracing::info!("  POST /v1/chat/completions - Chat completion");

    axum::serve(listener, app).await?;

    Ok(())
}
