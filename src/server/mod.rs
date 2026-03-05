//! HTTP server for inference
//!
//! Provides OpenAI-compatible REST API.

mod handlers;
mod routes;
mod streaming;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::Router;
use tokio::net::TcpListener;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::config::ServerConfig;
use crate::engine::Scheduler;

pub use handlers::AppState;
pub use routes::api_routes;
pub use streaming::{create_chat_stream, create_completion_stream, StreamToken};

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

/// Start the HTTP inference server with graceful shutdown
pub async fn start(scheduler: Arc<Scheduler<ServerRuntime>>, config: ServerConfig) -> Result<()> {
    let state = Arc::new(AppState::new(scheduler));

    // CORS: respect cors_enabled and cors_origins config
    let cors = if config.cors_enabled {
        if config.cors_origins.is_empty() {
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        } else {
            let origins: Vec<_> = config
                .cors_origins
                .iter()
                .filter_map(|o| o.parse().ok())
                .collect();
            CorsLayer::new()
                .allow_origin(AllowOrigin::list(origins))
                .allow_methods(Any)
                .allow_headers(Any)
        }
    } else {
        CorsLayer::new()
    };

    let app = Router::new()
        .merge(api_routes())
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(config.request_timeout_secs),
        ))
        .layer(RequestBodyLimitLayer::new(config.max_body_size))
        .layer(tower::limit::ConcurrencyLimitLayer::new(
            config.max_concurrent_requests,
        ))
        .with_state(state);

    let addr = config.addr();
    let listener = TcpListener::bind(&addr).await?;

    tracing::info!("Server listening on http://{}", addr);
    tracing::info!(
        "  Timeout: {}s | Max body: {} bytes | Max concurrent: {}",
        config.request_timeout_secs,
        config.max_body_size,
        config.max_concurrent_requests
    );
    tracing::info!("API endpoints:");
    tracing::info!("  GET  /health - Health check");
    tracing::info!("  GET  /v1/models - List models");
    tracing::info!("  POST /v1/completions - Text completion");
    tracing::info!("  POST /v1/chat/completions - Chat completion");

    // Graceful shutdown on SIGTERM/SIGINT
    let shutdown = async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to install SIGTERM handler");
            tokio::select! {
                _ = ctrl_c => tracing::info!("Received SIGINT, shutting down..."),
                _ = sigterm.recv() => tracing::info!("Received SIGTERM, shutting down..."),
            }
        }
        #[cfg(not(unix))]
        {
            ctrl_c.await.ok();
            tracing::info!("Received SIGINT, shutting down...");
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    tracing::info!("Server shut down gracefully");
    Ok(())
}
