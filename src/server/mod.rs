//! HTTP server for inference
//!
//! Provides OpenAI-compatible REST API.

mod config_watch;
mod handlers;
mod management;
pub mod metrics;
mod routes;
mod streaming;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::Router;
use tokio::net::TcpListener;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;

use crate::config::ServerConfig;
use crate::engine::Scheduler;

pub use handlers::AppState;
pub use routes::api_routes;
pub use streaming::{create_chat_stream, create_completion_stream, StreamToken};

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

/// Request ID for tracing (injected into request extensions for handler access)
#[derive(Clone)]
#[allow(dead_code)]
struct RequestId(String);

/// Wrapper to store request_logging config flag in request extensions
#[derive(Clone)]
struct RequestLogging(bool);

/// Request logging middleware — logs method, path, status, latency, and request_id
/// Respects ServerConfig.request_logging flag; if disabled, only injects request_id.
async fn request_logging(request: Request, next: Next) -> Response {
    let id = uuid::Uuid::new_v4().to_string();
    let logging_enabled = request
        .extensions()
        .get::<RequestLogging>()
        .is_none_or(|rl| rl.0);

    // Inject request_id into extensions for handlers to use
    let mut request = request;
    request.extensions_mut().insert(RequestId(id.clone()));

    if !logging_enabled {
        return next.run(request).await;
    }

    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let start = std::time::Instant::now();

    let response = next.run(request).await;
    let status = response.status().as_u16();
    let latency_ms = start.elapsed().as_millis();

    tracing::info!(
        request_id = %id,
        method = %method,
        path = %path,
        status = status,
        latency_ms = latency_ms,
        "request completed"
    );

    response
}

/// API key auth middleware
async fn auth_middleware(request: Request, next: Next) -> Response {
    let keys = request.extensions().get::<ApiKeys>().cloned();

    if let Some(ApiKeys(valid_keys)) = keys {
        if !valid_keys.is_empty() {
            let auth_header = request
                .headers()
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok());

            match auth_header {
                Some(header) if header.starts_with("Bearer ") => {
                    let token = &header[7..];
                    if !valid_keys.iter().any(|k| k == token) {
                        return (
                            StatusCode::UNAUTHORIZED,
                            axum::Json(serde_json::json!({
                                "error": {
                                    "message": "Invalid API key",
                                    "type": "invalid_api_key"
                                }
                            })),
                        )
                            .into_response();
                    }
                }
                _ => {
                    return (
                        StatusCode::UNAUTHORIZED,
                        axum::Json(serde_json::json!({
                            "error": {
                                "message": "Missing Authorization header. Use: Authorization: Bearer <api-key>",
                                "type": "invalid_api_key"
                            }
                        })),
                    )
                        .into_response();
                }
            }
        }
    }

    next.run(request).await
}

/// Wrapper to store valid API keys in request extensions
#[derive(Clone)]
struct ApiKeys(Vec<String>);

/// Middleware layer that injects API keys into request extensions
async fn inject_api_keys(
    axum::extract::State(keys): axum::extract::State<Vec<String>>,
    mut request: Request,
    next: Next,
) -> Response {
    if !keys.is_empty() {
        request.extensions_mut().insert(ApiKeys(keys));
    }
    next.run(request).await
}

/// Middleware layer that injects request logging config into request extensions
async fn inject_request_logging(
    axum::extract::State(rl): axum::extract::State<RequestLogging>,
    mut request: Request,
    next: Next,
) -> Response {
    request.extensions_mut().insert(rl);
    next.run(request).await
}

/// Start the HTTP inference server with graceful shutdown
pub async fn start(
    scheduler: Arc<Scheduler<ServerRuntime>>,
    config: ServerConfig,
    api_keys: Vec<String>,
) -> Result<()> {
    // Install Prometheus metrics recorder
    let metrics_handle = metrics::install_recorder()
        .map_err(|e| anyhow::anyhow!("Failed to install Prometheus metrics recorder: {}", e))?;
    let state = Arc::new(AppState::new(scheduler, metrics_handle));

    // Start config file watcher for hot-reload
    config_watch::spawn_config_watcher(state.user_config.clone());

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

    // Protected routes (require auth if api_key is set)
    let protected = api_routes().route_layer(middleware::from_fn(auth_middleware));

    // Health and metrics are exempt from auth
    let health_route = Router::new()
        .route("/health", axum::routing::get(handlers::health))
        .route("/metrics", axum::routing::get(metrics::metrics_handler));

    let app = Router::new()
        .merge(health_route)
        .merge(protected)
        .layer(cors)
        .layer(middleware::from_fn(request_logging))
        .layer(middleware::from_fn(metrics::metrics_middleware))
        .layer(TimeoutLayer::with_status_code(
            axum::http::StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(config.request_timeout_secs),
        ))
        .layer(RequestBodyLimitLayer::new(config.max_body_size))
        .layer(tower::limit::ConcurrencyLimitLayer::new(
            config.max_concurrent_requests,
        ))
        .layer(middleware::from_fn_with_state(
            api_keys.clone(),
            inject_api_keys,
        ))
        .layer(middleware::from_fn_with_state(
            RequestLogging(config.request_logging),
            inject_request_logging,
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
    if !api_keys.is_empty() {
        tracing::info!("  Authentication: enabled ({} key(s))", api_keys.len());
    }
    tracing::info!("API endpoints:");
    tracing::info!("  GET  /health - Health check (no auth required)");
    tracing::info!("  GET  /v1/models - List models");
    tracing::info!("  GET  /v1/models/{{id}} - Model details");
    tracing::info!("  POST /v1/completions - Text completion");
    tracing::info!("  POST /v1/chat/completions - Chat completion");
    tracing::info!("  POST /tokenize - Tokenize text");
    tracing::info!("  POST /detokenize - Detokenize tokens");
    tracing::info!("  GET  /api/tags - List local models with metadata");
    tracing::info!("  POST /api/show - Model details");
    tracing::info!("  GET  /api/ps - Currently loaded models");
    tracing::info!("  GET  /metrics - Prometheus metrics (no auth required)");
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
