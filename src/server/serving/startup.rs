//! Server startup, middleware, and TLS configuration.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use axum::extract::Request;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::Router;
use tokio::net::TcpListener;
use tower_http::cors::{preflight_request_headers, AllowOrigin, Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;

use crate::config::{FileConfig, LiveConfig, ReasoningFormat, ServerConfig};
use crate::engine::Scheduler;

use super::auth::{auth_middleware, inject_api_keys};
use super::concurrency::{limit_concurrency, ConcurrencyLimit};
use super::config_watch;
use super::metrics;
use super::tls::build_tls_config;
use crate::server::handlers::{self, AppState, Serving};
use crate::server::routes::api_routes;

use crate::runtime::ServerRuntime;

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

/// Middleware layer that injects request logging config into request extensions
async fn inject_request_logging(
    axum::extract::State(rl): axum::extract::State<RequestLogging>,
    mut request: Request,
    next: Next,
) -> Response {
    request.extensions_mut().insert(rl);
    next.run(request).await
}

/// The user config a running server reads, and the file it reloads from.
pub struct LiveSettings {
    /// The config requests read their live defaults from.
    pub config: Arc<LiveConfig>,
    /// File the watcher reloads. `None` runs without hot reload.
    pub path: Option<PathBuf>,
    /// `--reasoning-format`. Wins over the live `server.reasoning_format`.
    pub reasoning_format_flag: Option<ReasoningFormat>,
}

impl LiveSettings {
    /// A fixed config built from `server`, with no hot reload.
    pub fn fixed(server: &ServerConfig) -> Self {
        let file = FileConfig {
            server: server.clone(),
            ..Default::default()
        };
        Self {
            config: Arc::new(LiveConfig::new(file)),
            path: None,
            reasoning_format_flag: None,
        }
    }
}

/// Start the HTTP inference server with graceful shutdown
pub async fn start(
    scheduler: Arc<Scheduler<ServerRuntime>>,
    config: ServerConfig,
    api_keys: Vec<String>,
) -> Result<()> {
    let live = LiveSettings::fixed(&config);
    start_with_batch(
        scheduler,
        config,
        live,
        api_keys,
        Serving::OnDemand,
        Vec::new(),
        #[cfg(feature = "audio")]
        Vec::new(),
        #[cfg(feature = "audio")]
        Vec::new(),
        None,
    )
    .await
}

/// Start the HTTP server over the executors in `serving`.
///
/// A request for the start model runs on its replica's batching scheduler
/// when it has one. The caller spawns the task that drives each scheduler
/// (see `cli/serve.rs`).
#[allow(clippy::too_many_arguments)]
pub async fn start_with_batch(
    scheduler: Arc<Scheduler<ServerRuntime>>,
    config: ServerConfig,
    live: LiveSettings,
    api_keys: Vec<String>,
    serving: Serving,
    vision_embedders: Vec<(
        String,
        Arc<crate::server::handlers::VisionEmbedder>,
        std::path::PathBuf,
    )>,
    #[cfg(feature = "audio")] asr_models: Vec<(
        String,
        Arc<crate::server::handlers::AsrBundle>,
        std::path::PathBuf,
    )>,
    #[cfg(feature = "audio")] tts_models: Vec<(
        String,
        Arc<crate::server::handlers::TtsBundle>,
        std::path::PathBuf,
    )>,
    voice_dir: Option<std::path::PathBuf>,
) -> Result<()> {
    // Install Prometheus metrics recorder
    let metrics_handle = metrics::install_recorder()
        .map_err(|e| anyhow::anyhow!("Failed to install Prometheus metrics recorder: {}", e))?;
    if let Some(slo) = &config.slo {
        crate::server::slo::install(slo.clone());
        crate::server::slo::publish();
    }
    let state = Arc::new(
        AppState::new(scheduler, metrics_handle)
            .with_max_inflight_tokens(config.max_inflight_tokens)
            .with_live_config(Arc::clone(&live.config), live.reasoning_format_flag)
            .with_voice_dir(voice_dir)
            .with_serving(serving),
    );

    for (name, embedder, path) in vision_embedders {
        state.register_vision_embedder(name, embedder, path).await;
    }
    #[cfg(feature = "audio")]
    for (name, bundle, path) in asr_models {
        state.register_asr_model(name, bundle, path).await;
    }
    #[cfg(feature = "audio")]
    for (name, bundle, path) in tts_models {
        state.register_tts_model(name, bundle, path).await;
    }

    // Start config file watcher for hot-reload
    if let Some(path) = live.path {
        config_watch::spawn_config_watcher(live.config, path);
    }

    // Start keep-alive reaper: evicts models that exceed their per-request keep_alive TTL
    {
        let scheduler = Arc::clone(&state.scheduler);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                scheduler.evict_expired().await;
            }
        });
    }

    let cors = cors_layer(&config)?;

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
        .layer(middleware::from_fn_with_state(
            ConcurrencyLimit::new(config.max_concurrent_requests),
            limit_concurrency,
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
    let tls_config = match (&config.tls_cert, &config.tls_key) {
        (Some(cert), Some(key)) => Some(build_tls_config(cert, key)?),
        (None, None) => None,
        _ => anyhow::bail!("HTTPS needs both a TLS certificate and a TLS private key"),
    };

    let listener = TcpListener::bind(&addr).await?;
    let scheme = if tls_config.is_some() {
        "https"
    } else {
        "http"
    };

    log_startup_info(scheme, &addr, &config, &api_keys);

    // Graceful shutdown on SIGTERM/SIGINT
    #[cfg(unix)]
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|e| anyhow::anyhow!("Failed to install the SIGTERM handler: {e}"))?;
    let shutdown = async move {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
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

    if let Some(tls) = tls_config {
        let tls_acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(tls));

        // Manual TLS accept loop with graceful shutdown
        let shutdown = std::pin::pin!(shutdown);
        let mut shutdown = shutdown;
        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, _addr)) => {
                            let acceptor = tls_acceptor.clone();
                            let app = app.clone();
                            tokio::spawn(async move {
                                let Ok(tls_stream) = acceptor.accept(stream).await else {
                                    return;
                                };
                                let io = hyper_util::rt::TokioIo::new(tls_stream);
                                let service = hyper_util::service::TowerToHyperService::new(
                                    app.into_service(),
                                );
                                let _ = hyper_util::server::conn::auto::Builder::new(
                                    hyper_util::rt::TokioExecutor::new(),
                                )
                                .serve_connection(io, service)
                                .await;
                            });
                        }
                        Err(e) => {
                            tracing::error!("TCP accept error: {}", e);
                        }
                    }
                }
                _ = &mut shutdown => break,
            }
        }
    } else {
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await?;
    }

    tracing::info!("Server shut down gracefully");
    Ok(())
}

/// The CORS layer `config` sets: off, any origin, or the `cors_origins`
/// list. `Vary` names the three preflight request headers on every
/// response, whatever the origin policy.
///
/// # Errors
///
/// An error naming the first `cors_origins` entry that is no origin.
fn cors_layer(config: &ServerConfig) -> Result<CorsLayer> {
    let cors = if !config.cors_enabled {
        CorsLayer::new()
    } else if config.cors_origins.is_empty() {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        crate::config::check_cors_origins(&config.cors_origins).map_err(anyhow::Error::msg)?;
        let origins = config
            .cors_origins
            .iter()
            .map(|o| {
                o.parse().map_err(|_| {
                    anyhow::anyhow!("server.cors_origins entry {o:?} is not an origin")
                })
            })
            .collect::<Result<Vec<axum::http::HeaderValue>>>()?;
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(origins))
            .allow_methods(Any)
            .allow_headers(Any)
    };
    Ok(cors.vary(preflight_request_headers().collect::<Vec<_>>()))
}

fn log_startup_info(scheme: &str, addr: &str, config: &ServerConfig, api_keys: &[String]) {
    tracing::info!("Server listening on {}://{}", scheme, addr);
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
    for route in crate::server::route_list::served_routes() {
        tracing::info!(
            "  {:<6} {} - {}",
            route.method,
            crate::server::route_list::display_path(route.path),
            route.what
        );
    }
}
