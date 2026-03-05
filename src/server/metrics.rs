//! Prometheus metrics for the inference server

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use metrics_exporter_prometheus::PrometheusHandle;

use super::handlers::AppState;

/// Metric names
pub mod names {
    pub const REQUESTS_TOTAL: &str = "blazr_requests_total";
    pub const REQUESTS_ACTIVE: &str = "blazr_requests_active";
    pub const REQUEST_DURATION: &str = "blazr_request_duration_seconds";
    pub const TOKENS_PROMPTED: &str = "blazr_tokens_prompted_total";
    pub const TOKENS_GENERATED: &str = "blazr_tokens_generated_total";
    pub const MODELS_LOADED: &str = "blazr_models_loaded";
}

/// Install the global Prometheus recorder and return its handle for rendering.
/// Must be called once at startup before any metrics are recorded.
pub fn install_recorder() -> Result<PrometheusHandle, Box<dyn std::error::Error>> {
    let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    let handle = builder.install_recorder()?;

    // Register metric descriptions
    metrics::describe_counter!(names::REQUESTS_TOTAL, "Total HTTP requests");
    metrics::describe_gauge!(names::REQUESTS_ACTIVE, "Currently active requests");
    metrics::describe_histogram!(names::REQUEST_DURATION, "Request duration in seconds");
    metrics::describe_counter!(names::TOKENS_PROMPTED, "Total prompt tokens processed");
    metrics::describe_counter!(names::TOKENS_GENERATED, "Total tokens generated");
    metrics::describe_gauge!(names::MODELS_LOADED, "Number of loaded models");

    Ok(handle)
}

/// Middleware that records per-request metrics
pub async fn metrics_middleware(request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    metrics::counter!(names::REQUESTS_TOTAL, "method" => method.clone(), "path" => path.clone())
        .increment(1);
    metrics::gauge!(names::REQUESTS_ACTIVE).increment(1.0);

    let start = Instant::now();
    let response = next.run(request).await;
    let duration = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();

    metrics::gauge!(names::REQUESTS_ACTIVE).decrement(1.0);
    metrics::histogram!(names::REQUEST_DURATION, "method" => method, "path" => path, "status" => status)
        .record(duration);

    response
}

/// Record token counts after a generation completes
pub fn record_tokens(prompt_tokens: usize, completion_tokens: usize) {
    metrics::counter!(names::TOKENS_PROMPTED).increment(prompt_tokens as u64);
    metrics::counter!(names::TOKENS_GENERATED).increment(completion_tokens as u64);
}

/// GET /metrics — Prometheus text exposition format
pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> Response {
    // Update loaded models gauge
    let loaded_count = state.scheduler.list_loaded().await.len() as f64;
    metrics::gauge!(names::MODELS_LOADED).set(loaded_count);

    let output = state.metrics_handle.render();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        output,
    )
        .into_response()
}
