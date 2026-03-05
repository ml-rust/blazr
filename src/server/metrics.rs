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
    pub const TIME_TO_FIRST_TOKEN: &str = "blazr_time_to_first_token_seconds";
    pub const TOKENS_PER_SECOND: &str = "blazr_tokens_per_second";
    pub const INFLIGHT_TOKENS: &str = "blazr_inflight_tokens";
    pub const SCHEDULER_MODELS_AVAILABLE: &str = "blazr_scheduler_models_available";
    pub const SCHEDULER_EVICTIONS_TOTAL: &str = "blazr_scheduler_evictions_total";
    pub const SCHEDULER_LOADS_TOTAL: &str = "blazr_scheduler_loads_total";
    pub const QUEUE_DEPTH: &str = "blazr_queue_depth";
    pub const ACTIVE_DECODE_SLOTS: &str = "blazr_active_decode_slots";
    pub const TOKEN_BUDGET_UTILIZATION: &str = "blazr_token_budget_utilization_ratio";
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
    metrics::describe_histogram!(names::TIME_TO_FIRST_TOKEN, "Time to first token in seconds");
    metrics::describe_histogram!(
        names::TOKENS_PER_SECOND,
        "Decode tokens per second per request"
    );
    metrics::describe_gauge!(
        names::INFLIGHT_TOKENS,
        "Estimated in-flight tokens (prompt + decode)"
    );
    metrics::describe_gauge!(
        names::SCHEDULER_MODELS_AVAILABLE,
        "Number of models available in model directory"
    );
    metrics::describe_counter!(
        names::SCHEDULER_EVICTIONS_TOTAL,
        "Total model evictions from scheduler"
    );
    metrics::describe_counter!(
        names::SCHEDULER_LOADS_TOTAL,
        "Total model loads by scheduler"
    );
    metrics::describe_gauge!(
        names::QUEUE_DEPTH,
        "Number of requests waiting for admission (HPA/KEDA signal)"
    );
    metrics::describe_gauge!(
        names::ACTIVE_DECODE_SLOTS,
        "Number of active decode slots (requests currently generating)"
    );
    metrics::describe_gauge!(
        names::TOKEN_BUDGET_UTILIZATION,
        "Ratio of in-flight tokens to max budget (0.0-1.0, HPA/KEDA signal)"
    );

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

/// Record time-to-first-token (TTFT) in seconds
pub fn record_ttft(ttft_secs: f64) {
    metrics::histogram!(names::TIME_TO_FIRST_TOKEN).record(ttft_secs);
}

/// Record decode throughput (tokens/second) for a completed request
pub fn record_tokens_per_second(tps: f64) {
    if tps > 0.0 {
        metrics::histogram!(names::TOKENS_PER_SECOND).record(tps);
    }
}

/// Increment/decrement the active decode slot gauge
pub fn adjust_decode_slots(delta: f64) {
    if delta >= 0.0 {
        metrics::gauge!(names::ACTIVE_DECODE_SLOTS).increment(delta);
    } else {
        metrics::gauge!(names::ACTIVE_DECODE_SLOTS).decrement(-delta);
    }
}

/// Adjust the in-flight token gauge (positive = add, negative = subtract)
pub fn adjust_inflight_tokens(delta: f64) {
    if delta >= 0.0 {
        metrics::gauge!(names::INFLIGHT_TOKENS).increment(delta);
    } else {
        metrics::gauge!(names::INFLIGHT_TOKENS).decrement(-delta);
    }
}

/// GET /metrics — Prometheus text exposition format
pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> Response {
    // Update loaded models gauge
    let loaded_count = state.scheduler.list_loaded().await.len() as f64;
    metrics::gauge!(names::MODELS_LOADED).set(loaded_count);

    // Update available models gauge
    let available_count = state
        .scheduler
        .list_available()
        .map(|v| v.len())
        .unwrap_or(0) as f64;
    metrics::gauge!(names::SCHEDULER_MODELS_AVAILABLE).set(available_count);

    // Update token budget utilization for autoscaling
    if state.max_inflight_tokens > 0 {
        let current = state
            .inflight_tokens
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let ratio = current / state.max_inflight_tokens as f64;
        metrics::gauge!(names::TOKEN_BUDGET_UTILIZATION).set(ratio);
    }

    let output = state.metrics_handle.render();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        output,
    )
        .into_response()
}
