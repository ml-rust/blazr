//! Prometheus exporter, request middleware and `/metrics` for the server.
//!
//! The metric names and the recording helpers live in [`crate::telemetry`].

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use metrics_exporter_prometheus::PrometheusHandle;

use super::handlers::AppState;

pub use crate::telemetry::names;
#[cfg(feature = "cuda")]
pub use crate::telemetry::record::update_vram_used;
pub use crate::telemetry::record::{
    adjust_decode_slots, adjust_inflight_tokens, record_e2e, record_itl, record_prefix_cache_hit,
    record_prefix_cache_miss, record_tokens, record_tokens_per_second, record_ttft,
    update_kv_cache_utilization, update_replicas,
};

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
        "Requests waiting for a decode slot or for token-budget admission (HPA/KEDA signal)"
    );
    metrics::gauge!(names::QUEUE_DEPTH).increment(0.0);
    metrics::describe_counter!(
        names::SLO_VIOLATIONS_TOTAL,
        "Latency samples taken while a rolling percentile exceeds its server.slo threshold"
    );
    metrics::describe_gauge!(
        names::SLO_BATCH_HELD,
        "1 while a P95 TTFT or ITL breach holds batch growth, else 0"
    );
    metrics::describe_gauge!(
        names::DATA_PARALLEL_REPLICAS,
        "Data-parallel replicas of the model loaded at start"
    );
    metrics::describe_gauge!(
        names::REPLICA_ACTIVE_REQUESTS,
        "Requests in flight on each data-parallel replica, by replica"
    );
    metrics::describe_gauge!(
        names::ACTIVE_DECODE_SLOTS,
        "Number of active decode slots (requests currently generating)"
    );
    metrics::describe_gauge!(
        names::TOKEN_BUDGET_UTILIZATION,
        "Ratio of in-flight tokens to max budget (0.0-1.0, HPA/KEDA signal)"
    );
    metrics::describe_histogram!(
        names::INTER_TOKEN_LATENCY,
        "Inter-token latency (time between consecutive tokens) in seconds"
    );
    metrics::describe_gauge!(names::VRAM_USED_BYTES, "GPU VRAM currently in use (bytes)");
    metrics::describe_gauge!(
        names::KV_CACHE_UTILIZATION,
        "Ratio of allocated KV cache blocks to total pool (0.0-1.0)"
    );
    metrics::describe_counter!(
        names::PREFIX_CACHE_HITS,
        "Total prefix cache hits (shared KV blocks reused)"
    );
    metrics::describe_counter!(
        names::PREFIX_CACHE_MISSES,
        "Total prefix cache misses (new KV blocks allocated)"
    );
    metrics::describe_counter!(
        names::DECISION_REQUESTS_TOTAL,
        "Decision requests a model's worker finished, by model and outcome"
    );
    metrics::describe_counter!(
        names::DECISION_ROWS_TOTAL,
        "Prompt rows of finished decision requests, by model"
    );
    metrics::describe_counter!(
        names::DECISION_PREFILL_TOKENS_TOTAL,
        "Tokens decision runs prefilled, a shared prefix counted once, by model"
    );
    metrics::describe_histogram!(
        names::DECISION_LATENCY,
        "Seconds from queueing a decision request to its answers, by model"
    );
    metrics::describe_gauge!(
        names::DECISION_QUEUE_ROWS,
        "Decision rows queued or running, by model"
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

    // KV cache utilization over every batching scheduler's block pool
    let (allocated, total) = state
        .serving
        .schedulers()
        .iter()
        .filter_map(|rs| rs.block_stats())
        .fold((0, 0), |(a, t), stats| {
            (a + stats.allocated_blocks, t + stats.total_blocks)
        });
    update_kv_cache_utilization(allocated, total);

    // Data-parallel replica count and per-replica load
    update_replicas(state.serving.replica_count(), &state.serving.replica_load());

    // Update VRAM usage (CUDA only)
    #[cfg(feature = "cuda")]
    {
        // Get device from first loaded model if available
        let loaded = state.scheduler.list_loaded().await;
        if !loaded.is_empty() {
            let device = boostr::CudaDevice::new(0);
            update_vram_used(&device);
        }
    }

    let output = state.metrics_handle.render();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        output,
    )
        .into_response()
}
