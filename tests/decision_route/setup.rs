//! The axum router in-process over a model scheduler that serves one
//! decision model, and requests sent through it with tower's `oneshot`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use blazr::boostr::DType;
use blazr::config::{DecisionConfig, DtypeRequest, DtypeSource};
use blazr::engine::Scheduler;
use blazr::server::handlers::{AppState, ServerRuntime};
use serde_json::Value;
use tower::ServiceExt;

/// The decision limits the route tests run under: every fixture fits, a
/// 70-question request passes `max_rows`, and two 5-row requests together
/// pass `max_queue_rows`.
pub const MAX_ROWS: usize = 64;
pub const MAX_QUEUE_ROWS: usize = 8;

/// A served decision model.
pub struct Served {
    pub router: Router,
    pub state: Arc<AppState>,
    /// The name requests use: the model's entry in its parent directory.
    pub name: String,
}

/// Serve the model at `path` (a checkpoint directory or a `.gguf`) as the
/// start model of a scheduler over its parent directory. `config` is the
/// sidecar for a model that ships none beside it. `batch_wait_ms` is how
/// long the worker waits for more requests to merge. The prefix-state
/// cache is off, so each request runs its own prefix and merges as it
/// would without one.
pub fn serve(path: &Path, config: Option<PathBuf>, batch_wait_ms: u64) -> Served {
    let decision = DecisionConfig {
        config,
        max_rows: MAX_ROWS,
        max_queue_rows: MAX_QUEUE_ROWS,
        batch_wait_ms,
        prefix_cache_mb: 0,
        ..DecisionConfig::default()
    };
    serve_config(path, decision)
}

/// Serve the model at `path` as [`serve`] does, with the decision settings
/// `decision`. `start_model` is set to the model's name.
pub fn serve_config(path: &Path, decision: DecisionConfig) -> Served {
    let parent = path
        .parent()
        .unwrap_or_else(|| panic!("{} has no parent directory", path.display()));
    let name = path
        .file_name()
        .unwrap_or_else(|| panic!("{} has no file name", path.display()))
        .to_string_lossy()
        .into_owned();
    let device = blazr::runtime::open_device(0)
        .unwrap_or_else(|e| panic!("opening the server runtime's device 0: {e}"));
    let decision = DecisionConfig {
        start_model: Some(name.clone()),
        ..decision
    };
    let scheduler = Scheduler::<ServerRuntime>::new(parent.to_path_buf(), device)
        .with_dtype(Some(DtypeRequest {
            dtype: DType::F32,
            source: DtypeSource::Flag,
        }))
        .with_decision(decision);
    let metrics = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    let state = Arc::new(AppState::new(Arc::new(scheduler), metrics));
    Served {
        router: blazr::server::api_routes().with_state(Arc::clone(&state)),
        state,
        name,
    }
}

async fn send(router: &Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = router
        .clone()
        .oneshot(request)
        .await
        .unwrap_or_else(|e| panic!("router call: {e}"));
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap_or_else(|e| panic!("response body: {e}"));
    let value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "status {status}: the body is not JSON ({e}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, value)
}

/// POST `body` (raw bytes) to `uri`.
pub async fn post_raw(router: &Router, uri: &str, body: Vec<u8>) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap_or_else(|e| panic!("building POST {uri}: {e}"));
    send(router, request).await
}

/// POST the JSON `body` to `uri`.
pub async fn post(router: &Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|e| panic!("encoding the body: {e}"));
    post_raw(router, uri, bytes).await
}

/// GET `uri`.
pub async fn get(router: &Router, uri: &str) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("building GET {uri}: {e}"));
    send(router, request).await
}
