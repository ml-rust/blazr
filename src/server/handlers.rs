//! HTTP request handlers
//!
//! Core handlers: health, models, tokenize/detokenize.
//! Completion and chat handlers are in separate modules.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use tokio::sync::RwLock;

use super::generation::error_response;
use super::metrics;
use crate::config::UserConfig;
use crate::engine::{Scheduler, SlotManager};

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

/// Shared application state
pub struct AppState {
    pub scheduler: Arc<Scheduler<ServerRuntime>>,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    pub user_config: Arc<RwLock<UserConfig>>,
    /// Current in-flight token count (prompt + estimated decode tokens)
    pub inflight_tokens: AtomicUsize,
    /// Maximum in-flight token budget (0 = unlimited)
    pub max_inflight_tokens: usize,
    /// Inference slot manager
    pub slot_manager: SlotManager,
}

impl AppState {
    pub fn new(
        scheduler: Arc<Scheduler<ServerRuntime>>,
        metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    ) -> Self {
        Self {
            scheduler,
            metrics_handle,
            user_config: Arc::new(RwLock::new(UserConfig::load())),
            inflight_tokens: AtomicUsize::new(0),
            max_inflight_tokens: 0,
            slot_manager: SlotManager::new(0), // unlimited by default
        }
    }

    pub fn with_max_inflight_tokens(mut self, max: usize) -> Self {
        self.max_inflight_tokens = max;
        self
    }

    /// Try to admit a request with the given token budget.
    /// Returns `false` (and does not increment) if the budget would be exceeded.
    pub fn try_admit(&self, tokens: usize) -> bool {
        if self.max_inflight_tokens == 0 {
            self.inflight_tokens.fetch_add(tokens, Ordering::Relaxed);
            metrics::adjust_inflight_tokens(tokens as f64);
            return true;
        }
        loop {
            let current = self.inflight_tokens.load(Ordering::Relaxed);
            if current + tokens > self.max_inflight_tokens {
                return false;
            }
            if self
                .inflight_tokens
                .compare_exchange_weak(
                    current,
                    current + tokens,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                metrics::adjust_inflight_tokens(tokens as f64);
                return true;
            }
        }
    }

    /// Release tokens back to the budget after a request completes
    pub fn release(&self, tokens: usize) {
        self.inflight_tokens.fetch_sub(tokens, Ordering::Relaxed);
        metrics::adjust_inflight_tokens(-(tokens as f64));
    }
}

/// Health check endpoint — returns status and loaded model info
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let loaded = state
        .scheduler
        .list_loaded()
        .await
        .into_iter()
        .map(|m| m.name)
        .collect::<Vec<_>>();

    let gpu_memory = get_gpu_memory_info();

    let response = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        loaded_models: loaded,
        gpu_memory,
    };
    (StatusCode::OK, Json(response)).into_response()
}

/// Query GPU memory info (CUDA only)
fn get_gpu_memory_info() -> Option<GpuMemoryInfo> {
    #[cfg(feature = "cuda")]
    {
        use boostr::Runtime;
        let device = <boostr::CudaRuntime as Runtime>::default_device();
        if let Ok((free, total)) = device.memory_info() {
            let used = total.saturating_sub(free);
            return Some(GpuMemoryInfo {
                used_bytes: used,
                free_bytes: free,
                total_bytes: total,
                used_gb: used as f64 / (1024.0 * 1024.0 * 1024.0),
                total_gb: total as f64 / (1024.0 * 1024.0 * 1024.0),
            });
        }
    }
    None
}

/// List available models
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.scheduler.list_available() {
        Ok(models) => {
            let response = ModelsResponse {
                object: "list".to_string(),
                data: models
                    .iter()
                    .map(|m| ModelInfo {
                        id: m.name.clone(),
                        object: "model".to_string(),
                        created: 0,
                        owned_by: "local".to_string(),
                    })
                    .collect(),
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Get single model details
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Response {
    match state.scheduler.list_available() {
        Ok(models) => {
            if let Some(m) = models.iter().find(|m| m.name == model_id) {
                let info = ModelInfo {
                    id: m.name.clone(),
                    object: "model".to_string(),
                    created: 0,
                    owned_by: "local".to_string(),
                };
                (StatusCode::OK, Json(info)).into_response()
            } else {
                error_response(
                    StatusCode::NOT_FOUND,
                    &format!("Model '{}' not found", model_id),
                    "invalid_request_error",
                )
            }
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &e.to_string(),
            "server_error",
        ),
    }
}

/// Tokenize text endpoint
pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenizeRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "invalid_request_error",
            );
        }
    };

    match executor.tokenizer().encode(&request.content) {
        Ok(tokens) => {
            let response = TokenizeResponse {
                tokens: tokens.iter().map(|&t| t as i64).collect(),
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Tokenization failed: {}", e),
            "server_error",
        ),
    }
}

/// Detokenize token IDs endpoint
pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetokenizeRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "invalid_request_error",
            );
        }
    };

    let token_ids: Vec<u32> = request.tokens.iter().map(|&t| t as u32).collect();
    match executor.tokenizer().decode(&token_ids) {
        Ok(text) => {
            let response = DetokenizeResponse { content: text };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Detokenization failed: {}", e),
            "server_error",
        ),
    }
}

// ── Types ──

#[derive(Serialize)]
pub struct GpuMemoryInfo {
    pub used_bytes: u64,
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub used_gb: f64,
    pub total_gb: f64,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub loaded_models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory: Option<GpuMemoryInfo>,
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Deserialize)]
pub struct TokenizeRequest {
    pub model: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<i64>,
}

#[derive(Deserialize)]
pub struct DetokenizeRequest {
    pub model: String,
    pub tokens: Vec<i64>,
}

#[derive(Serialize)]
pub struct DetokenizeResponse {
    pub content: String,
}

// ── Slot management ──

/// Create a new inference slot
pub async fn create_slot(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateSlotRequest>,
) -> Response {
    match state.slot_manager.allocate(&request.model).await {
        Ok(id) => {
            let response = SlotResponse {
                id,
                model: request.model,
                status: "active".to_string(),
            };
            (StatusCode::CREATED, Json(response)).into_response()
        }
        Err(e) => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &e,
            "slot_allocation_failed",
        ),
    }
}

/// List all active inference slots
pub async fn list_slots(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let slots = state.slot_manager.list().await;
    let response: Vec<SlotListEntry> = slots
        .into_iter()
        .map(|s| SlotListEntry {
            id: s.id,
            model: s.model,
            total_tokens: s.total_tokens,
            idle_seconds: s.last_accessed.elapsed().as_secs(),
        })
        .collect();
    (StatusCode::OK, Json(response)).into_response()
}

/// Free an inference slot
pub async fn delete_slot(
    State(state): State<Arc<AppState>>,
    Path(slot_id): Path<String>,
) -> Response {
    if state.slot_manager.free(&slot_id).await {
        StatusCode::NO_CONTENT.into_response()
    } else {
        error_response(
            StatusCode::NOT_FOUND,
            &format!("Slot '{}' not found", slot_id),
            "slot_not_found",
        )
    }
}

#[derive(Deserialize)]
pub struct CreateSlotRequest {
    pub model: String,
}

#[derive(Serialize)]
pub struct SlotResponse {
    pub id: String,
    pub model: String,
    pub status: String,
}

#[derive(Serialize)]
pub struct SlotListEntry {
    pub id: String,
    pub model: String,
    pub total_tokens: usize,
    pub idle_seconds: u64,
}
