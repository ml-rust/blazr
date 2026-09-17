//! `GET /health` — status, loaded text models, and GPU memory info.

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Serialize;

use super::state::AppState;

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
