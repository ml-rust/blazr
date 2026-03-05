//! Model management API handlers
//!
//! Provides endpoints for listing, inspecting, and monitoring loaded models.

use std::sync::Arc;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::handlers::AppState;

/// GET /api/tags — List local models with metadata
pub async fn list_tags(State(state): State<Arc<AppState>>) -> Response {
    match state.scheduler.list_available() {
        Ok(models) => {
            let model_list: Vec<ModelTag> = models
                .iter()
                .map(|m| {
                    let size = dir_size(&m.path);
                    let modified = std::fs::metadata(&m.path)
                        .and_then(|meta| meta.modified())
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);

                    ModelTag {
                        name: m.name.clone(),
                        model: m.name.clone(),
                        modified_at: chrono::DateTime::from_timestamp(modified, 0)
                            .map(|dt| dt.to_rfc3339())
                            .unwrap_or_default(),
                        size,
                        details: ModelDetails {
                            format: m.format.clone(),
                            family: String::new(),
                            parameter_size: String::new(),
                            quantization_level: String::new(),
                        },
                    }
                })
                .collect();

            (StatusCode::OK, Json(TagsResponse { models: model_list })).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// POST /api/show — Show model details
pub async fn show_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ShowRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.name).await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": format!("model not found: {}", e) })),
            )
                .into_response();
        }
    };

    let config = executor.config();
    let template_name = format!("{:?}", executor.chat_template());

    let response = ShowResponse {
        modelfile: String::new(),
        parameters: format!(
            "num_ctx {}\ntemperature 1.0\ntop_p 1.0",
            config.inference.max_context_len.unwrap_or(4096)
        ),
        template: template_name,
        details: ModelDetails {
            format: String::new(),
            family: config.model_type().to_string(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        },
        model_info: serde_json::json!({
            "model_type": config.model_type(),
            "vocab_size": config.vocab_size(),
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "max_seq_len": config.model.max_seq_len,
        }),
    };

    (StatusCode::OK, Json(response)).into_response()
}

/// GET /api/ps — List currently loaded models
pub async fn list_running(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let loaded = state.scheduler.list_loaded().await;
    let models: Vec<RunningModel> = loaded
        .iter()
        .map(|m| RunningModel {
            name: m.name.clone(),
            model: m.name.clone(),
            size: dir_size(&m.path),
            expires_at: String::new(),
            size_vram: 0,
            details: ModelDetails {
                format: String::new(),
                family: String::new(),
                parameter_size: String::new(),
                quantization_level: String::new(),
            },
        })
        .collect();

    (StatusCode::OK, Json(RunningResponse { models }))
}

/// Calculate directory size in bytes
fn dir_size(path: &std::path::Path) -> u64 {
    if path.is_file() {
        return std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }
    match std::fs::read_dir(path) {
        Ok(entries) => entries
            .filter_map(|e| match e {
                Ok(entry) => Some(entry),
                Err(e) => {
                    tracing::warn!(error = %e, "failed to read directory entry");
                    None
                }
            })
            .map(|e| std::fs::metadata(e.path()).map(|m| m.len()).unwrap_or(0))
            .sum(),
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "failed to read model directory size");
            0
        }
    }
}

// ── Request/Response types ──

#[derive(Serialize)]
pub struct TagsResponse {
    pub models: Vec<ModelTag>,
}

#[derive(Serialize)]
pub struct ModelTag {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    pub details: ModelDetails,
}

#[derive(Serialize)]
pub struct ModelDetails {
    pub format: String,
    pub family: String,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Deserialize)]
pub struct ShowRequest {
    pub name: String,
}

#[derive(Serialize)]
pub struct ShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: ModelDetails,
    pub model_info: serde_json::Value,
}

#[derive(Serialize)]
pub struct RunningResponse {
    pub models: Vec<RunningModel>,
}

#[derive(Serialize)]
pub struct RunningModel {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub expires_at: String,
    pub size_vram: u64,
    pub details: ModelDetails,
}
