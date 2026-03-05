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

/// DELETE /api/delete — Delete a local model
pub async fn delete_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DeleteRequest>,
) -> Response {
    // Unload model if currently loaded
    state.scheduler.unload(&request.name).await;

    // Find model path
    match state.scheduler.list_available() {
        Ok(models) => {
            if let Some(m) = models.iter().find(|m| m.name == request.name) {
                let path = &m.path;
                let result = if path.is_dir() {
                    std::fs::remove_dir_all(path)
                } else {
                    std::fs::remove_file(path)
                };
                match result {
                    Ok(()) => {
                        tracing::info!("Deleted model: {}", request.name);
                        StatusCode::OK.into_response()
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({ "error": format!("Failed to delete: {}", e) })),
                    )
                        .into_response(),
                }
            } else {
                (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({ "error": format!("model '{}' not found", request.name) })),
                )
                    .into_response()
            }
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// POST /api/copy — Copy/alias a model
pub async fn copy_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CopyRequest>,
) -> Response {
    match state.scheduler.list_available() {
        Ok(models) => {
            if let Some(m) = models.iter().find(|m| m.name == request.source) {
                let src = &m.path;
                // Determine destination path (sibling to source)
                let dest = if let Some(parent) = src.parent() {
                    parent.join(&request.destination)
                } else {
                    std::path::PathBuf::from(&request.destination)
                };

                if dest.exists() {
                    return (
                        StatusCode::CONFLICT,
                        Json(serde_json::json!({ "error": format!("destination '{}' already exists", request.destination) })),
                    )
                        .into_response();
                }

                let result = if src.is_dir() {
                    copy_dir_recursive(src, &dest)
                } else {
                    std::fs::copy(src, &dest).map(|_| ())
                };

                match result {
                    Ok(()) => {
                        tracing::info!(
                            "Copied model '{}' -> '{}'",
                            request.source,
                            request.destination
                        );
                        StatusCode::OK.into_response()
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({ "error": format!("Copy failed: {}", e) })),
                    )
                        .into_response(),
                }
            } else {
                (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({ "error": format!("source model '{}' not found", request.source) })),
                )
                    .into_response()
            }
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Recursively copy a directory
fn copy_dir_recursive(src: &std::path::Path, dest: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dest_path)?;
        } else {
            std::fs::copy(&src_path, &dest_path)?;
        }
    }
    Ok(())
}

/// POST /api/pull — Pull model from HuggingFace with streaming progress
pub async fn pull_model(Json(request): Json<PullRequest>) -> Response {
    use axum::body::Body;
    use futures::stream;
    use tokio::sync::mpsc;

    let (tx, rx) = mpsc::channel::<String>(32);
    let repo = request.name.clone();

    // Spawn download in background
    tokio::spawn(async move {
        let send_status = |tx: &mpsc::Sender<String>, status: &str| {
            let msg = serde_json::json!({ "status": status });
            let _ = tx.blocking_send(format!("{}\n", msg));
        };

        let api = match hf_hub::api::sync::Api::new() {
            Ok(api) => api,
            Err(e) => {
                let msg = serde_json::json!({ "error": format!("Failed to init HF API: {}", e) });
                let _ = tx.blocking_send(format!("{}\n", msg));
                return;
            }
        };

        let repo_api = api.model(repo.clone());

        let model_name = repo.split('/').next_back().unwrap_or(&repo);
        let model_dir = std::env::var("BLAZR_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("./models"));
        let dest = model_dir.join(model_name);
        let _ = std::fs::create_dir_all(&dest);

        let files_to_try = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
        ];

        for filename in files_to_try {
            send_status(&tx, &format!("pulling {}", filename));
            if let Ok(cached) = repo_api.get(filename) {
                let _ = std::fs::copy(&cached, dest.join(filename));
                send_status(&tx, &format!("downloaded {}", filename));
            }
        }

        // Try sharded safetensors
        if let Ok(cached) = repo_api.get("model.safetensors.index.json") {
            let index_dest = dest.join("model.safetensors.index.json");
            let _ = std::fs::copy(&cached, &index_dest);
            if let Ok(content) = std::fs::read_to_string(&index_dest) {
                if let Ok(index) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                        let mut shards: Vec<String> = weight_map
                            .values()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                        shards.sort();
                        shards.dedup();
                        for shard in &shards {
                            send_status(&tx, &format!("pulling {}", shard));
                            if let Ok(cached) = repo_api.get(shard) {
                                let _ = std::fs::copy(&cached, dest.join(shard));
                                send_status(&tx, &format!("downloaded {}", shard));
                            }
                        }
                    }
                }
            }
        }

        send_status(&tx, "success");
    });

    // Stream status messages as NDJSON
    let body_stream = stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|msg| (Ok::<_, std::convert::Infallible>(msg), rx))
    });

    let body = Body::from_stream(body_stream);
    Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
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

#[derive(Deserialize)]
pub struct DeleteRequest {
    pub name: String,
}

#[derive(Deserialize)]
pub struct CopyRequest {
    pub source: String,
    pub destination: String,
}

#[derive(Deserialize)]
pub struct PullRequest {
    pub name: String,
}
