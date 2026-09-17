//! `DELETE /api/delete` — remove a local text model from disk.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;

use crate::server::handlers::AppState;

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

#[derive(Deserialize)]
pub struct DeleteRequest {
    pub name: String,
}
