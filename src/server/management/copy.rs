//! `POST /api/copy` — copy/alias a local text model under a new name.

use std::path::Path;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;

use crate::server::handlers::AppState;

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
fn copy_dir_recursive(src: &Path, dest: &Path) -> std::io::Result<()> {
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

#[derive(Deserialize)]
pub struct CopyRequest {
    pub source: String,
    pub destination: String,
}
