//! `POST /api/copy` — copy/alias a local text model under a new name.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;

use crate::server::handlers::AppState;
use crate::server::model_name::resolve;

/// The path of the copy of `src`, beside `src`. A file keeps its extension, so
/// `x/a.gguf` to `foo` is the listed `x/foo.gguf`.
///
/// `None` when `destination` is anything but one plain name, because such a
/// destination leaves the directory of the source.
fn copy_destination(src: &Path, destination: &str) -> Option<PathBuf> {
    let mut components = Path::new(destination).components();
    let plain = matches!(components.next(), Some(Component::Normal(_)));
    if !plain || components.next().is_some() {
        return None;
    }
    let named = src.parent()?.join(destination);
    if src.is_dir() {
        return Some(named);
    }
    let Some(extension) = src.extension() else {
        return Some(named);
    };
    if named.extension() == Some(extension) {
        return Some(named);
    }
    let mut with_extension = named.into_os_string();
    with_extension.push(".");
    with_extension.push(extension);
    Some(with_extension.into())
}

/// POST /api/copy — Copy/alias a model
pub async fn copy_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CopyRequest>,
) -> Response {
    let source = resolve(&state, &request.source).await;
    match state.scheduler.list_available() {
        Ok(models) => {
            if let Some(m) = models.iter().find(|m| m.name == source) {
                let src = &m.path;
                let Some(dest) = copy_destination(src, &request.destination) else {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "error": format!(
                                "destination '{}' is not one plain name. Send a name with no \
                                 directory, such as 'my-model'",
                                request.destination
                            )
                        })),
                    )
                        .into_response();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::handlers::test_state;

    async fn copy(state: &Arc<AppState>, source: &str, destination: &str) -> StatusCode {
        let request = Json(CopyRequest {
            source: source.to_string(),
            destination: destination.to_string(),
        });
        copy_model(State(Arc::clone(state)), request).await.status()
    }

    #[tokio::test]
    async fn a_destination_outside_the_source_directory_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().join("models");
        std::fs::create_dir(&root).expect("mkdir");
        std::fs::write(root.join("a.gguf"), b"").expect("write");
        let state = Arc::new(test_state(&root));
        let outside = dir.path().join("escaped");
        let absolute = outside.display().to_string();

        for destination in [
            absolute.as_str(),
            "../escaped",
            "../../escaped",
            "sub/escaped",
            "..",
            ".",
            "",
        ] {
            assert_eq!(
                copy(&state, "a.gguf", destination).await,
                StatusCode::BAD_REQUEST,
                "{destination:?}"
            );
        }
        assert!(!outside.exists());
        assert!(!dir.path().join("escaped.gguf").exists());
    }

    /// A copy keeps the source extension, so the copy is listed and loads.
    #[tokio::test]
    async fn a_copy_of_a_file_keeps_its_extension_and_is_listed() {
        let dir = tempfile::tempdir().expect("temp dir");
        let x = dir.path().join("x");
        std::fs::create_dir(&x).expect("mkdir");
        std::fs::write(x.join("a.gguf"), b"a").expect("write");
        std::fs::write(x.join("b.gguf"), b"b").expect("write");
        let state = Arc::new(test_state(dir.path()));

        assert_eq!(copy(&state, "x/a.gguf", "foo").await, StatusCode::OK);
        assert!(x.join("foo.gguf").is_file());
        assert!(!x.join("foo").exists());
        assert_eq!(copy(&state, "x/a.gguf", "bar.gguf").await, StatusCode::OK);
        assert!(x.join("bar.gguf").is_file());
        assert_eq!(copy(&state, "x/a.gguf", "foo").await, StatusCode::CONFLICT);
        let names: Vec<String> = state
            .scheduler
            .list_available()
            .expect("lists")
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert!(names.contains(&"x/foo.gguf".to_string()), "{names:?}");
    }
}
