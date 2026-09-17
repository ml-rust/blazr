//! Shared types and filesystem helpers used across the management endpoints.

use std::path::Path;

use serde::Serialize;

/// Calculate directory size in bytes
pub(super) fn dir_size(path: &Path) -> u64 {
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

#[derive(Serialize)]
pub struct ModelDetails {
    pub format: String,
    pub family: String,
    pub parameter_size: String,
    pub quantization_level: String,
}
