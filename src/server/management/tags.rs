//! `GET /api/tags` — local text models plus resident vision/ASR/TTS models.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

use super::common::{dir_size, ModelDetails};
use crate::engine::AvailableModelInfo;
use crate::server::handlers::{AppState, AuxiliaryModel};

/// GET /api/tags — List local models with metadata
pub async fn list_tags(State(state): State<Arc<AppState>>) -> Response {
    match state.scheduler.list_available() {
        Ok(models) => {
            let mut model_list: Vec<ModelTag> = models.iter().map(text_model_tag).collect();
            let auxiliary = state.auxiliary_models().await;
            model_list.extend(auxiliary.iter().map(auxiliary_model_tag));

            (StatusCode::OK, Json(TagsResponse { models: model_list })).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Build a [`ModelTag`] for a text model from the scheduler's directory scan.
fn text_model_tag(m: &AvailableModelInfo) -> ModelTag {
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
}

/// Build a [`ModelTag`] for a resident vision/ASR/TTS model. `family` carries
/// its kind (`"tts"`/`"asr"`/`"vision"`); `size`/`modified_at` come from its
/// source directory when one is known, else the Ollama-parity zero values.
fn auxiliary_model_tag(m: &AuxiliaryModel) -> ModelTag {
    let size = m.path.as_deref().map(dir_size).unwrap_or(0);
    let modified_at = m
        .path
        .as_deref()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|meta| meta.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|d| chrono::DateTime::from_timestamp(d.as_secs() as i64, 0))
        .map(|dt| dt.to_rfc3339())
        .unwrap_or_default();

    ModelTag {
        name: m.name.clone(),
        model: m.name.clone(),
        modified_at,
        size,
        details: ModelDetails {
            format: String::new(),
            family: m.kind.as_str().to_string(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        },
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::handlers::AuxiliaryKind;
    use std::path::PathBuf;

    #[test]
    fn auxiliary_model_tag_carries_kind_as_family() {
        let m = AuxiliaryModel {
            name: "vox".to_string(),
            kind: AuxiliaryKind::Tts,
            path: None,
        };
        let tag = auxiliary_model_tag(&m);
        assert_eq!(tag.name, "vox");
        assert_eq!(tag.model, "vox");
        assert_eq!(tag.details.family, "tts");
        assert_eq!(tag.size, 0);
    }

    #[test]
    fn auxiliary_model_tag_sizes_from_a_real_path() {
        let dir = std::env::temp_dir().join("blazr-management-test-model");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("weights.bin"), b"0123456789").unwrap();

        let m = AuxiliaryModel {
            name: "tiny".to_string(),
            kind: AuxiliaryKind::Asr,
            path: Some(dir.clone()),
        };
        let tag = auxiliary_model_tag(&m);
        assert_eq!(tag.details.family, "asr");
        assert_eq!(tag.size, 10);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn vision_family_string_is_vision() {
        let m = AuxiliaryModel {
            name: "clip".to_string(),
            kind: AuxiliaryKind::Vision,
            path: None::<PathBuf>,
        };
        assert_eq!(auxiliary_model_tag(&m).details.family, "vision");
    }
}
