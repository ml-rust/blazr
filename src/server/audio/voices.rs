//! `GET /v1/audio/voices` — list a TTS model's voice catalog.

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::server::gen_types::error_response;
use crate::server::handlers::AppState;

#[derive(Debug, Deserialize)]
pub struct VoicesQuery {
    pub model: Option<String>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct VoiceEntry {
    pub id: String,
    pub name: String,
    pub language: String,
}

#[derive(Debug, Serialize)]
pub struct VoicesResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<VoiceEntry>,
}

/// `GET /v1/audio/voices?model=<name>` — the voice catalog of a loaded TTS
/// model, including a `zero-shot` entry when the bundle carries one
/// (VoxCPM2). 400 when `model` is missing or names a TTS model that is not
/// loaded — the same rule `/v1/audio/speech`'s bundle lookup applies.
pub async fn voices(
    State(state): State<Arc<AppState>>,
    Query(query): Query<VoicesQuery>,
) -> Response {
    let model = match query.model {
        Some(m) if !m.is_empty() => m,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Missing required query parameter 'model'",
                "invalid_request_error",
            );
        }
    };

    let bundle = match state.tts_model(&model).await {
        Some(b) => b,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!(
                    "TTS model '{}' is not loaded. Start the server with \
                     `--tts-model {}=PATH` to register it.",
                    model, model
                ),
                "invalid_request_error",
            );
        }
    };

    let data = bundle.voices().iter().map(voice_entry).collect();

    (
        StatusCode::OK,
        Json(VoicesResponse {
            object: "list".to_string(),
            model,
            data,
        }),
    )
        .into_response()
}

fn voice_entry(v: &voxr::Voice) -> VoiceEntry {
    VoiceEntry {
        id: v.id.clone(),
        name: v.display_name.clone(),
        language: v.lang.bcp47().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxr::{Lang, Voice};

    #[test]
    fn voice_entry_maps_id_name_and_language() {
        let v = Voice::new("af_alloy", Lang::EnUs, "alloy");
        let entry = voice_entry(&v);
        assert_eq!(
            entry,
            VoiceEntry {
                id: "af_alloy".to_string(),
                name: "alloy".to_string(),
                language: "en-us".to_string(),
            }
        );
    }

    #[test]
    fn voices_response_shape_serializes_as_expected() {
        let response = VoicesResponse {
            object: "list".to_string(),
            model: "vox".to_string(),
            data: vec![VoiceEntry {
                id: "zero-shot".to_string(),
                name: "zero-shot".to_string(),
                language: "ms".to_string(),
            }],
        };
        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["model"], "vox");
        assert_eq!(json["data"][0]["id"], "zero-shot");
        assert_eq!(json["data"][0]["name"], "zero-shot");
        assert_eq!(json["data"][0]["language"], "ms");
    }
}
