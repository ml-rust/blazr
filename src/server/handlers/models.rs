//! `GET /v1/models` and `GET /v1/models/{id}` — OpenAI-compatible model
//! listing, merging text models (from the scheduler's model directory) with
//! resident auxiliary models (vision/ASR/TTS, registered from
//! `--vision-model`/`--asr-model`/`--tts-model`).

use std::sync::Arc;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

use super::auxiliary::AuxiliaryModel;
use super::state::AppState;
use crate::server::generation::error_response;

/// List available models
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.scheduler.list_available() {
        Ok(text_models) => {
            let auxiliary = state.auxiliary_models().await;
            let ids = merged_model_ids(
                text_models.into_iter().map(|m| m.name).collect(),
                &auxiliary,
            );
            let response = ModelsResponse {
                object: "list".to_string(),
                data: ids.into_iter().map(to_model_info).collect(),
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
        Ok(text_models) => {
            if let Some(m) = text_models.iter().find(|m| m.name == model_id) {
                return (StatusCode::OK, Json(to_model_info(m.name.clone()))).into_response();
            }
            let auxiliary = state.auxiliary_models().await;
            if auxiliary.iter().any(|m| m.name == model_id) {
                return (StatusCode::OK, Json(to_model_info(model_id))).into_response();
            }
            error_response(
                StatusCode::NOT_FOUND,
                &format!("Model '{}' not found", model_id),
                "invalid_request_error",
            )
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &e.to_string(),
            "server_error",
        ),
    }
}

/// Build the merged id list: every text model, then every auxiliary model
/// whose name is not already a text model (or an earlier auxiliary entry).
/// Text models always win a name collision.
fn merged_model_ids(mut text: Vec<String>, auxiliary: &[AuxiliaryModel]) -> Vec<String> {
    let mut seen: std::collections::HashSet<String> = text.iter().cloned().collect();
    for aux in auxiliary {
        if seen.insert(aux.name.clone()) {
            text.push(aux.name.clone());
        }
    }
    text
}

fn to_model_info(id: String) -> ModelInfo {
    ModelInfo {
        id,
        object: "model".to_string(),
        created: 0,
        owned_by: "local".to_string(),
    }
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

#[cfg(test)]
mod tests {
    use super::super::auxiliary::AuxiliaryKind;
    use super::*;

    fn aux(name: &str, kind: AuxiliaryKind) -> AuxiliaryModel {
        AuxiliaryModel {
            name: name.to_string(),
            kind,
            path: None,
        }
    }

    #[test]
    fn text_models_come_first() {
        let ids = merged_model_ids(vec!["llama".to_string()], &[aux("vox", AuxiliaryKind::Tts)]);
        assert_eq!(ids, vec!["llama".to_string(), "vox".to_string()]);
    }

    #[test]
    fn auxiliary_models_are_appended() {
        let ids = merged_model_ids(
            vec!["llama".to_string()],
            &[
                aux("vox", AuxiliaryKind::Tts),
                aux("tiny", AuxiliaryKind::Asr),
            ],
        );
        assert_eq!(ids, vec!["llama", "vox", "tiny"]);
    }

    #[test]
    fn duplicate_name_is_not_repeated() {
        let ids = merged_model_ids(
            vec!["shared".to_string()],
            &[aux("shared", AuxiliaryKind::Vision)],
        );
        assert_eq!(ids, vec!["shared".to_string()]);
    }

    #[test]
    fn empty_auxiliary_list_leaves_text_models_untouched() {
        let ids = merged_model_ids(vec!["llama".to_string(), "mistral".to_string()], &[]);
        assert_eq!(ids, vec!["llama".to_string(), "mistral".to_string()]);
    }
}
