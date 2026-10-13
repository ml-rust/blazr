//! `GET /v1/models` and `GET /v1/models/{id}` — OpenAI-compatible model
//! listing, merging text models (from the scheduler's model directory) with
//! resident auxiliary models (vision/ASR/TTS, registered from
//! `--vision-model`/`--asr-model`/`--tts-model`). A decision model's entry
//! carries a `decision` object.

use std::sync::Arc;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

use super::auxiliary::AuxiliaryModel;
use super::state::AppState;
use crate::loader::decision::DecisionInfo;
use crate::server::generation::error_response;

/// Every text model with its decision report, read on a blocking thread:
/// the listing reads the model directory, sidecars and GGUF headers.
async fn text_models(state: &AppState) -> anyhow::Result<Vec<(String, Option<DecisionInfo>)>> {
    let scheduler = Arc::clone(&state.scheduler);
    tokio::task::spawn_blocking(
        move || -> anyhow::Result<Vec<(String, Option<DecisionInfo>)>> {
            let models = scheduler.list_available()?;
            Ok(models
                .into_iter()
                .map(|m| {
                    let decision = match scheduler.decision_info(&m.name, &m.path) {
                        Ok(info) => info.map(|i| (*i).clone()),
                        Err(e) => {
                            tracing::warn!("model '{}': no decision report: {e}", m.name);
                            None
                        }
                    };
                    (m.name, decision)
                })
                .collect())
        },
    )
    .await
    .map_err(|e| anyhow::anyhow!("the model listing stopped: {e}"))?
}

/// List available models
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match text_models(&state).await {
        Ok(text_models) => {
            let auxiliary = state.auxiliary_models().await;
            let ids = merged_model_ids(
                text_models.iter().map(|(name, _)| name.clone()).collect(),
                &auxiliary,
            );
            let response = ModelsResponse {
                object: "list".to_string(),
                data: ids
                    .into_iter()
                    .map(|id| {
                        let decision = decision_of(&text_models, &id);
                        to_model_info(id, decision)
                    })
                    .collect(),
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
    match text_models(&state).await {
        Ok(text_models) => {
            if text_models.iter().any(|(name, _)| *name == model_id) {
                let decision = decision_of(&text_models, &model_id);
                return (StatusCode::OK, Json(to_model_info(model_id, decision))).into_response();
            }
            let auxiliary = state.auxiliary_models().await;
            if auxiliary.iter().any(|m| m.name == model_id) {
                return (StatusCode::OK, Json(to_model_info(model_id, None))).into_response();
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

/// The decision report of the text model `id`.
fn decision_of(text_models: &[(String, Option<DecisionInfo>)], id: &str) -> Option<DecisionInfo> {
    text_models
        .iter()
        .find(|(name, _)| name == id)
        .and_then(|(_, decision)| decision.clone())
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

fn to_model_info(id: String, decision: Option<DecisionInfo>) -> ModelInfo {
    ModelInfo {
        id,
        object: "model".to_string(),
        created: 0,
        owned_by: "local".to_string(),
        decision,
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
    /// A System One decision model's family, limits, contract and
    /// calibration, read without loading its weights.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<DecisionInfo>,
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
