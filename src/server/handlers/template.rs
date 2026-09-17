//! `POST /apply-template` — render a chat template without running inference.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use super::state::AppState;
use crate::server::generation::error_response;
use crate::server::tools::{request_msg_to_chat_msg, ChatRequestMessage};

/// Apply chat template to messages without running inference (like llama.cpp's /apply-template)
pub async fn apply_template(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ApplyTemplateRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "invalid_request_error",
            );
        }
    };

    let msgs: Vec<crate::model::chat_template::ChatMessage> = request
        .messages
        .iter()
        .map(request_msg_to_chat_msg)
        .collect();
    let template = if let Some(ref tpl_name) = request.template {
        crate::model::chat_template::ChatTemplate::from_name(tpl_name)
    } else {
        executor.chat_template().clone()
    };
    let prompt = template.apply(&msgs);
    let response = ApplyTemplateResponse { prompt };
    (StatusCode::OK, Json(response)).into_response()
}

#[derive(Deserialize)]
pub struct ApplyTemplateRequest {
    pub model: String,
    pub messages: Vec<ChatRequestMessage>,
    #[serde(default)]
    pub template: Option<String>,
}

#[derive(Serialize)]
pub struct ApplyTemplateResponse {
    pub prompt: String,
}
