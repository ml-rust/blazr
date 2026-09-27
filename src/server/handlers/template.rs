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
use crate::model::chat_template::{ChatMessage, RenderOptions};
use crate::server::generation::error_response;
use crate::server::tools::{request_msg_to_chat_msg, ChatRequestMessage, Tool};

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

    let template = match request.template.as_deref() {
        Some(name) => match crate::server::chat::named_template(name) {
            Ok(template) => template,
            Err(message) => {
                return error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
            }
        },
        None => executor.chat_template().clone(),
    };
    let native_tools = template.renders_tools_natively();
    let msgs: Vec<ChatMessage> = request
        .messages
        .iter()
        .map(|m| request_msg_to_chat_msg(m, native_tools, None))
        .collect();
    let mut options = RenderOptions {
        add_generation_prompt: request.add_generation_prompt,
        ..RenderOptions::default()
    };
    if native_tools {
        options.tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .filter_map(|t| serde_json::to_value(t).ok())
                .collect()
        });
    }
    if let Some(kwargs) = request.chat_template_kwargs {
        options.extra.extend(kwargs);
    }
    match template.apply_with(&msgs, &options) {
        Ok(prompt) => (StatusCode::OK, Json(ApplyTemplateResponse { prompt })).into_response(),
        Err(e) => error_response(
            StatusCode::BAD_REQUEST,
            &format!("Rejected input: {}", e),
            "invalid_request_error",
        ),
    }
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
pub struct ApplyTemplateRequest {
    pub model: String,
    pub messages: Vec<ChatRequestMessage>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// Open the assistant turn at the end. Default: true.
    #[serde(default = "default_true")]
    pub add_generation_prompt: bool,
    /// Extra template variables, passed through to a Jinja template as is.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Serialize)]
pub struct ApplyTemplateResponse {
    pub prompt: String,
}
