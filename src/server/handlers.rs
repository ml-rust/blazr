//! HTTP request handlers

use std::sync::Arc;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use super::streaming::{create_chat_stream, create_completion_stream};
use crate::config::GenerationConfig;
use crate::engine::Scheduler;

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

/// Shared application state
pub struct AppState {
    pub scheduler: Arc<Scheduler<ServerRuntime>>,
}

impl AppState {
    pub fn new(scheduler: Arc<Scheduler<ServerRuntime>>) -> Self {
        Self { scheduler }
    }
}

/// Health check endpoint
pub async fn health() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

/// List available models
pub async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.scheduler.list_available() {
        Ok(models) => {
            let response = ModelsResponse {
                object: "list".to_string(),
                data: models
                    .iter()
                    .map(|m| ModelInfo {
                        id: m.name.clone(),
                        object: "model".to_string(),
                        created: 0,
                        owned_by: "local".to_string(),
                    })
                    .collect(),
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// Text completion endpoint
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    // Get executor for model
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Model not found: {}", e),
                        r#type: "invalid_request_error".to_string(),
                    },
                }),
            )
                .into_response();
        }
    };

    // Build generation config
    let gen_config = GenerationConfig {
        max_tokens: request.max_tokens.unwrap_or(256),
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        stop_sequences: request.stop.clone().unwrap_or_default(),
        ..Default::default()
    };

    if request.stream.unwrap_or(false) {
        // Streaming response
        let stream = executor.generate(&request.prompt, &gen_config);
        let token_stream = stream.filter_map(|r| async move { r.ok().map(|t| t.text) });

        // Collect into iterator (simplified for now)
        let tokens: Vec<String> = token_stream.collect().await;
        let id = format!("cmpl-{}", uuid::Uuid::new_v4());

        create_completion_stream(id, request.model, tokens.into_iter()).into_response()
    } else {
        // Non-streaming response
        match executor.generate_text(&request.prompt, &gen_config).await {
            Ok(text) => {
                let response = CompletionResponse {
                    id: format!("cmpl-{}", uuid::Uuid::new_v4()),
                    object: "text_completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: request.model,
                    choices: vec![CompletionChoice {
                        text,
                        index: 0,
                        finish_reason: "stop".to_string(),
                    }],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: e.to_string(),
                        r#type: "server_error".to_string(),
                    },
                }),
            )
                .into_response(),
        }
    }
}

/// Chat completion endpoint
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Response {
    // Get executor for model
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Model not found: {}", e),
                        r#type: "invalid_request_error".to_string(),
                    },
                }),
            )
                .into_response();
        }
    };

    // Format messages into prompt
    let prompt = format_chat_messages(&request.messages);

    // Build generation config
    let gen_config = GenerationConfig {
        max_tokens: request.max_tokens.unwrap_or(256),
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        stop_sequences: request.stop.clone().unwrap_or_default(),
        ..Default::default()
    };

    if request.stream.unwrap_or(false) {
        // Streaming response
        let stream = executor.generate(&prompt, &gen_config);
        let token_stream = stream.filter_map(|r| async move { r.ok().map(|t| t.text) });

        let tokens: Vec<String> = token_stream.collect().await;
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        create_chat_stream(id, request.model, tokens.into_iter()).into_response()
    } else {
        // Non-streaming response
        match executor.generate_text(&prompt, &gen_config).await {
            Ok(text) => {
                let response = ChatResponse {
                    id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: request.model,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: text,
                        },
                        finish_reason: "stop".to_string(),
                    }],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: e.to_string(),
                        r#type: "server_error".to_string(),
                    },
                }),
            )
                .into_response(),
        }
    }
}

/// Format chat messages into a prompt string
fn format_chat_messages(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                prompt.push_str(&format!("<|system|>\n{}\n", msg.content));
            }
            "user" => {
                prompt.push_str(&format!("<|user|>\n{}\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<|assistant|>\n{}\n", msg.content));
            }
            _ => {
                prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
            }
        }
    }

    prompt.push_str("<|assistant|>\n");
    prompt
}

// Request/Response types

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
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

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
}
