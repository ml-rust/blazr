//! HTTP request handlers

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use super::streaming::{create_chat_stream, create_completion_stream, StreamToken};
use crate::config::GenerationConfig;
use crate::engine::{FinishReason, Scheduler};

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

/// Get single model details
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Response {
    match state.scheduler.list_available() {
        Ok(models) => {
            if let Some(m) = models.iter().find(|m| m.name == model_id) {
                let info = ModelInfo {
                    id: m.name.clone(),
                    object: "model".to_string(),
                    created: 0,
                    owned_by: "local".to_string(),
                };
                (StatusCode::OK, Json(info)).into_response()
            } else {
                error_response(
                    StatusCode::NOT_FOUND,
                    &format!("Model '{}' not found", model_id),
                    "invalid_request_error",
                )
            }
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &e.to_string(),
            "server_error",
        ),
    }
}

/// Tokenize text endpoint
pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenizeRequest>,
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

    match executor.tokenizer().encode(&request.content) {
        Ok(tokens) => {
            let response = TokenizeResponse {
                tokens: tokens.iter().map(|&t| t as i64).collect(),
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Tokenization failed: {}", e),
            "server_error",
        ),
    }
}

/// Detokenize token IDs endpoint
pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetokenizeRequest>,
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

    let token_ids: Vec<u32> = request.tokens.iter().map(|&t| t as u32).collect();
    match executor.tokenizer().decode(&token_ids) {
        Ok(text) => {
            let response = DetokenizeResponse { content: text };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Detokenization failed: {}", e),
            "server_error",
        ),
    }
}

/// Sampling parameters shared between completion and chat requests
struct SamplingParams {
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    stop: Option<Vec<String>>,
    seed: Option<u64>,
}

impl SamplingParams {
    fn into_gen_config(self) -> GenerationConfig {
        GenerationConfig {
            max_tokens: self.max_tokens.unwrap_or(256),
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k.unwrap_or(0),
            min_p: self.min_p.unwrap_or(0.05),
            repeat_penalty: self.repeat_penalty.unwrap_or(1.1),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            stop_sequences: self.stop.unwrap_or_default(),
            seed: self.seed,
            ..Default::default()
        }
    }
}

/// Stream generation tokens with stop sequence checking.
///
/// Accumulates generated text and checks for stop sequences after each token.
/// When a stop sequence is found, sends a final token with the text before
/// the stop sequence and FinishReason::Stop.
async fn stream_with_stop_sequences(
    executor: Arc<crate::engine::Executor<ServerRuntime>>,
    prompt: String,
    gen_config: GenerationConfig,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) {
    let stop_sequences = gen_config.stop_sequences.clone();
    let stream = executor.generate(&prompt, &gen_config);
    let mut stream = std::pin::pin!(stream);
    let mut accumulated = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                accumulated.push_str(&token.text);

                // Check stop sequences
                let mut stopped = false;
                for stop in &stop_sequences {
                    if let Some(pos) = accumulated.find(stop.as_str()) {
                        // Found stop sequence — send only text before it
                        // We may have already sent some of this text in previous tokens.
                        // Since we stream token-by-token, the stop sequence detection
                        // only matters for the current token. If the stop sequence spans
                        // multiple tokens, some prefix was already sent.
                        // Truncate the current token's text to exclude the stop portion.
                        let already_sent = accumulated.len() - token.text.len();
                        if pos >= already_sent {
                            // Stop sequence starts within the current token
                            let safe_text = &token.text[..pos - already_sent];
                            let _ = tx
                                .send(StreamToken {
                                    text: safe_text.to_string(),
                                    finish_reason: Some(FinishReason::Stop),
                                })
                                .await;
                        } else {
                            // Stop sequence was in previously sent tokens — just send final
                            let _ = tx
                                .send(StreamToken {
                                    text: String::new(),
                                    finish_reason: Some(FinishReason::Stop),
                                })
                                .await;
                        }
                        stopped = true;
                        break;
                    }
                }

                if stopped {
                    break;
                }

                // No stop sequence — forward token
                let is_final = token.finish_reason.is_some();
                if tx
                    .send(StreamToken {
                        text: token.text,
                        finish_reason: token.finish_reason,
                    })
                    .await
                    .is_err()
                {
                    break; // client disconnected
                }
                if is_final {
                    break;
                }
            }
            Err(e) => {
                tracing::error!("Generation error during streaming: {}", e);
                break;
            }
        }
    }
}

/// Text completion endpoint
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    if let Err(e) = validate_generation_params(
        request.temperature,
        request.top_p,
        request.max_tokens,
    ) {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

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

    let gen_config = SamplingParams {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        min_p: request.min_p,
        repeat_penalty: request.repeat_penalty,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        stop: request.stop.clone(),
        seed: request.seed,
    }
    .into_gen_config();

    if request.stream.unwrap_or(false) {
        let id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let model_name = request.model;
        let prompt = request.prompt;
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamToken>(32);

        tokio::spawn(stream_with_stop_sequences(executor, prompt, gen_config, tx));

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        create_completion_stream(id, model_name, Box::pin(rx_stream)).into_response()
    } else {
        match executor.generate_text(&request.prompt, &gen_config).await {
            Ok(result) => {
                let response = CompletionResponse {
                    id: format!("cmpl-{}", uuid::Uuid::new_v4()),
                    object: "text_completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: request.model,
                    choices: vec![CompletionChoice {
                        text: result.text,
                        index: 0,
                        finish_reason: result.finish_reason.as_str().to_string(),
                    }],
                    usage: Usage {
                        prompt_tokens: result.prompt_tokens,
                        completion_tokens: result.completion_tokens,
                        total_tokens: result.prompt_tokens + result.completion_tokens,
                    },
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            ),
        }
    }
}

/// Chat completion endpoint
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Response {
    if let Err(e) = validate_generation_params(
        request.temperature,
        request.top_p,
        request.max_tokens,
    ) {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    if request.messages.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "messages array must not be empty",
            "invalid_request_error",
        );
    }

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

    let prompt = format_chat_messages(&request.messages);

    let gen_config = SamplingParams {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        min_p: request.min_p,
        repeat_penalty: request.repeat_penalty,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        stop: request.stop.clone(),
        seed: request.seed,
    }
    .into_gen_config();

    if request.stream.unwrap_or(false) {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let model_name = request.model;
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamToken>(32);

        tokio::spawn(stream_with_stop_sequences(executor, prompt, gen_config, tx));

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        create_chat_stream(id, model_name, Box::pin(rx_stream)).into_response()
    } else {
        match executor.generate_text(&prompt, &gen_config).await {
            Ok(result) => {
                let response = ChatResponse {
                    id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: request.model,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: result.text,
                        },
                        finish_reason: result.finish_reason.as_str().to_string(),
                    }],
                    usage: Usage {
                        prompt_tokens: result.prompt_tokens,
                        completion_tokens: result.completion_tokens,
                        total_tokens: result.prompt_tokens + result.completion_tokens,
                    },
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            ),
        }
    }
}

/// Validate generation parameters, return error message if invalid
fn validate_generation_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
) -> Result<(), String> {
    if let Some(t) = temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(format!(
                "temperature must be between 0 and 2, got {}",
                t
            ));
        }
    }
    if let Some(p) = top_p {
        if !(0.0..=1.0).contains(&p) {
            return Err(format!("top_p must be between 0 and 1, got {}", p));
        }
    }
    if let Some(m) = max_tokens {
        if m == 0 {
            return Err("max_tokens must be greater than 0".to_string());
        }
    }
    Ok(())
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

/// Build a standard OpenAI error response
fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                r#type: error_type.to_string(),
            },
        }),
    )
        .into_response()
}

// ── Request/Response types ──

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
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub user: Option<String>,
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
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub user: Option<String>,
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

#[derive(Deserialize)]
pub struct TokenizeRequest {
    pub model: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<i64>,
}

#[derive(Deserialize)]
pub struct DetokenizeRequest {
    pub model: String,
    pub tokens: Vec<i64>,
}

#[derive(Serialize)]
pub struct DetokenizeResponse {
    pub content: String,
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
