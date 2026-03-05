//! HTTP request handlers

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use tokio::sync::RwLock;

use super::metrics;
use super::streaming::{create_chat_stream, create_completion_stream, StreamToken};
use crate::config::{GenerationConfig, UserConfig};
use crate::engine::{parse_keep_alive, FinishReason, Scheduler};

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

/// Shared application state
pub struct AppState {
    pub scheduler: Arc<Scheduler<ServerRuntime>>,
    pub metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    pub user_config: Arc<RwLock<UserConfig>>,
    /// Current in-flight token count (prompt + estimated decode tokens)
    pub inflight_tokens: AtomicUsize,
    /// Maximum in-flight token budget (0 = unlimited)
    pub max_inflight_tokens: usize,
}

impl AppState {
    pub fn new(
        scheduler: Arc<Scheduler<ServerRuntime>>,
        metrics_handle: metrics_exporter_prometheus::PrometheusHandle,
    ) -> Self {
        Self {
            scheduler,
            metrics_handle,
            user_config: Arc::new(RwLock::new(UserConfig::load())),
            inflight_tokens: AtomicUsize::new(0),
            max_inflight_tokens: 0,
        }
    }

    pub fn with_max_inflight_tokens(mut self, max: usize) -> Self {
        self.max_inflight_tokens = max;
        self
    }

    /// Try to admit a request with the given token budget.
    /// Returns `false` (and does not increment) if the budget would be exceeded.
    pub fn try_admit(&self, tokens: usize) -> bool {
        if self.max_inflight_tokens == 0 {
            self.inflight_tokens.fetch_add(tokens, Ordering::Relaxed);
            metrics::adjust_inflight_tokens(tokens as f64);
            return true;
        }
        // CAS loop to atomically check and increment
        loop {
            let current = self.inflight_tokens.load(Ordering::Relaxed);
            if current + tokens > self.max_inflight_tokens {
                return false;
            }
            if self
                .inflight_tokens
                .compare_exchange_weak(
                    current,
                    current + tokens,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                metrics::adjust_inflight_tokens(tokens as f64);
                return true;
            }
        }
    }

    /// Release tokens back to the budget after a request completes
    pub fn release(&self, tokens: usize) {
        self.inflight_tokens.fetch_sub(tokens, Ordering::Relaxed);
        metrics::adjust_inflight_tokens(-(tokens as f64));
    }
}

/// Health check endpoint — returns status and loaded model info
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let loaded = state
        .scheduler
        .list_loaded()
        .await
        .into_iter()
        .map(|m| m.name)
        .collect::<Vec<_>>();

    let response = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        loaded_models: loaded,
    };
    (StatusCode::OK, Json(response)).into_response()
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub loaded_models: Vec<String>,
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
                                    error: None,
                                })
                                .await;
                        } else {
                            // Stop sequence was in previously sent tokens — just send final
                            let _ = tx
                                .send(StreamToken {
                                    text: String::new(),
                                    finish_reason: Some(FinishReason::Stop),
                                    error: None,
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
                        error: None,
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
                let _ = tx
                    .send(StreamToken {
                        text: String::new(),
                        finish_reason: None,
                        error: Some(e.to_string()),
                    })
                    .await;
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
    if let Err(e) =
        validate_generation_params(request.temperature, request.top_p, request.max_tokens)
    {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    let load_start = Instant::now();
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
    let load_duration_ms = load_start.elapsed().as_millis() as u64;

    // Prepend context tokens (from previous turn) to the prompt if provided
    let prompt = match decode_context_prefix(&executor, &request.context) {
        Ok(prefix) if prefix.is_empty() => request.prompt.clone(),
        Ok(prefix) => format!("{}{}", prefix, request.prompt),
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to decode context tokens: {}", e),
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

    // Token budget admission control
    let prompt_token_count = executor
        .tokenizer()
        .encode(&prompt)
        .map(|t| t.len())
        .unwrap_or(prompt.len() / 4);
    let estimated_tokens = prompt_token_count + gen_config.max_tokens;
    if !state.try_admit(estimated_tokens) {
        return overloaded_response();
    }

    if request.stream.unwrap_or(false) {
        let id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let model_name = request.model.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamToken>(32);
        let state_clone = Arc::clone(&state);
        let budget = estimated_tokens;

        tokio::spawn(async move {
            stream_with_stop_sequences(executor, prompt, gen_config, tx).await;
            state_clone.release(budget);
        });

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        create_completion_stream(id, model_name, Box::pin(rx_stream)).into_response()
    } else {
        let echo = request.echo.unwrap_or(false);
        let model_name = request.model.clone();
        let start = Instant::now();
        match executor.generate_text(&prompt, &gen_config).await {
            Ok(result) => {
                let elapsed = start.elapsed();
                let text = if echo {
                    format!("{}{}", prompt, result.text)
                } else {
                    result.text
                };
                let done_reason = result.finish_reason.as_str().to_string();
                let response = CompletionResponse {
                    id: format!("cmpl-{}", uuid::Uuid::new_v4()),
                    object: "text_completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: request.model,
                    choices: vec![CompletionChoice {
                        text,
                        index: 0,
                        finish_reason: done_reason.clone(),
                    }],
                    usage: Usage {
                        prompt_tokens: result.prompt_tokens,
                        completion_tokens: result.completion_tokens,
                        total_tokens: result.prompt_tokens + result.completion_tokens,
                        total_duration_ms: elapsed.as_millis() as u64,
                        prompt_eval_duration_ms: result.prompt_eval_duration_ms,
                        load_duration_ms,
                        tokens_per_second: if elapsed.as_secs_f64() > 0.0 {
                            result.completion_tokens as f64 / elapsed.as_secs_f64()
                        } else {
                            0.0
                        },
                    },
                    done_reason: Some(done_reason),
                };
                record_generation_metrics(
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.prompt_eval_duration_ms,
                    elapsed,
                );
                apply_keep_alive(&state, &model_name, &request.keep_alive).await;
                state.release(estimated_tokens);
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                state.release(estimated_tokens);
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "server_error",
                )
            }
        }
    }
}

/// Chat completion endpoint
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Response {
    if let Err(e) =
        validate_generation_params(request.temperature, request.top_p, request.max_tokens)
    {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    if request.messages.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "messages array must not be empty",
            "invalid_request_error",
        );
    }

    let load_start = Instant::now();
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
    let load_duration_ms = load_start.elapsed().as_millis() as u64;

    // Prepend context tokens (from previous turn) if provided
    let context_prefix = match decode_context_prefix(&executor, &request.context) {
        Ok(prefix) => prefix,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to decode context tokens: {}", e),
                "invalid_request_error",
            );
        }
    };

    let prompt =
        if request.raw.unwrap_or(false) {
            // Raw mode: concatenate message contents without template
            request
                .messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            let mut msgs: Vec<crate::model::chat_template::ChatMessage> = Vec::new();
            // Prepend system override if provided and no system message exists
            if let Some(ref sys) = request.system {
                let has_system = request.messages.iter().any(|m| m.role == "system");
                if !has_system {
                    msgs.push(crate::model::chat_template::ChatMessage {
                        role: "system".to_string(),
                        content: sys.clone(),
                    });
                }
            }
            msgs.extend(request.messages.iter().map(|m| {
                crate::model::chat_template::ChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                }
            }));
            // Use per-request template override if provided, otherwise model's template
            if let Some(ref tpl_name) = request.template {
                crate::model::chat_template::ChatTemplate::from_name(tpl_name).apply(&msgs)
            } else {
                executor.chat_template().apply(&msgs)
            }
        };
    let prompt = if context_prefix.is_empty() {
        prompt
    } else {
        format!("{}{}", context_prefix, prompt)
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

    // Token budget admission control
    let prompt_token_count = executor
        .tokenizer()
        .encode(&prompt)
        .map(|t| t.len())
        .unwrap_or(prompt.len() / 4);
    let estimated_tokens = prompt_token_count + gen_config.max_tokens;
    if !state.try_admit(estimated_tokens) {
        return overloaded_response();
    }

    if request.stream.unwrap_or(false) {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let model_name = request.model;
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamToken>(32);
        let state_clone = Arc::clone(&state);
        let budget = estimated_tokens;

        tokio::spawn(async move {
            stream_with_stop_sequences(executor, prompt, gen_config, tx).await;
            state_clone.release(budget);
        });

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        create_chat_stream(id, model_name, Box::pin(rx_stream)).into_response()
    } else {
        let start = Instant::now();
        let think_enabled = request.think.unwrap_or(false);
        let model_name = request.model.clone();
        match executor.generate_text(&prompt, &gen_config).await {
            Ok(result) => {
                let elapsed = start.elapsed();
                let done_reason = result.finish_reason.as_str().to_string();

                let (content, thinking) = if think_enabled {
                    let extracted = crate::model::think::extract_thinking(&result.text);
                    let thinking = if extracted.thinking.is_empty() {
                        None
                    } else {
                        Some(extracted.thinking.join("\n\n"))
                    };
                    (extracted.content, thinking)
                } else {
                    (result.text, None)
                };

                let response = ChatResponse {
                    id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                    object: "chat.completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: request.model,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content,
                        },
                        finish_reason: done_reason.clone(),
                    }],
                    thinking,
                    usage: Usage {
                        prompt_tokens: result.prompt_tokens,
                        completion_tokens: result.completion_tokens,
                        total_tokens: result.prompt_tokens + result.completion_tokens,
                        total_duration_ms: elapsed.as_millis() as u64,
                        prompt_eval_duration_ms: result.prompt_eval_duration_ms,
                        load_duration_ms,
                        tokens_per_second: if elapsed.as_secs_f64() > 0.0 {
                            result.completion_tokens as f64 / elapsed.as_secs_f64()
                        } else {
                            0.0
                        },
                    },
                    done_reason: Some(done_reason),
                };
                record_generation_metrics(
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.prompt_eval_duration_ms,
                    elapsed,
                );
                apply_keep_alive(&state, &model_name, &request.keep_alive).await;
                state.release(estimated_tokens);
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                state.release(estimated_tokens);
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "server_error",
                )
            }
        }
    }
}

/// Decode context tokens (from a previous turn) into a string prefix.
/// Returns an empty string if no context tokens are provided.
/// Record generation metrics (tokens, TTFT, throughput) after a non-streaming request.
fn record_generation_metrics(
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt_eval_ms: u64,
    elapsed: std::time::Duration,
) {
    metrics::record_tokens(prompt_tokens, completion_tokens);
    metrics::record_ttft(prompt_eval_ms as f64 / 1000.0);
    metrics::record_tokens_per_second(if elapsed.as_secs_f64() > 0.0 {
        completion_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    });
}

fn decode_context_prefix(
    executor: &std::sync::Arc<crate::engine::Executor<ServerRuntime>>,
    context: &Option<Vec<i64>>,
) -> Result<String, String> {
    if let Some(ref ctx_tokens) = context {
        let token_ids: Vec<u32> = ctx_tokens.iter().map(|&t| t as u32).collect();
        executor
            .tokenizer()
            .decode(&token_ids)
            .map_err(|e| e.to_string())
    } else {
        Ok(String::new())
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
            return Err(format!("temperature must be between 0 and 2, got {}", t));
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

/// Return a 503 Service Unavailable response with Retry-After header
fn overloaded_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [("retry-after", "5")],
        axum::Json(serde_json::json!({
            "error": {
                "message": "Server overloaded: in-flight token budget exceeded. Retry after a short delay.",
                "type": "server_overloaded"
            }
        })),
    )
        .into_response()
}

/// Apply per-request keep_alive to the model's scheduler entry
async fn apply_keep_alive(state: &AppState, model_name: &str, keep_alive: &Option<String>) {
    if let Some(ref ka) = keep_alive {
        let duration = parse_keep_alive(ka);
        state.scheduler.set_keep_alive(model_name, duration).await;
    }
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
    pub echo: Option<bool>,
    /// Conversational context: token IDs from a previous response to prepend (Ollama-compatible)
    #[serde(default)]
    pub context: Option<Vec<i64>>,
    /// Model keep-alive duration (e.g., "5m", "1h", "0" to unload immediately, "-1" for forever)
    #[serde(default)]
    pub keep_alive: Option<String>,
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
    /// Ollama-compatible done_reason field
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
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
    /// Bypass chat template and send raw prompt (Ollama-compatible)
    #[serde(default)]
    pub raw: Option<bool>,
    /// Override system prompt (Ollama-compatible, prepended as system message)
    #[serde(default)]
    pub system: Option<String>,
    /// Override chat template (e.g., "llama3", "chatml", "mistral", "phi3", "gemma", "deepseek")
    #[serde(default)]
    pub template: Option<String>,
    /// Conversational context: token IDs from a previous response to prepend (Ollama-compatible)
    #[serde(default)]
    pub context: Option<Vec<i64>>,
    /// Enable think mode: extract `<think>` blocks from reasoning models (DeepSeek-R1, QwQ)
    #[serde(default)]
    pub think: Option<bool>,
    /// Model keep-alive duration (e.g., "5m", "1h", "0" to unload immediately, "-1" for forever).
    /// Overrides the server default for this request.
    #[serde(default)]
    pub keep_alive: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub user: Option<String>,
}

pub use crate::model::chat_template::ChatMessage;

#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    /// Ollama-compatible done_reason field
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    /// Extracted thinking content from reasoning models (when `think: true`)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
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
    /// Total request duration in milliseconds
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub total_duration_ms: u64,
    /// Prefill duration in milliseconds (time to first token)
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub prompt_eval_duration_ms: u64,
    /// Model load duration in milliseconds (0 if already loaded)
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub load_duration_ms: u64,
    /// Decode tokens per second
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub tokens_per_second: f64,
}

fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
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
