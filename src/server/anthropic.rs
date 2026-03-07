//! Anthropic Messages API compatibility layer (`POST /v1/messages`, `/v1/messages/count_tokens`)
//!
//! Translates Anthropic-format requests into the internal chat completion pipeline.
//! Supports text content blocks, system prompts, streaming, and stop sequences.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::generation::{
    generate_via_scheduler, overloaded_response, record_generation_metrics,
    stream_with_stop_sequences, validate_generation_params, HasSamplingFields, SamplingParams,
};
use super::handlers::AppState;
use super::metrics;
use super::streaming::{create_chat_stream, StreamToken};
use crate::model::chat_template::ChatMessage;

/// Anthropic Messages API request
#[derive(Deserialize)]
pub struct AnthropicRequest {
    /// Model name
    pub model: String,
    /// Max tokens to generate (required in Anthropic API)
    pub max_tokens: usize,
    /// Message array
    pub messages: Vec<AnthropicMessage>,
    /// System prompt (string or array of content blocks)
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    /// Stop sequences
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to stream
    #[serde(default)]
    pub stream: Option<bool>,
    /// Temperature (0.0 - 1.0)
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Metadata
    #[serde(default)]
    #[allow(dead_code)]
    pub metadata: Option<serde_json::Value>,
}

/// Anthropic system can be a string or array of content blocks
#[derive(Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

impl AnthropicSystem {
    fn text(&self) -> String {
        match self {
            AnthropicSystem::Text(s) => s.clone(),
            AnthropicSystem::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    AnthropicContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// Anthropic message
#[derive(Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    /// Content: string or array of content blocks
    pub content: AnthropicContent,
}

/// Content can be a string or array of content blocks
#[derive(Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

impl AnthropicContent {
    fn text(&self) -> String {
        match self {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    AnthropicContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// Anthropic content block
#[derive(Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        #[allow(dead_code)]
        source: serde_json::Value,
    },
}

impl HasSamplingFields for AnthropicRequest {
    fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            max_tokens: Some(self.max_tokens),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: self.stop_sequences.clone(),
            seed: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            json_mode: false,
            mirostat_mode: None,
            mirostat_tau: None,
            mirostat_eta: None,
            dynatemp_range: None,
            dynatemp_exponent: None,
            dry_multiplier: None,
            dry_base: None,
            dry_allowed_length: None,
            dry_sequence_breakers: None,
            typical_p: None,
            grammar: None,
            lora_adapter: None,
        }
    }
}

/// Anthropic Messages API response
#[derive(Serialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub r#type: &'static str,
    pub role: &'static str,
    pub content: Vec<AnthropicResponseBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Serialize)]
pub struct AnthropicResponseBlock {
    pub r#type: &'static str,
    pub text: String,
}

#[derive(Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Anthropic token count response
#[derive(Serialize)]
pub struct TokenCountResponse {
    pub input_tokens: usize,
}

/// POST /v1/messages — Anthropic Messages API
pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AnthropicRequest>,
) -> Response {
    if let Err(e) =
        validate_generation_params(request.temperature, request.top_p, Some(request.max_tokens))
    {
        return anthropic_error(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    if request.messages.is_empty() {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "messages array must not be empty",
            "invalid_request_error",
        );
    }

    let load_start = Instant::now();
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return anthropic_error(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "not_found_error",
            );
        }
    };
    let _load_duration_ms = load_start.elapsed().as_millis() as u64;

    // Build chat messages
    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(ref sys) = request.system {
        msgs.push(ChatMessage {
            role: "system".to_string(),
            content: sys.text(),
        });
    }
    for m in &request.messages {
        msgs.push(ChatMessage {
            role: m.role.clone(),
            content: m.content.text(),
        });
    }

    let prompt = executor.chat_template().apply(&msgs);
    let gen_config = request.sampling_params().into_gen_config();

    // Token budget admission
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
        let id = format!("msg_{}", uuid::Uuid::new_v4());
        let model_name = request.model;
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamToken>(32);
        let state_clone = Arc::clone(&state);
        let budget = estimated_tokens;

        metrics::adjust_decode_slots(1.0);
        tokio::spawn(async move {
            stream_with_stop_sequences(executor, prompt, gen_config, tx).await;
            state_clone.release(budget);
            metrics::adjust_decode_slots(-1.0);
        });

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        create_chat_stream(id, model_name, Box::pin(rx_stream)).into_response()
    } else {
        metrics::adjust_decode_slots(1.0);
        let start = Instant::now();
        let model_name = request.model.clone();

        let gen_result = if let Some(ref rs) = state.request_scheduler {
            match generate_via_scheduler(rs, &executor, &prompt, &gen_config).await {
                Ok(r) => Ok(crate::engine::GenerationResult {
                    text: r.text,
                    prompt_tokens: r.prompt_tokens,
                    completion_tokens: r.completion_tokens,
                    finish_reason: r.finish_reason,
                    prompt_eval_duration_ms: r.prompt_eval_duration_ms,
                    token_logprobs: None,
                }),
                Err(e) => Err(anyhow::anyhow!(e)),
            }
        } else {
            executor.generate_text(&prompt, &gen_config).await
        };

        match gen_result {
            Ok(result) => {
                let elapsed = start.elapsed();
                let stop_reason = match result.finish_reason {
                    crate::engine::FinishReason::Stop => Some("end_turn".to_string()),
                    crate::engine::FinishReason::Length => Some("max_tokens".to_string()),
                    crate::engine::FinishReason::Eos => Some("end_turn".to_string()),
                };

                record_generation_metrics(
                    &model_name,
                    &None,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.prompt_eval_duration_ms,
                    elapsed,
                );
                state.release(estimated_tokens);
                metrics::adjust_decode_slots(-1.0);

                let response = AnthropicResponse {
                    id: format!("msg_{}", uuid::Uuid::new_v4()),
                    r#type: "message",
                    role: "assistant",
                    content: vec![AnthropicResponseBlock {
                        r#type: "text",
                        text: result.text,
                    }],
                    model: request.model,
                    stop_reason,
                    stop_sequence: None,
                    usage: AnthropicUsage {
                        input_tokens: result.prompt_tokens,
                        output_tokens: result.completion_tokens,
                    },
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                state.release(estimated_tokens);
                metrics::adjust_decode_slots(-1.0);
                anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "api_error",
                )
            }
        }
    }
}

/// POST /v1/messages/count_tokens — Token counting
pub async fn count_tokens(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AnthropicRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return anthropic_error(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "not_found_error",
            );
        }
    };

    // Build the prompt the same way as messages endpoint
    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(ref sys) = request.system {
        msgs.push(ChatMessage {
            role: "system".to_string(),
            content: sys.text(),
        });
    }
    for m in &request.messages {
        msgs.push(ChatMessage {
            role: m.role.clone(),
            content: m.content.text(),
        });
    }

    let prompt = executor.chat_template().apply(&msgs);
    let token_count = executor
        .tokenizer()
        .encode(&prompt)
        .map(|t| t.len())
        .unwrap_or(prompt.len() / 4);

    let response = TokenCountResponse {
        input_tokens: token_count,
    };
    (StatusCode::OK, Json(response)).into_response()
}

/// Anthropic-style error response
fn anthropic_error(status: StatusCode, message: &str, error_type: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_content_text() {
        let json = r#""hello world""#;
        let content: AnthropicContent = serde_json::from_str(json).unwrap();
        assert_eq!(content.text(), "hello world");
    }

    #[test]
    fn test_anthropic_content_blocks() {
        let json = r#"[{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]"#;
        let content: AnthropicContent = serde_json::from_str(json).unwrap();
        assert_eq!(content.text(), "hello world");
    }

    #[test]
    fn test_anthropic_system_text() {
        let json = r#""You are helpful""#;
        let sys: AnthropicSystem = serde_json::from_str(json).unwrap();
        assert_eq!(sys.text(), "You are helpful");
    }

    #[test]
    fn test_anthropic_system_blocks() {
        let json = r#"[{"type": "text", "text": "You are helpful"}]"#;
        let sys: AnthropicSystem = serde_json::from_str(json).unwrap();
        assert_eq!(sys.text(), "You are helpful");
    }

    #[test]
    fn test_anthropic_request_deserialization() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].content.text(), "Hello");
    }

    #[test]
    fn test_anthropic_request_with_system() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "system": "Be concise",
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system.unwrap().text(), "Be concise");
    }

    #[test]
    fn test_anthropic_request_with_content_blocks() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": " World"}]}
            ]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages[0].content.text(), "Hello World");
    }

    #[test]
    fn test_anthropic_response_serialization() {
        let resp = AnthropicResponse {
            id: "msg_123".to_string(),
            r#type: "message",
            role: "assistant",
            content: vec![AnthropicResponseBlock {
                r#type: "text",
                text: "Hello!".to_string(),
            }],
            model: "test".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["usage"]["input_tokens"], 10);
    }
}
