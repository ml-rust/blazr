//! OpenAI Responses API (`POST /v1/responses`)
//!
//! The newer OpenAI API format (used by O1/O3 models). Supports input items
//! (messages), reasoning output, and structured output via text format.
//! This translates to the internal chat completion pipeline.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::generation::{
    error_response, generate_via_scheduler, overloaded_response, record_generation_metrics,
    validate_generation_params,
};
use super::handlers::AppState;
use super::metrics;
use crate::model::chat_template::ChatMessage;

/// OpenAI Responses API request
#[derive(Deserialize)]
pub struct ResponsesRequest {
    /// Model name
    pub model: String,
    /// Input: string prompt or array of input items
    pub input: ResponsesInput,
    /// Max output tokens
    #[serde(default = "default_max_tokens")]
    pub max_output_tokens: usize,
    /// Temperature
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Whether to include reasoning/thinking in output
    #[serde(default)]
    #[allow(dead_code)]
    pub reasoning: Option<ReasoningConfig>,
    /// Instructions (system prompt)
    #[serde(default)]
    pub instructions: Option<String>,
    /// Text response format
    #[serde(default)]
    #[allow(dead_code)]
    pub text: Option<TextFormat>,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// Metadata
    #[serde(default)]
    #[allow(dead_code)]
    pub metadata: Option<serde_json::Value>,
}

fn default_max_tokens() -> usize {
    4096
}

/// Input can be a string or array of input items
#[derive(Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

/// An input item (message)
#[derive(Deserialize)]
pub struct InputItem {
    /// Role: "user", "assistant", "system", "developer"
    pub role: String,
    /// Content: string or array of content parts
    pub content: ResponsesContent,
}

/// Content: string or array of content parts
#[derive(Deserialize)]
#[serde(untagged)]
pub enum ResponsesContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

impl ResponsesContent {
    fn text(&self) -> String {
        match self {
            ResponsesContent::Text(s) => s.clone(),
            ResponsesContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ResponsesContentPart::InputText { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// Content part types
#[derive(Deserialize)]
#[serde(tag = "type")]
#[allow(clippy::enum_variant_names)]
pub enum ResponsesContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        #[allow(dead_code)]
        image_url: Option<String>,
    },
    #[serde(rename = "input_audio")]
    InputAudio {
        #[allow(dead_code)]
        data: Option<String>,
    },
}

/// Reasoning configuration
#[derive(Deserialize)]
pub struct ReasoningConfig {
    /// Effort level: "low", "medium", "high"
    #[serde(default = "default_effort")]
    #[allow(dead_code)]
    pub effort: String,
}

fn default_effort() -> String {
    "medium".to_string()
}

/// Text format configuration
#[derive(Deserialize)]
pub struct TextFormat {
    /// "text" or "json_object" or "json_schema"
    #[serde(default = "default_text_format")]
    #[allow(dead_code)]
    pub format: TextFormatType,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum TextFormatType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema {
        #[allow(dead_code)]
        schema: Option<serde_json::Value>,
    },
}

fn default_text_format() -> TextFormatType {
    TextFormatType::Text
}

/// OpenAI Responses API response
#[derive(Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub model: String,
    pub output: Vec<OutputItem>,
    pub usage: ResponsesUsage,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct OutputItem {
    pub r#type: &'static str,
    pub id: String,
    pub role: &'static str,
    pub content: Vec<OutputContent>,
    pub status: &'static str,
}

#[derive(Serialize)]
pub struct OutputContent {
    pub r#type: &'static str,
    pub text: String,
}

#[derive(Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

/// POST /v1/responses — OpenAI Responses API
pub async fn responses(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    if let Err(e) = validate_generation_params(
        request.temperature,
        request.top_p,
        Some(request.max_output_tokens),
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

    // Build chat messages from input
    let mut msgs: Vec<ChatMessage> = Vec::new();

    // Add instructions as system message
    if let Some(ref instructions) = request.instructions {
        msgs.push(ChatMessage {
            role: "system".to_string(),
            content: instructions.clone(),
        });
    }

    match &request.input {
        ResponsesInput::Text(text) => {
            msgs.push(ChatMessage {
                role: "user".to_string(),
                content: text.clone(),
            });
        }
        ResponsesInput::Items(items) => {
            for item in items {
                // Map "developer" role to "system"
                let role = if item.role == "developer" {
                    "system".to_string()
                } else {
                    item.role.clone()
                };
                msgs.push(ChatMessage {
                    role,
                    content: item.content.text(),
                });
            }
        }
    }

    let prompt = executor.chat_template().apply(&msgs);

    let gen_config = crate::config::GenerationConfig {
        max_tokens: request.max_output_tokens,
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        stop_sequences: request.stop.unwrap_or_default(),
        ..Default::default()
    };

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

            let response = ResponsesResponse {
                id: format!("resp_{}", uuid::Uuid::new_v4()),
                object: "response",
                created_at: chrono::Utc::now().timestamp(),
                model: request.model,
                output: vec![OutputItem {
                    r#type: "message",
                    id: format!("msg_{}", uuid::Uuid::new_v4()),
                    role: "assistant",
                    content: vec![OutputContent {
                        r#type: "output_text",
                        text: result.text,
                    }],
                    status: "completed",
                }],
                usage: ResponsesUsage {
                    input_tokens: result.prompt_tokens,
                    output_tokens: result.completion_tokens,
                    total_tokens: result.prompt_tokens + result.completion_tokens,
                },
                status: "completed",
                error: None,
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            state.release(estimated_tokens);
            metrics::adjust_decode_slots(-1.0);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_responses_input_text() {
        let json = r#"{"model": "test", "input": "Hello"}"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("expected text input"),
        }
    }

    #[test]
    fn test_responses_input_items() {
        let json = r#"{
            "model": "test",
            "input": [
                {"role": "user", "content": "Hello"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Items(items) => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].role, "user");
                assert_eq!(items[0].content.text(), "Hello");
            }
            _ => panic!("expected items input"),
        }
    }

    #[test]
    fn test_responses_input_content_parts() {
        let json = r#"{
            "model": "test",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hello "}, {"type": "input_text", "text": "World"}]}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Items(items) => {
                assert_eq!(items[0].content.text(), "Hello World");
            }
            _ => panic!("expected items input"),
        }
    }

    #[test]
    fn test_responses_with_instructions() {
        let json = r#"{
            "model": "test",
            "input": "Hi",
            "instructions": "You are helpful",
            "max_output_tokens": 100
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.instructions.unwrap(), "You are helpful");
        assert_eq!(req.max_output_tokens, 100);
    }

    #[test]
    fn test_responses_response_serialization() {
        let resp = ResponsesResponse {
            id: "resp_123".to_string(),
            object: "response",
            created_at: 1234567890,
            model: "test".to_string(),
            output: vec![OutputItem {
                r#type: "message",
                id: "msg_123".to_string(),
                role: "assistant",
                content: vec![OutputContent {
                    r#type: "output_text",
                    text: "Hello!".to_string(),
                }],
                status: "completed",
            }],
            usage: ResponsesUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
            },
            status: "completed",
            error: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "response");
        assert_eq!(json["output"][0]["type"], "message");
        assert_eq!(json["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(json["usage"]["total_tokens"], 15);
        assert_eq!(json["status"], "completed");
    }

    #[test]
    fn test_developer_role_mapping() {
        let json = r#"{
            "model": "test",
            "input": [
                {"role": "developer", "content": "System instructions"},
                {"role": "user", "content": "Hi"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Items(items) => {
                assert_eq!(items[0].role, "developer");
                assert_eq!(items[1].role, "user");
            }
            _ => panic!("expected items"),
        }
    }
}
