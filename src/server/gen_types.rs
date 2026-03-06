//! Shared response types and utility functions for generation endpoints.
//!
//! Contains OpenAI-compatible response structs, logprob types, error helpers,
//! and validation utilities used across completion, chat, and infill handlers.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::handlers::AppState;
use crate::engine::{parse_keep_alive, GeneratedToken};

/// OpenAI response_format parameter
#[derive(Deserialize, Default, Clone)]
pub struct ResponseFormat {
    /// "text" (default), "json_object", or "json_schema"
    #[serde(default = "default_response_format_type")]
    pub r#type: String,
    /// JSON schema for structured output (used when type = "json_schema")
    #[serde(default)]
    pub json_schema: Option<serde_json::Value>,
}

fn default_response_format_type() -> String {
    "text".to_string()
}

/// Validate generation parameters, return error message if invalid
pub fn validate_generation_params(
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
pub fn overloaded_response() -> Response {
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

/// Convert executor token logprobs into API response format
pub fn convert_logprobs(tokens: &[GeneratedToken]) -> LogprobResult {
    LogprobResult {
        content: tokens
            .iter()
            .map(|t| LogprobEntry {
                token: t.text.clone(),
                logprob: t.logprob.unwrap_or(f32::NEG_INFINITY),
                top_logprobs: t
                    .top_logprobs
                    .as_ref()
                    .map(|tops| {
                        tops.iter()
                            .map(|tl| TopLogprob {
                                token: tl.text.clone(),
                                logprob: tl.logprob,
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            })
            .collect(),
    }
}

/// Apply per-request keep_alive to the model's scheduler entry
pub async fn apply_keep_alive(state: &AppState, model_name: &str, keep_alive: &Option<String>) {
    if let Some(ref ka) = keep_alive {
        let duration = parse_keep_alive(ka);
        state.scheduler.set_keep_alive(model_name, duration).await;
    }
}

/// Build a standard OpenAI error response
pub fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    (
        status,
        axum::Json(ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                r#type: error_type.to_string(),
            },
        }),
    )
        .into_response()
}

// ── Shared response types ──

/// OpenAI-compatible logprobs result
#[derive(Serialize, Clone)]
pub struct LogprobResult {
    pub content: Vec<LogprobEntry>,
}

/// Log probability for a single token position
#[derive(Serialize, Clone)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub top_logprobs: Vec<TopLogprob>,
}

/// A single alternative token with its log probability
#[derive(Serialize, Clone)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub total_duration_ms: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub prompt_eval_duration_ms: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub load_duration_ms: u64,
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
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
}
