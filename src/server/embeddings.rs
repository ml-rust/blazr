//! Embeddings endpoint handler (`POST /v1/embeddings`)
//!
//! Supports OpenAI-compatible embedding generation with multiple pooling
//! strategies. Uses the model's token embeddings (from `forward_embed`)
//! with configurable pooling (mean, cls, last, none).

use std::sync::Arc;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::encoding::encode_f32_base64;
use super::generation::error_response;
use super::handlers::AppState;
use super::pooling::{l2_normalize, pool_cls, pool_last, pool_mean};

/// Embedding request (OpenAI-compatible)
#[derive(Deserialize)]
pub struct EmbeddingRequest {
    /// Model name
    pub model: String,
    /// Input text(s) to embed — single string or array of strings
    pub input: EmbeddingInput,
    /// Encoding format: "float" (default) or "base64"
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
    /// Pooling strategy: "mean" (default), "cls", "last", "none"
    #[serde(default = "default_pooling")]
    pub pooling: String,
    /// Whether to L2-normalize embeddings (default: false)
    #[serde(default)]
    pub normalize: bool,
    /// Optional user identifier
    #[serde(default)]
    #[allow(dead_code)]
    pub user: Option<String>,
}

fn default_encoding_format() -> String {
    "float".to_string()
}

fn default_pooling() -> String {
    "mean".to_string()
}

/// Input can be a single string or array of strings
#[derive(Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    fn texts(&self) -> Vec<&str> {
        match self {
            EmbeddingInput::Single(s) => vec![s.as_str()],
            EmbeddingInput::Batch(v) => v.iter().map(|s| s.as_str()).collect(),
        }
    }
}

/// Embedding response (OpenAI-compatible)
#[derive(Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: EmbeddingValue,
}

/// Embedding values — float array or base64-encoded
#[derive(Serialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

/// Embeddings endpoint
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
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

    // Validate pooling strategy
    let pooling = match request.pooling.as_str() {
        "mean" | "cls" | "last" | "none" => request.pooling.as_str(),
        other => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!(
                    "Invalid pooling strategy '{}'. Must be one of: mean, cls, last, none",
                    other
                ),
                "invalid_request_error",
            );
        }
    };

    let texts = request.input.texts();
    if texts.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Input must not be empty",
            "invalid_request_error",
        );
    }

    let mut embeddings_data = Vec::with_capacity(texts.len());
    let mut total_prompt_tokens = 0;

    for (i, text) in texts.iter().enumerate() {
        // Tokenize
        let token_ids = match executor.tokenizer().encode(text) {
            Ok(ids) => ids,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Tokenization failed: {}", e),
                    "server_error",
                );
            }
        };
        let num_tokens = token_ids.len();
        total_prompt_tokens += num_tokens;

        // Get hidden states from model's embedding layer
        let hidden = match executor.get_embeddings(&token_ids).await {
            Ok(h) => h,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Embedding generation failed: {}", e),
                    "server_error",
                );
            }
        };

        // hidden shape: [num_tokens, hidden_size]
        let hidden_size = if hidden.is_empty() {
            0
        } else {
            hidden.len() / num_tokens
        };

        // Apply pooling
        let mut embedding = match pooling {
            "mean" => pool_mean(&hidden, num_tokens, hidden_size),
            "cls" => pool_cls(&hidden, hidden_size),
            "last" => pool_last(&hidden, num_tokens, hidden_size),
            "none" => hidden,
            _ => unreachable!(),
        };

        // Optionally L2-normalize
        if request.normalize {
            l2_normalize(&mut embedding);
        }

        let value = if request.encoding_format == "base64" {
            EmbeddingValue::Base64(encode_f32_base64(&embedding))
        } else {
            EmbeddingValue::Float(embedding)
        };

        embeddings_data.push(EmbeddingData {
            object: "embedding",
            index: i,
            embedding: value,
        });
    }

    let response = EmbeddingResponse {
        object: "list",
        data: embeddings_data,
        model: request.model,
        usage: EmbeddingUsage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: total_prompt_tokens,
        },
    };

    (StatusCode::OK, Json(response)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_input_single() {
        let json = r#"{"model": "test", "input": "hello"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input.texts(), vec!["hello"]);
    }

    #[test]
    fn test_embedding_input_batch() {
        let json = r#"{"model": "test", "input": ["hello", "world"]}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input.texts(), vec!["hello", "world"]);
    }

    #[test]
    fn test_defaults() {
        let json = r#"{"model": "test", "input": "hello"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.encoding_format, "float");
        assert_eq!(req.pooling, "mean");
        assert!(!req.normalize);
    }
}
