//! `POST /tokenize` and `POST /detokenize`.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use super::state::AppState;
use super::variant::lookup_error_response;
use crate::server::generation::{
    encode_with_allowed_special, error_response, policy_error_response,
};

/// Tokenize text endpoint
pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TokenizeRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => return lookup_error_response(&e),
    };

    // Untrusted text with no chat template around it: nothing in it becomes a
    // control token unless the caller explicitly asks for that token by name.
    let tokens = match encode_with_allowed_special(
        executor.tokenizer(),
        &request.content,
        request.allowed_special.as_deref(),
    ) {
        Ok(tokens) => tokens,
        Err(e) => return policy_error_response(&e),
    };
    let response = TokenizeResponse {
        tokens: tokens.iter().map(|&t| t as i64).collect(),
    };
    (StatusCode::OK, Json(response)).into_response()
}

/// Detokenize token IDs endpoint
pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DetokenizeRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => return lookup_error_response(&e),
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

#[derive(Deserialize)]
pub struct TokenizeRequest {
    pub model: String,
    pub content: String,
    /// Special tokens the caller permits `content` to spell out, tiktoken-style.
    ///
    /// Absent (the default) means none: the text is encoded as ordinary
    /// content, so a `content` of `"<|im_start|>"` yields the tokens of that
    /// literal string rather than the real control-token id. Naming a token
    /// here opts that one token back in; any *other* special token in the text
    /// is then refused rather than silently promoted.
    #[serde(default)]
    pub allowed_special: Option<Vec<String>>,
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
