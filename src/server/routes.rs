//! Route definitions

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};

use super::handlers::{
    chat_completions, completions, detokenize, get_model, health, list_models, tokenize, AppState,
};

/// Create the API router with OpenAI-compatible endpoints
pub fn api_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Health check
        .route("/health", get(health))
        // OpenAI-compatible endpoints
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model_id}", get(get_model))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        // Tokenization endpoints
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
}
