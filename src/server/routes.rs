//! Route definitions

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};

use super::handlers::{
    chat_completions, completions, detokenize, get_model, list_models, tokenize, AppState,
};
use super::management::{list_running, list_tags, show_model};

/// Create the API router with OpenAI-compatible endpoints (auth-protected)
pub fn api_routes() -> Router<Arc<AppState>> {
    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model_id}", get(get_model))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        // Tokenization endpoints
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        // Model management endpoints
        .route("/api/tags", get(list_tags))
        .route("/api/show", post(show_model))
        .route("/api/ps", get(list_running))
}
