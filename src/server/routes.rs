//! Route definitions

use std::sync::Arc;

use axum::{
    routing::{delete, get, post},
    Router,
};

use super::chat::chat_completions;
use super::completions::completions;
use super::handlers::{
    create_slot, delete_slot, detokenize, get_model, list_models, list_slots, tokenize, AppState,
};
use super::management::{
    copy_model, delete_model, list_running, list_tags, pull_model, show_model,
};

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
        // Slot management endpoints
        .route("/api/slots", get(list_slots).post(create_slot))
        .route("/api/slots/{slot_id}", delete(delete_slot))
        // Model management endpoints
        .route("/api/tags", get(list_tags))
        .route("/api/show", post(show_model))
        .route("/api/ps", get(list_running))
        .route("/api/delete", delete(delete_model))
        .route("/api/copy", post(copy_model))
        .route("/api/pull", post(pull_model))
}
