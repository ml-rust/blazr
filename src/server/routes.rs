//! Route definitions

use std::sync::Arc;

use axum::{
    routing::{delete, get, post},
    Router,
};

use super::anthropic::{count_tokens, messages};
use super::audio::{speech, transcriptions};
use super::chat::chat_completions;
use super::completions::completions;
use super::embeddings::embeddings;
use super::handlers::{
    apply_template, create_slot, delete_slot, detokenize, get_model, list_models, list_slots,
    tokenize, AppState,
};
use super::infill::infill;
use super::lora::{list_loras, load_lora, unload_lora};
use super::management::{
    copy_model, delete_model, list_running, list_tags, pull_model, show_model,
};
use super::rerank::rerank;
use super::responses::responses;

/// Create the API router with OpenAI-compatible endpoints (auth-protected)
pub fn api_routes() -> Router<Arc<AppState>> {
    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model_id}", get(get_model))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/responses", post(responses))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/infill", post(infill))
        .route("/v1/audio/speech", post(speech))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/rerank", post(rerank))
        .route("/v1/rerank", post(rerank))
        // Tokenization endpoints
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/apply-template", post(apply_template))
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
        // LoRA adapter management endpoints
        .route("/v1/lora", get(list_loras).post(load_lora))
        .route("/v1/lora/{name}", delete(unload_lora))
}
