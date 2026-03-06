//! LoRA adapter management HTTP handlers
//!
//! Endpoints:
//!   POST   /v1/lora/load      — Load an adapter from disk and register it with a model
//!   DELETE /v1/lora/{name}    — Unload a named adapter from a model
//!   GET    /v1/lora            — List all loaded adapters for a model

use std::sync::Arc;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::generation::error_response;
use super::handlers::AppState;

/// Request body for `POST /v1/lora/load`
#[derive(Deserialize)]
pub struct LoadLoraRequest {
    /// Name of the base model to attach this adapter to (same model identifier used in completions)
    pub model: String,
    /// Logical name for this adapter — used to reference it in generation requests
    pub name: String,
    /// File-system path to the adapter directory or `adapter_model.safetensors` file
    pub path: String,
}

/// Single adapter entry returned by the list endpoint
#[derive(Serialize)]
pub struct LoraAdapterEntry {
    pub name: String,
}

/// Response from `GET /v1/lora`
#[derive(Serialize)]
pub struct ListLorasResponse {
    pub model: String,
    pub adapters: Vec<LoraAdapterEntry>,
}

/// `POST /v1/lora/load`
///
/// Loads a HuggingFace PEFT-format LoRA adapter from the given path and attaches
/// it to the named model.  The model is loaded into the scheduler if not already running.
pub async fn load_lora(
    State(state): State<Arc<AppState>>,
    Json(request): Json<LoadLoraRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model '{}' not found: {}", request.model, e),
                "invalid_request_error",
            );
        }
    };

    let adapter_path = std::path::Path::new(&request.path);
    match executor.load_lora(adapter_path, &request.name) {
        Ok(()) => {
            tracing::info!(
                model = %request.model,
                adapter = %request.name,
                path = %request.path,
                "LoRA adapter loaded via API"
            );
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "ok",
                    "model": request.model,
                    "adapter": request.name,
                })),
            )
                .into_response()
        }
        Err(e) => error_response(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("Failed to load LoRA adapter '{}': {}", request.name, e),
            "lora_load_error",
        ),
    }
}

/// `DELETE /v1/lora/{name}?model=<model>`
///
/// Unloads the named adapter from the model.  The `model` query parameter selects
/// which model's registry to update.
pub async fn unload_lora(
    State(state): State<Arc<AppState>>,
    Path(adapter_name): Path<String>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let model_name = match params.get("model") {
        Some(m) => m.clone(),
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Missing required query parameter: model",
                "invalid_request_error",
            );
        }
    };

    let executor = match state.scheduler.get_executor(&model_name).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model '{}' not found: {}", model_name, e),
                "invalid_request_error",
            );
        }
    };

    if executor.unload_lora(&adapter_name) {
        tracing::info!(
            model = %model_name,
            adapter = %adapter_name,
            "LoRA adapter unloaded via API"
        );
        StatusCode::NO_CONTENT.into_response()
    } else {
        error_response(
            StatusCode::NOT_FOUND,
            &format!(
                "LoRA adapter '{}' not found for model '{}'",
                adapter_name, model_name
            ),
            "lora_not_found",
        )
    }
}

/// `GET /v1/lora?model=<model>`
///
/// Lists all LoRA adapters currently loaded for the given model.
pub async fn list_loras(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let model_name = match params.get("model") {
        Some(m) => m.clone(),
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Missing required query parameter: model",
                "invalid_request_error",
            );
        }
    };

    let executor = match state.scheduler.get_executor(&model_name).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model '{}' not found: {}", model_name, e),
                "invalid_request_error",
            );
        }
    };

    let adapters: Vec<LoraAdapterEntry> = executor
        .list_loras()
        .into_iter()
        .map(|name| LoraAdapterEntry { name })
        .collect();

    (
        StatusCode::OK,
        Json(ListLorasResponse {
            model: model_name,
            adapters,
        }),
    )
        .into_response()
}
