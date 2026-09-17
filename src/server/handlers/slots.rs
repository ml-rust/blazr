//! `/api/slots` — inference slot lifecycle (create, list, free).

use std::sync::Arc;

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::state::AppState;
use crate::server::generation::error_response;

/// Create a new inference slot
pub async fn create_slot(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateSlotRequest>,
) -> Response {
    match state.slot_manager.allocate(&request.model).await {
        Ok(id) => {
            let response = SlotResponse {
                id,
                model: request.model,
                status: "active".to_string(),
            };
            (StatusCode::CREATED, Json(response)).into_response()
        }
        Err(e) => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &e,
            "slot_allocation_failed",
        ),
    }
}

/// List all active inference slots
pub async fn list_slots(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let slots = state.slot_manager.list().await;
    let response: Vec<SlotListEntry> = slots
        .into_iter()
        .map(|s| SlotListEntry {
            id: s.id,
            model: s.model,
            total_tokens: s.total_tokens,
            idle_seconds: s.last_accessed.elapsed().as_secs(),
        })
        .collect();
    (StatusCode::OK, Json(response)).into_response()
}

/// Free an inference slot
pub async fn delete_slot(
    State(state): State<Arc<AppState>>,
    Path(slot_id): Path<String>,
) -> Response {
    if state.slot_manager.free(&slot_id).await {
        StatusCode::NO_CONTENT.into_response()
    } else {
        error_response(
            StatusCode::NOT_FOUND,
            &format!("Slot '{}' not found", slot_id),
            "slot_not_found",
        )
    }
}

#[derive(Deserialize)]
pub struct CreateSlotRequest {
    pub model: String,
}

#[derive(Serialize)]
pub struct SlotResponse {
    pub id: String,
    pub model: String,
    pub status: String,
}

#[derive(Serialize)]
pub struct SlotListEntry {
    pub id: String,
    pub model: String,
    pub total_tokens: usize,
    pub idle_seconds: u64,
}
