//! `GET /api/ps` — currently loaded text models plus resident vision/ASR/TTS
//! models (always resident once registered, no separate load/unload cycle).

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Serialize;

use super::common::{dir_size, ModelDetails};
use crate::server::handlers::AppState;

/// GET /api/ps — List currently loaded models
pub async fn list_running(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let loaded = state.scheduler.list_loaded().await;
    let mut models: Vec<RunningModel> = loaded
        .iter()
        .map(|m| RunningModel {
            name: m.name.clone(),
            model: m.name.clone(),
            size: dir_size(&m.path),
            expires_at: String::new(),
            size_vram: 0,
            details: ModelDetails {
                format: String::new(),
                family: String::new(),
                parameter_size: String::new(),
                quantization_level: String::new(),
            },
        })
        .collect();

    let auxiliary = state.auxiliary_models().await;
    models.extend(auxiliary.iter().map(|m| RunningModel {
        name: m.name.clone(),
        model: m.name.clone(),
        size: m.path.as_deref().map(dir_size).unwrap_or(0),
        expires_at: String::new(),
        size_vram: 0,
        details: ModelDetails {
            format: String::new(),
            family: m.kind.as_str().to_string(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        },
    }));

    (StatusCode::OK, Json(RunningResponse { models }))
}

#[derive(Serialize)]
pub struct RunningResponse {
    pub models: Vec<RunningModel>,
}

#[derive(Serialize)]
pub struct RunningModel {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub expires_at: String,
    pub size_vram: u64,
    pub details: ModelDetails,
}
