//! `POST /api/show` — model config and template details.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use super::common::ModelDetails;
use crate::server::handlers::{classify_lookup, AppState};

/// POST /api/show — Show model details
pub async fn show_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ShowRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.name).await {
        Ok(e) => e,
        Err(e) => {
            let failure = classify_lookup(&e);
            return (
                failure.status,
                Json(serde_json::json!({ "error": failure.message })),
            )
                .into_response();
        }
    };

    let config = executor.config();
    let template_name = executor.chat_template().describe();

    let response = ShowResponse {
        modelfile: String::new(),
        parameters: format!(
            "num_ctx {}\ntemperature 1.0\ntop_p 1.0",
            config.inference.max_context_len.unwrap_or(4096)
        ),
        template: template_name,
        details: ModelDetails {
            format: String::new(),
            family: config.model_type().to_string(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        },
        model_info: serde_json::json!({
            "model_type": config.model_type(),
            "vocab_size": config.vocab_size(),
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "max_seq_len": config.model.max_seq_len,
        }),
    };

    (StatusCode::OK, Json(response)).into_response()
}

#[derive(Deserialize)]
pub struct ShowRequest {
    pub name: String,
}

#[derive(Serialize)]
pub struct ShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: ModelDetails,
    pub model_info: serde_json::Value,
}
