//! OpenAI Responses API (`POST /v1/responses`)
//!
//! The newer OpenAI API format (used by O1/O3 models). Supports input items
//! (messages), reasoning output, and structured output via text format.
//! This translates to the internal chat completion pipeline.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::chat_vision::{decode_image_sources, image_route, marked_text, ImageRoute};
use super::generation::{
    chat_prompt_error_response, error_response, overloaded_response, record_generation_metrics,
    validate_generation_params,
};
use super::handlers::{select_executor, AppState};
use super::metrics;
use super::responses_types::{
    output_items, ResponsesInput, ResponsesRequest, ResponsesResponse, ResponsesUsage,
};
use super::scheduled::generate_text_routed;
use crate::model::chat_template::{encode_chat_prompt_with_markers, ChatMessage, RenderOptions};

/// POST /v1/responses — OpenAI Responses API
pub async fn responses(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    if let Err(e) = validate_generation_params(
        request.temperature,
        request.top_p,
        request.max_output_tokens,
    ) {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    let executor =
        match select_executor(&state, &request.model, request.lora_adapter.as_deref()).await {
            Ok(e) => e,
            Err(response) => return *response,
        };

    // Decode the `input_image` parts, then pick the path that serves them:
    // the vision tower, the model's own encoders, or a 400.
    let media = {
        let sources = match request.input.image_sources() {
            Ok(sources) => sources,
            Err(message) => {
                return error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
            }
        };
        // `sources` borrows the request, so it ends here: the handler moves
        // request fields into the reply below.
        match decode_image_sources(&sources).await {
            Ok(media) => media,
            Err(message) => {
                return error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
            }
        }
    };
    let multimodal_model = matches!(executor.model(), boostr::model::LoadedModel::Multimodal(_));
    let route = match image_route(executor.vision_tower().is_some(), multimodal_model, &media) {
        Ok(route) => route,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
        }
    };

    // Build chat messages from input
    let mut msgs: Vec<ChatMessage> = Vec::new();

    // Add instructions as system message
    if let Some(ref instructions) = request.instructions {
        msgs.push(ChatMessage::new("system", instructions.clone()));
    }

    match &request.input {
        ResponsesInput::Text(text) => {
            msgs.push(ChatMessage::new("user", text.clone()));
        }
        ResponsesInput::Items(items) => {
            for item in items {
                // Map "developer" role to "system"
                let role = if item.role == "developer" {
                    "system".to_string()
                } else {
                    item.role.clone()
                };
                let content = marked_text(&item.content.media_parts(), route.image_marker());
                msgs.push(ChatMessage::new(role, content));
            }
        }
    }

    // One encode of the assembled prompt, under the template's own allow-list
    // plus whatever markers this route wrote in: any other control token in
    // the caller's text is refused, not promoted.
    let (prompt, prompt_tokens) = match encode_chat_prompt_with_markers(
        executor.chat_template(),
        executor.tokenizer(),
        &msgs,
        "",
        &RenderOptions::default(),
        &route.server_markers(),
    ) {
        Ok(encoded) => encoded,
        Err(e) => return chat_prompt_error_response(&e),
    };
    // The marker allow-list let the caller's text spell a marker too: each
    // one must appear exactly once per image.
    if let (ImageRoute::Tower, Some(tower)) = (route, executor.vision_tower()) {
        if let Err(message) = tower.check_marker_counts(&prompt_tokens, media.images.len()) {
            return error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
        }
    }

    // Each value the request omits takes the live `generation` default.
    let defaults = state.config().generation.clone();
    let gen_config = crate::config::GenerationConfig {
        max_tokens: request.max_output_tokens.unwrap_or(defaults.max_tokens),
        temperature: request.temperature.unwrap_or(defaults.temperature),
        top_p: request.top_p.unwrap_or(defaults.top_p),
        stop_sequences: request.stop.unwrap_or(defaults.stop_sequences.clone()),
        ..defaults
    };

    // Token budget admission
    let estimated_tokens = prompt_tokens.len() + gen_config.max_tokens;
    if !state.try_admit(estimated_tokens) {
        return overloaded_response();
    }

    // The executor and scheduler this request runs on. With data
    // parallelism, the router picks a replica here.
    let placement = state.route(executor, &prompt_tokens);
    let executor = Arc::clone(&placement.executor);

    metrics::adjust_decode_slots(1.0);
    let start = Instant::now();
    let model_name = request.model.clone();

    // The media route picks the generator; text may go through the batching
    // scheduler.
    let gen_result = if route == ImageRoute::Tower {
        executor
            .generate_images_text(&prompt_tokens, &media.images, &gen_config)
            .await
    } else if route == ImageRoute::Multimodal {
        executor
            .generate_multimodal_text(&prompt_tokens, &media.images, &media.audio, &gen_config)
            .await
    } else {
        generate_text_routed(&placement, &prompt_tokens, &gen_config).await
    };

    match gen_result {
        Ok(result) => {
            let elapsed = start.elapsed();
            record_generation_metrics(
                &model_name,
                &None,
                result.prompt_tokens,
                result.completion_tokens,
                result.prompt_eval_duration_ms,
                elapsed,
            );
            state.release(estimated_tokens);
            metrics::adjust_decode_slots(-1.0);

            let response = ResponsesResponse {
                id: format!("resp_{}", uuid::Uuid::new_v4()),
                object: "response",
                created_at: chrono::Utc::now().timestamp(),
                model: request.model,
                output: output_items(&prompt, result.text, state.reasoning_format()),
                usage: ResponsesUsage {
                    input_tokens: result.prompt_tokens,
                    output_tokens: result.completion_tokens,
                    total_tokens: result.prompt_tokens + result.completion_tokens,
                },
                status: "completed",
                error: None,
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            state.release(estimated_tokens);
            metrics::adjust_decode_slots(-1.0);
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e.to_string(),
                "server_error",
            )
        }
    }
}
