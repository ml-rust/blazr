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

use super::generation::{
    chat_prompt_error_response, error_response, generate_via_scheduler, overloaded_response,
    record_generation_metrics, validate_generation_params,
};
use super::handlers::AppState;
use super::metrics;
use super::responses_types::{
    output_items, ResponsesInput, ResponsesRequest, ResponsesResponse, ResponsesUsage,
};
use crate::model::chat_template::{encode_chat_prompt, ChatMessage, RenderOptions};

/// POST /v1/responses — OpenAI Responses API
pub async fn responses(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    if let Err(e) = validate_generation_params(
        request.temperature,
        request.top_p,
        Some(request.max_output_tokens),
    ) {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "invalid_request_error",
            );
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
                msgs.push(ChatMessage::new(role, item.content.text()));
            }
        }
    }

    // One encode of the assembled prompt, under the template's own allow-list:
    // a control token the server did not insert is refused, not promoted.
    let (prompt, prompt_tokens) = match encode_chat_prompt(
        executor.chat_template(),
        executor.tokenizer(),
        &msgs,
        "",
        &RenderOptions::default(),
    ) {
        Ok(encoded) => encoded,
        Err(e) => return chat_prompt_error_response(&e),
    };

    let gen_config = crate::config::GenerationConfig {
        max_tokens: request.max_output_tokens,
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        stop_sequences: request.stop.unwrap_or_default(),
        ..Default::default()
    };

    // Token budget admission
    let estimated_tokens = prompt_tokens.len() + gen_config.max_tokens;
    if !state.try_admit(estimated_tokens) {
        return overloaded_response();
    }

    metrics::adjust_decode_slots(1.0);
    let start = Instant::now();
    let model_name = request.model.clone();

    let gen_result = if let Some(ref rs) = state.request_scheduler {
        match generate_via_scheduler(rs, prompt_tokens.clone(), &gen_config).await {
            Ok(r) => Ok(crate::engine::GenerationResult {
                text: r.text,
                prompt_tokens: r.prompt_tokens,
                completion_tokens: r.completion_tokens,
                finish_reason: r.finish_reason,
                prompt_eval_duration_ms: r.prompt_eval_duration_ms,
                token_logprobs: None,
            }),
            Err(e) => Err(anyhow::anyhow!(e)),
        }
    } else {
        executor.generate_text(&prompt_tokens, &gen_config).await
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
                output: output_items(&prompt, result.text, state.reasoning_format),
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
