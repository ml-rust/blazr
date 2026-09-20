//! Anthropic Messages API compatibility layer (`POST /v1/messages`, `/v1/messages/count_tokens`)
//!
//! Translates Anthropic-format requests into the internal chat completion pipeline.
//! Supports text content blocks, system prompts, streaming, and stop sequences.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::anthropic_types::{
    response_blocks, AnthropicRequest, AnthropicResponse, AnthropicUsage, TokenCountResponse,
};
use super::generation::{
    generate_via_scheduler, overloaded_response, record_generation_metrics,
    stream_with_stop_sequences, validate_generation_params, HasSamplingFields,
};
use super::handlers::AppState;
use super::metrics;
use super::reasoning::ReasoningParser;
use super::streaming::{create_chat_stream, StreamToken};
use crate::model::chat_template::{encode_chat_prompt, ChatMessage, RenderOptions};

/// POST /v1/messages — Anthropic Messages API
pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AnthropicRequest>,
) -> Response {
    if let Err(e) =
        validate_generation_params(request.temperature, request.top_p, Some(request.max_tokens))
    {
        return anthropic_error(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    if request.messages.is_empty() {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "messages array must not be empty",
            "invalid_request_error",
        );
    }

    let load_start = Instant::now();
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return anthropic_error(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "not_found_error",
            );
        }
    };
    let _load_duration_ms = load_start.elapsed().as_millis() as u64;

    // Build chat messages
    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(ref sys) = request.system {
        msgs.push(ChatMessage::new("system", sys.text()));
    }
    for m in &request.messages {
        msgs.push(ChatMessage::new(m.role.clone(), m.content.text()));
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
        Err(e) => return anthropic_prompt_error(e),
    };
    let gen_config = request.sampling_params().into_gen_config();
    let reasoning_format = state.reasoning_format;

    // Token budget admission
    let estimated_tokens = prompt_tokens.len() + gen_config.max_tokens;
    if !state.try_admit(estimated_tokens) {
        return overloaded_response();
    }

    if request.stream.unwrap_or(false) {
        let id = format!("msg_{}", uuid::Uuid::new_v4());
        let model_name = request.model;
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamToken>(32);
        let state_clone = Arc::clone(&state);
        let budget = estimated_tokens;

        metrics::adjust_decode_slots(1.0);
        tokio::spawn(async move {
            stream_with_stop_sequences(executor, prompt_tokens, gen_config, tx).await;
            state_clone.release(budget);
            metrics::adjust_decode_slots(-1.0);
        });

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let parser = ReasoningParser::for_output(reasoning_format, &prompt);
        create_chat_stream(id, model_name, Box::pin(rx_stream), parser).into_response()
    } else {
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
                let stop_reason = match result.finish_reason {
                    crate::engine::FinishReason::Stop => Some("end_turn".to_string()),
                    crate::engine::FinishReason::Length => Some("max_tokens".to_string()),
                    crate::engine::FinishReason::Eos => Some("end_turn".to_string()),
                };

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

                let response = AnthropicResponse {
                    id: format!("msg_{}", uuid::Uuid::new_v4()),
                    r#type: "message",
                    role: "assistant",
                    content: response_blocks(&prompt, result.text, reasoning_format),
                    model: request.model,
                    stop_reason,
                    stop_sequence: None,
                    usage: AnthropicUsage {
                        input_tokens: result.prompt_tokens,
                        output_tokens: result.completion_tokens,
                    },
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                state.release(estimated_tokens);
                metrics::adjust_decode_slots(-1.0);
                anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "api_error",
                )
            }
        }
    }
}

/// POST /v1/messages/count_tokens — Token counting
pub async fn count_tokens(
    State(state): State<Arc<AppState>>,
    Json(request): Json<AnthropicRequest>,
) -> Response {
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return anthropic_error(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "not_found_error",
            );
        }
    };

    // Build the prompt the same way as messages endpoint
    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(ref sys) = request.system {
        msgs.push(ChatMessage::new("system", sys.text()));
    }
    for m in &request.messages {
        msgs.push(ChatMessage::new(m.role.clone(), m.content.text()));
    }

    let (_prompt, prompt_tokens) = match encode_chat_prompt(
        executor.chat_template(),
        executor.tokenizer(),
        &msgs,
        "",
        &RenderOptions::default(),
    ) {
        Ok(encoded) => encoded,
        Err(e) => return anthropic_prompt_error(e),
    };

    let response = TokenCountResponse {
        input_tokens: prompt_tokens.len(),
    };
    (StatusCode::OK, Json(response)).into_response()
}

/// The 400 response for a chat prompt that did not assemble: a template that
/// refused the conversation, or text spelling a control token.
fn anthropic_prompt_error(err: crate::model::chat_template::ChatPromptError) -> Response {
    anthropic_error(
        StatusCode::BAD_REQUEST,
        &format!("Rejected input: {}", err),
        "invalid_request_error",
    )
}

/// Anthropic-style error response
fn anthropic_error(status: StatusCode, message: &str, error_type: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        })),
    )
        .into_response()
}
