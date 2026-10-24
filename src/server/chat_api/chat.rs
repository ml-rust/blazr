//! Chat completion endpoint handler

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::chat_core::{prepare_chat, PreparedChat};
use super::chat_sse::chat_sse;
use super::chat_types::{ChatChoice, ChatRequest, ChatResponse};
use super::route_stream::spawn_media_stream;
use super::tool_calls::extract_tool_calls;
use crate::model::chat_template::ChatTemplate;
use crate::server::generation::{
    apply_keep_alive, convert_logprobs, error_response, record_generation_metrics, Usage,
};
use crate::server::handlers::{AppState, DecodeSlot};
use crate::server::reasoning::{split_reasoning_for_format, ReasoningParser};

/// The built-in format a request's `template` field names, or the message
/// for a name blazr does not know.
pub fn named_template(name: &str) -> Result<ChatTemplate, String> {
    ChatTemplate::builtin(name).ok_or_else(|| {
        format!(
            "unknown chat template {name:?}; built-in names: {}",
            crate::model::chat_template::BUILTIN_NAMES.join(", ")
        )
    })
}

/// Chat completion endpoint
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Response {
    let prepared = match prepare_chat(&state, &request).await {
        Ok(prepared) => prepared,
        Err(response) => return *response,
    };
    if request.stream.unwrap_or(false) {
        stream_reply(&state, &request, prepared)
    } else {
        complete_reply(&state, &request, prepared).await
    }
}

/// The SSE reply of a streamed chat request.
fn stream_reply(
    state: &Arc<AppState>,
    request: &ChatRequest,
    prepared: PreparedChat<'_>,
) -> Response {
    let PreparedChat {
        placement,
        route,
        prompt,
        prompt_tokens,
        media,
        gen_config,
        permit,
        visible_tools,
        reasoning_format,
        ..
    } = prepared;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let include_usage = request.include_usage();
    let model_name = request.model.clone();

    // The keep-alive applies once the last token is out, as on the
    // non-streaming path.
    let keep_alive_state = Arc::clone(state);
    let keep_alive_model = model_name.clone();
    let keep_alive = request.keep_alive.clone();
    let rx_stream = spawn_media_stream(
        permit,
        placement,
        route,
        prompt_tokens,
        media,
        gen_config,
        async move {
            apply_keep_alive(&keep_alive_state, &keep_alive_model, &keep_alive).await;
        },
    );

    let parser = ReasoningParser::for_output(reasoning_format, &prompt);
    let rx_stream = Box::pin(rx_stream);
    chat_sse(
        id,
        model_name,
        rx_stream,
        parser,
        visible_tools,
        include_usage,
    )
}

/// The JSON reply of a non-streaming chat request: `n` choices.
async fn complete_reply(
    state: &AppState,
    request: &ChatRequest,
    prepared: PreparedChat<'_>,
) -> Response {
    let _slot = DecodeSlot::enter();
    let has_tools = !prepared.visible_tools.is_empty();
    let start = Instant::now();
    let think_enabled = request.think.unwrap_or(false);
    let n = request.n.unwrap_or(1).max(1);
    let model_name = request.model.clone();

    let mut choices = Vec::with_capacity(n);
    let mut total_prompt_tokens = 0;
    let mut total_completion_tokens = 0;
    let mut first_prompt_eval_ms = 0u64;
    let mut thinking = None;

    for i in 0..n {
        let mut iter_config = prepared.gen_config.clone();
        if let Some(s) = iter_config.seed {
            iter_config.seed = Some(s.wrapping_add(i as u64));
        }

        // The media route picks the generator. Text can go through the
        // batching scheduler.
        let gen_result = prepared.generate_once(&iter_config).await;

        match gen_result {
            Ok(result) => {
                if i == 0 {
                    total_prompt_tokens = result.prompt_tokens;
                    first_prompt_eval_ms = result.prompt_eval_duration_ms;
                }
                total_completion_tokens += result.completion_tokens;

                let (reasoning_content, content) = split_reasoning_for_format(
                    prepared.reasoning_format,
                    &prepared.prompt,
                    result.text,
                );
                if i == 0 && think_enabled {
                    thinking = reasoning_content.clone();
                }

                let logprobs = result
                    .token_logprobs
                    .as_ref()
                    .map(|tl| convert_logprobs(tl));

                // Extract tool calls if tools were provided
                let (final_content, tool_calls) = if has_tools {
                    extract_tool_calls(&content, prepared.visible_tools)
                } else {
                    (Some(content), None)
                };

                let finish = if tool_calls.is_some() {
                    "tool_calls".to_string()
                } else {
                    result.finish_reason.as_str().to_string()
                };

                choices.push(ChatChoice {
                    index: i,
                    message: super::tools::ChatResponseMessage {
                        role: "assistant".to_string(),
                        content: final_content,
                        reasoning_content,
                        tool_calls,
                    },
                    finish_reason: finish,
                    logprobs,
                });
            }
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "server_error",
                );
            }
        }
    }

    let elapsed = start.elapsed();
    let done_reason = choices
        .last()
        .map(|c| c.finish_reason.clone())
        .unwrap_or_default();
    let response = ChatResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: model_name.clone(),
        choices,
        thinking,
        usage: Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
            total_duration_ms: elapsed.as_millis() as u64,
            prompt_eval_duration_ms: first_prompt_eval_ms,
            load_duration_ms: prepared.load_duration_ms,
            tokens_per_second: if elapsed.as_secs_f64() > 0.0 {
                total_completion_tokens as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            },
        },
        done_reason: Some(done_reason),
    };
    record_generation_metrics(
        &model_name,
        &request.user,
        total_prompt_tokens,
        total_completion_tokens,
        first_prompt_eval_ms,
        elapsed,
    );
    apply_keep_alive(state, &model_name, &request.keep_alive).await;
    (StatusCode::OK, Json(response)).into_response()
}
