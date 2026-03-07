//! Chat completion endpoint handler

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::chat_types::{ChatChoice, ChatRequest, ChatResponse};
use super::generation::{
    apply_keep_alive, convert_logprobs, decode_context_prefix, error_response,
    generate_via_scheduler, overloaded_response, record_generation_metrics,
    stream_with_stop_sequences, validate_generation_params, HasSamplingFields, Usage,
};
use super::handlers::AppState;
use super::metrics;
use super::streaming::{create_chat_stream, StreamToken};
use super::tools::{build_tools_system_prompt, extract_tool_calls, request_msg_to_chat_msg};

/// Chat completion endpoint
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatRequest>,
) -> Response {
    if let Err(e) =
        validate_generation_params(request.temperature, request.top_p, request.max_tokens)
    {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    if request.messages.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "messages array must not be empty",
            "invalid_request_error",
        );
    }

    let load_start = Instant::now();
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
    let load_duration_ms = load_start.elapsed().as_millis() as u64;

    let context_prefix = match decode_context_prefix(&executor, &request.context) {
        Ok(prefix) => prefix,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to decode context tokens: {}", e),
                "invalid_request_error",
            );
        }
    };

    let tools_system_prompt = request.tools.as_deref().and_then(build_tools_system_prompt);
    let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());

    // Check for multimodal content (images/audio) — currently unsupported at model level
    let has_multimodal = request.messages.iter().any(|m| {
        m.content
            .as_ref()
            .is_some_and(|c| c.has_images() || c.has_audio())
    });
    if has_multimodal {
        tracing::warn!("Request contains multimodal content (images/audio). Text will be extracted but media content requires a vision/audio model.");
    }

    let prompt = if request.raw.unwrap_or(false) {
        request
            .messages
            .iter()
            .map(|m| m.content.as_ref().map(|c| c.text()).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        let mut msgs: Vec<crate::model::chat_template::ChatMessage> = Vec::new();
        if let Some(ref sys) = request.system {
            let has_system = request.messages.iter().any(|m| m.role == "system");
            if !has_system {
                msgs.push(crate::model::chat_template::ChatMessage {
                    role: "system".to_string(),
                    content: sys.clone(),
                });
            }
        }
        // Inject tools system prompt
        if let Some(ref tools_prompt) = tools_system_prompt {
            msgs.push(crate::model::chat_template::ChatMessage {
                role: "system".to_string(),
                content: tools_prompt.clone(),
            });
        }
        msgs.extend(request.messages.iter().map(request_msg_to_chat_msg));
        if let Some(ref tpl_name) = request.template {
            crate::model::chat_template::ChatTemplate::from_name(tpl_name).apply(&msgs)
        } else {
            executor.chat_template().apply(&msgs)
        }
    };
    let prompt = if context_prefix.is_empty() {
        prompt
    } else {
        format!("{}{}", context_prefix, prompt)
    };

    let gen_config = request.sampling_params().into_gen_config();

    // Token budget admission control
    let prompt_token_count = executor
        .tokenizer()
        .encode(&prompt)
        .map(|t| t.len())
        .unwrap_or(prompt.len() / 4);
    let estimated_tokens = prompt_token_count + gen_config.max_tokens;
    if !state.try_admit(estimated_tokens) {
        return overloaded_response();
    }

    if request.stream.unwrap_or(false) {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let model_name = request.model;
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamToken>(32);
        let state_clone = Arc::clone(&state);
        let budget = estimated_tokens;

        metrics::adjust_decode_slots(1.0);
        tokio::spawn(async move {
            stream_with_stop_sequences(executor, prompt, gen_config, tx).await;
            state_clone.release(budget);
            metrics::adjust_decode_slots(-1.0);
        });

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        create_chat_stream(id, model_name, Box::pin(rx_stream)).into_response()
    } else {
        metrics::adjust_decode_slots(1.0);
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
            let mut iter_config = gen_config.clone();
            if let Some(s) = iter_config.seed {
                iter_config.seed = Some(s.wrapping_add(i as u64));
            }

            // Use RequestScheduler if available (continuous batching), else direct generation
            let gen_result = if let Some(ref rs) = state.request_scheduler {
                match generate_via_scheduler(rs, &executor, &prompt, &iter_config).await {
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
                executor.generate_text(&prompt, &iter_config).await
            };

            match gen_result {
                Ok(result) => {
                    if i == 0 {
                        total_prompt_tokens = result.prompt_tokens;
                        first_prompt_eval_ms = result.prompt_eval_duration_ms;
                    }
                    total_completion_tokens += result.completion_tokens;

                    let (content, think_content) = if think_enabled {
                        let extracted = crate::model::think::extract_thinking(&result.text);
                        let t = if extracted.thinking.is_empty() {
                            None
                        } else {
                            Some(extracted.thinking.join("\n\n"))
                        };
                        (extracted.content, t)
                    } else {
                        (result.text, None)
                    };
                    if i == 0 {
                        thinking = think_content;
                    }

                    let logprobs = result
                        .token_logprobs
                        .as_ref()
                        .map(|tl| convert_logprobs(tl));

                    // Extract tool calls if tools were provided
                    let (final_content, tool_calls) = if has_tools {
                        extract_tool_calls(&content)
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
                            tool_calls,
                        },
                        finish_reason: finish,
                        logprobs,
                    });
                }
                Err(e) => {
                    state.release(estimated_tokens);
                    metrics::adjust_decode_slots(-1.0);
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
            model: request.model,
            choices,
            thinking,
            usage: Usage {
                prompt_tokens: total_prompt_tokens,
                completion_tokens: total_completion_tokens,
                total_tokens: total_prompt_tokens + total_completion_tokens,
                total_duration_ms: elapsed.as_millis() as u64,
                prompt_eval_duration_ms: first_prompt_eval_ms,
                load_duration_ms,
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
        apply_keep_alive(&state, &model_name, &request.keep_alive).await;
        state.release(estimated_tokens);
        metrics::adjust_decode_slots(-1.0);
        (StatusCode::OK, Json(response)).into_response()
    }
}
