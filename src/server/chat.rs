//! Chat completion endpoint handler

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::chat_types::{ChatChoice, ChatRequest, ChatResponse};
use super::chat_vision::{decode_request_media, image_route, ImageRoute};
use super::generation::{
    apply_keep_alive, chat_prompt_error_response, convert_logprobs, decode_context_prefix,
    error_response, overloaded_response, policy_error_response, record_generation_metrics,
    stream_images_with_stop_sequences, stream_multimodal_with_stop_sequences,
    stream_with_stop_sequences, validate_generation_params, HasSamplingFields, Usage,
};
use super::handlers::AppState;
use super::metrics;
use super::reasoning::{split_reasoning_for_format, ReasoningParser};
use super::scheduled::generate_text_routed;
use super::streaming::{create_chat_stream, StreamToken};
use super::tool_calls::extract_tool_calls;
use super::tools::{build_tools_system_prompt, request_msg_to_chat_msg};
use crate::model::chat_template::{encode_chat_prompt_with_markers, ChatMessage, ChatTemplate};

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

    let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());

    // The request may name a built-in format; otherwise the model's template.
    let override_template = match request.template.as_deref().map(named_template) {
        Some(Ok(template)) => Some(template),
        Some(Err(message)) => {
            return error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
        }
        None => None,
    };
    let template = override_template
        .as_ref()
        .unwrap_or_else(|| executor.chat_template());
    // A Jinja template takes the tools and the structured turns itself; a
    // built-in format takes a tool description as a system turn.
    let native_tools = template.renders_tools_natively();
    let tools_system_prompt = if native_tools {
        None
    } else {
        request.tools.as_deref().and_then(build_tools_system_prompt)
    };

    // Decode the media parts, then pick the path that serves them: the
    // vision tower, the model's own encoders, or a 400.
    let media = match decode_request_media(&request.messages).await {
        Ok(media) => media,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
        }
    };
    let multimodal_model = matches!(executor.model(), boostr::model::LoadedModel::Multimodal(_));
    let route = match image_route(executor.vision_tower().is_some(), multimodal_model, &media) {
        Ok(route) => route,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
        }
    };
    if route == ImageRoute::Tower && request.raw.unwrap_or(false) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "raw mode renders no image markers; send images without `raw`",
            "invalid_request_error",
        );
    }

    // The prompt is encoded here, once, under an explicit special-token policy:
    // whatever the server itself inserts is allowed and everything else in the
    // caller's text is refused. The ids double as the token-budget count.
    let (prompt, prompt_tokens) = if request.raw.unwrap_or(false) {
        // Raw mode inserts no delimiters of its own, so nothing in the caller's
        // text may become a control token.
        let raw_prompt = format!(
            "{}{}",
            context_prefix,
            request
                .messages
                .iter()
                .map(|m| m.content.as_ref().map(|c| c.text()).unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n")
        );
        match executor
            .tokenizer()
            .encode_with(&raw_prompt, &splintr::SpecialMode::Ordinary)
        {
            Ok(ids) => (raw_prompt, ids),
            Err(e) => return policy_error_response(&e),
        }
    } else {
        let mut msgs: Vec<ChatMessage> = Vec::new();
        if let Some(ref sys) = request.system {
            let has_system = request.messages.iter().any(|m| m.role == "system");
            if !has_system {
                msgs.push(ChatMessage::new("system", sys.clone()));
            }
        }
        // Inject tools system prompt
        if let Some(ref tools_prompt) = tools_system_prompt {
            msgs.push(ChatMessage::new("system", tools_prompt.clone()));
        }
        msgs.extend(
            request
                .messages
                .iter()
                .map(|m| request_msg_to_chat_msg(m, native_tools, route.image_marker())),
        );
        let options = request.render_options(native_tools);
        match encode_chat_prompt_with_markers(
            template,
            executor.tokenizer(),
            &msgs,
            &context_prefix,
            &options,
            &route.server_markers(),
        ) {
            Ok(encoded) => encoded,
            Err(e) => return chat_prompt_error_response(&e),
        }
    };
    // The marker allow-list let the caller's text spell a marker too: each
    // one must appear exactly once per image.
    if let (ImageRoute::Tower, Some(tower)) = (route, executor.vision_tower()) {
        if let Err(message) = tower.check_marker_counts(&prompt_tokens, media.images.len()) {
            return error_response(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
        }
    }

    let gen_config = request
        .sampling_params()
        .into_gen_config(&state.config().generation);
    let reasoning_format = request
        .reasoning_format
        .unwrap_or_else(|| state.reasoning_format());

    // Token budget admission control
    let estimated_tokens = prompt_tokens.len() + gen_config.max_tokens;
    if !state.try_admit(estimated_tokens) {
        return overloaded_response();
    }

    // The executor and scheduler this request runs on. With data
    // parallelism, the router picks a replica here.
    let placement = state.route(executor, &prompt_tokens);
    let executor = Arc::clone(&placement.executor);

    if request.stream.unwrap_or(false) {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let model_name = request.model;
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamToken>(32);
        let state_clone = Arc::clone(&state);
        let budget = estimated_tokens;

        metrics::adjust_decode_slots(1.0);
        tokio::spawn(async move {
            let _placement = placement;
            match route {
                ImageRoute::Tower => {
                    stream_images_with_stop_sequences(
                        executor,
                        prompt_tokens,
                        media.images,
                        gen_config,
                        tx,
                    )
                    .await
                }
                ImageRoute::Multimodal => {
                    stream_multimodal_with_stop_sequences(
                        executor,
                        prompt_tokens,
                        media.images,
                        media.audio,
                        gen_config,
                        tx,
                    )
                    .await
                }
                ImageRoute::Text => {
                    stream_with_stop_sequences(executor, prompt_tokens, gen_config, tx).await
                }
            }
            state_clone.release(budget);
            metrics::adjust_decode_slots(-1.0);
        });

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let parser = ReasoningParser::for_output(reasoning_format, &prompt);
        create_chat_stream(id, model_name, Box::pin(rx_stream), parser).into_response()
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

            // The media route picks the generator; text may go through the
            // batching scheduler.
            let gen_result = if route == ImageRoute::Tower {
                executor
                    .generate_images_text(&prompt_tokens, &media.images, &iter_config)
                    .await
            } else if route == ImageRoute::Multimodal {
                executor
                    .generate_multimodal_text(
                        &prompt_tokens,
                        &media.images,
                        &media.audio,
                        &iter_config,
                    )
                    .await
            } else {
                generate_text_routed(&placement, &prompt_tokens, &iter_config).await
            };

            match gen_result {
                Ok(result) => {
                    if i == 0 {
                        total_prompt_tokens = result.prompt_tokens;
                        first_prompt_eval_ms = result.prompt_eval_duration_ms;
                    }
                    total_completion_tokens += result.completion_tokens;

                    let (reasoning_content, content) =
                        split_reasoning_for_format(reasoning_format, &prompt, result.text);
                    if i == 0 && think_enabled {
                        thinking = reasoning_content.clone();
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
                            reasoning_content,
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
