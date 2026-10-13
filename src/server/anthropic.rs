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

use super::anthropic_stream::create_message_stream;
use super::anthropic_types::{
    response_blocks, AnthropicRequest, AnthropicResponse, AnthropicUsage, TokenCountResponse,
};
use super::chat_vision::{decode_image_sources, image_route, ImageRoute};
use super::generation::{
    overloaded_response, record_generation_metrics, stream_images_with_stop_sequences,
    stream_multimodal_with_stop_sequences, stream_with_stop_sequences, validate_generation_params,
    HasSamplingFields,
};
use super::handlers::{classify_lookup, AppState};
use super::metrics;
use super::reasoning::ReasoningParser;
use super::scheduled::generate_text_routed;
use super::streaming::StreamToken;
use crate::model::chat_template::{
    encode_chat_prompt, encode_chat_prompt_with_markers, RenderOptions,
};

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
        Err(e) => return anthropic_lookup_error(&e),
    };
    let _load_duration_ms = load_start.elapsed().as_millis() as u64;

    // Decode the `image` blocks, then pick the path that serves them: the
    // vision tower, the model's own encoders, or a 400.
    let media = {
        let sources = match request.image_sources() {
            Ok(sources) => sources,
            Err(message) => {
                return anthropic_error(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
            }
        };
        // `sources` borrows the request, so it ends here: the handler moves
        // request fields into the reply below.
        match decode_image_sources(&sources).await {
            Ok(media) => media,
            Err(message) => {
                return anthropic_error(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
            }
        }
    };
    let multimodal_model = matches!(executor.model(), boostr::model::LoadedModel::Multimodal(_));
    let route = match image_route(executor.vision_tower().is_some(), multimodal_model, &media) {
        Ok(route) => route,
        Err(message) => {
            return anthropic_error(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
        }
    };
    let msgs = request.chat_messages(route.image_marker());

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
        Err(e) => return anthropic_prompt_error(e),
    };
    // The marker allow-list let the caller's text spell a marker too: each
    // one must appear exactly once per image.
    if let (ImageRoute::Tower, Some(tower)) = (route, executor.vision_tower()) {
        if let Err(message) = tower.check_marker_counts(&prompt_tokens, media.images.len()) {
            return anthropic_error(StatusCode::BAD_REQUEST, &message, "invalid_request_error");
        }
    }
    let gen_config = request
        .sampling_params()
        .into_gen_config(&state.config().generation);
    let reasoning_format = state.reasoning_format();

    // Token budget admission
    let estimated_tokens = prompt_tokens.len() + gen_config.max_tokens;
    if !state.try_admit(estimated_tokens) {
        return overloaded_response();
    }

    // The executor and scheduler this request runs on. With data
    // parallelism, the router picks a replica here.
    let placement = state.route(executor, &prompt_tokens);
    let executor = Arc::clone(&placement.executor);

    if request.stream.unwrap_or(false) {
        let id = format!("msg_{}", uuid::Uuid::new_v4());
        let model_name = request.model;
        let input_tokens = prompt_tokens.len();
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
        create_message_stream(id, model_name, input_tokens, Box::pin(rx_stream), parser)
            .into_response()
    } else {
        metrics::adjust_decode_slots(1.0);
        let start = Instant::now();
        let model_name = request.model.clone();

        // The media route picks the generator; text may go through the
        // batching scheduler.
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
        Err(e) => return anthropic_lookup_error(&e),
    };

    // The prompt as the messages endpoint assembles it, except that counting
    // fetches no image: each one counts as its `[image]` note.
    let msgs = request.chat_messages(None);

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

/// The Anthropic error response of a failed executor lookup: the status
/// and message of [`classify_lookup`], under the Anthropic error type of
/// that status.
fn anthropic_lookup_error(error: &anyhow::Error) -> Response {
    let failure = classify_lookup(error);
    anthropic_error(
        failure.status,
        &failure.message,
        anthropic_error_type(failure.status),
    )
}

/// The Anthropic error `type` of an HTTP status.
fn anthropic_error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
        s if s.is_client_error() => "invalid_request_error",
        _ => "api_error",
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelNotFound;

    async fn body_of(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body reads");
        serde_json::from_slice(&bytes).expect("body is JSON")
    }

    #[tokio::test]
    async fn lookup_errors_keep_the_anthropic_envelope() {
        let missing = anthropic_lookup_error(&ModelNotFound::new("x").into());
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body_of(missing).await,
            serde_json::json!({"type": "error", "error": {
                "type": "not_found_error",
                "message": "Model not found: x",
            }})
        );

        let load = anthropic_lookup_error(&anyhow::anyhow!("weights do not read"));
        assert_eq!(load.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            body_of(load).await,
            serde_json::json!({"type": "error", "error": {
                "type": "api_error",
                "message": "weights do not read",
            }})
        );
    }

    #[test]
    fn each_status_has_its_anthropic_type() {
        let cases = [
            (StatusCode::BAD_REQUEST, "invalid_request_error"),
            (StatusCode::NOT_FOUND, "not_found_error"),
            (StatusCode::UNPROCESSABLE_ENTITY, "invalid_request_error"),
            (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
            (StatusCode::SERVICE_UNAVAILABLE, "overloaded_error"),
        ];
        for (status, kind) in cases {
            assert_eq!(anthropic_error_type(status), kind);
        }
    }
}
