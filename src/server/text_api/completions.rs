//! The `/v1/completions` handler.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::completion_choices::generate_choices;
use super::completion_prompt::{resolve_prompt, PromptContext, ResolvedPrompt};
use super::completion_types::{CompletionRequest, CompletionResponse};
use crate::server::generation::{
    admit_generation, apply_keep_alive, decode_context_prefix, error_response, overloaded_response,
    record_generation_metrics, resolve_constraint, HasSamplingFields, Usage,
};
use crate::server::handlers::{select_executor, AppState, DecodeSlot};
use crate::server::scheduled_stream::stream_routed;
use crate::server::streaming::{create_completion_stream, StreamToken};

/// Text completion endpoint
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    if let Err(e) = request.check_fields() {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    let load_start = Instant::now();
    let executor =
        match select_executor(&state, &request.model, request.lora_adapter.as_deref()).await {
            Ok(e) => e,
            Err(response) => return *response,
        };
    let load_duration_ms = load_start.elapsed().as_millis() as u64;

    let prefix = match decode_context_prefix(&executor, &request.context) {
        Ok(prefix) => prefix,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Failed to decode context tokens: {}", e),
                "invalid_request_error",
            );
        }
    };

    let mut gen_config = request
        .sampling_params()
        .into_gen_config(&state.config().generation);
    let requested =
        resolve_constraint(request.grammar.as_deref(), request.response_format.as_ref());
    if let Err(e) = admit_generation(&state, &mut gen_config, requested, executor.tokenizer()).await
    {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    // Each prompt is encoded once, and these ids reach the model.
    // Caller text becomes a control token only when `allowed_special` names it.
    // `/v1/tokenize` follows the same rule. Token id prompts skip the tokenizer.
    let echo = request.echo.unwrap_or(false);
    let context = PromptContext {
        tokenizer: executor.tokenizer(),
        vocab_size: executor.vocab_size(),
        prefix: &prefix,
        suffix: request.suffix.as_deref(),
        allowed_special: request.allowed_special.as_deref(),
        echo,
    };
    let mut prompts: Vec<ResolvedPrompt> = Vec::new();
    for item in request.prompt.items() {
        match resolve_prompt(&item, &context) {
            Ok(p) => prompts.push(p),
            Err(e) => return e.into_response(),
        }
    }
    let Some(first) = prompts.first() else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "prompt must not be empty",
            "invalid_request_error",
        );
    };

    // Token budget admission control: every prompt with its generation budget.
    let estimated_tokens: usize = prompts
        .iter()
        .map(|p| p.tokens.len() + gen_config.max_tokens)
        .sum();
    let Some(permit) = state.try_admit(estimated_tokens) else {
        return overloaded_response();
    };

    // The executor and scheduler this request runs on. With data
    // parallelism, the router picks a replica here.
    let placement = state.route(executor, &first.tokens);

    if request.stream.unwrap_or(false) {
        let id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let model_name = request.model.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamToken>(32);
        let Some(prompt) = prompts.into_iter().next() else {
            return error_response(
                StatusCode::BAD_REQUEST,
                "prompt must not be empty",
                "invalid_request_error",
            );
        };
        // `echo` sends the prompt text as the first chunk.
        if echo {
            let _ = tx.try_send(StreamToken::text(prompt.echo, None));
        }

        // The keep-alive applies once the last token is out, as on the
        // non-streaming path.
        let keep_alive_state = Arc::clone(&state);
        let keep_alive_model = model_name.clone();
        let keep_alive = request.keep_alive.clone();
        let slot = DecodeSlot::enter();
        tokio::spawn(async move {
            let _guards = (permit, slot);
            stream_routed(&placement, prompt.tokens, gen_config, tx).await;
            apply_keep_alive(&keep_alive_state, &keep_alive_model, &keep_alive).await;
        });

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let include_usage = request.include_usage();
        create_completion_stream(id, model_name, Box::pin(rx_stream), include_usage).into_response()
    } else {
        let _slot = DecodeSlot::enter();
        let _permit = permit;
        let n = request.choices_per_prompt();
        let model_name = request.model.clone();
        let start = Instant::now();

        let generated = match generate_choices(&placement, &prompts, n, echo, &gen_config).await {
            Ok(g) => g,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "server_error",
                );
            }
        };

        let elapsed = start.elapsed();
        let done_reason = generated
            .choices
            .last()
            .map(|c| c.finish_reason.clone())
            .unwrap_or_default();
        let (prompt_tokens, completion_tokens) =
            (generated.prompt_tokens, generated.completion_tokens);
        let response = CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: request.model,
            choices: generated.choices,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                total_duration_ms: elapsed.as_millis() as u64,
                prompt_eval_duration_ms: generated.first_prompt_eval_ms,
                load_duration_ms,
                tokens_per_second: if elapsed.as_secs_f64() > 0.0 {
                    completion_tokens as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                },
            },
            done_reason: Some(done_reason),
        };
        record_generation_metrics(
            &model_name,
            &request.user,
            prompt_tokens,
            completion_tokens,
            generated.first_prompt_eval_ms,
            elapsed,
        );
        apply_keep_alive(&state, &model_name, &request.keep_alive).await;
        (StatusCode::OK, Json(response)).into_response()
    }
}

/// End-to-end runs through the router on a loaded seeded model.
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use std::time::Duration;

    use anyhow::{anyhow, Result};
    use serde_json::json;

    use crate::server::ollama_api::e2e_rig::{Rig, MODEL};

    #[tokio::test]
    async fn streaming_with_more_than_one_choice_is_a_400() -> Result<()> {
        let rig = Rig::new().await?;
        let body = json!({"model": MODEL, "prompt": "Hi", "stream": true, "n": 2});
        let (status, raw) = rig.post("/v1/completions", &body).await?;
        assert_eq!(status, 400, "{raw}");
        assert!(raw.contains("streaming supports n = 1"), "{raw}");
        Ok(())
    }

    #[tokio::test]
    async fn a_streamed_request_applies_keep_alive_after_its_last_token() -> Result<()> {
        let rig = Rig::new().await?;
        let body = json!({
            "model": MODEL,
            "prompt": "Hi",
            "stream": true,
            "max_tokens": 3,
            "temperature": 0,
            "keep_alive": "0",
        });
        let (status, raw) = rig.post("/v1/completions", &body).await?;
        assert_eq!(status, 200, "{raw}");
        // The stream closes before the task applies the keep-alive.
        for _ in 0..200 {
            if !rig.state.scheduler.is_loaded(MODEL).await {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        Err(anyhow!("the model is still loaded after the stream ended"))
    }
}
