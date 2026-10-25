//! The `/v1/completions` handler.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use futures::StreamExt;

use super::completion_choices::generate_choices;
use super::completion_prompt::{resolve_prompt, PromptContext, ResolvedPrompt};
use super::completion_types::{CompletionRequest, CompletionResponse};
use crate::server::chat_vision::{ImageRoute, RequestMedia};
use crate::server::completion_stream::create_completion_choices_stream;
use crate::server::generation::{
    admit_generation, apply_keep_alive, choice_config, decode_context_prefix, error_response,
    overloaded_response, record_generation_metrics, resolve_constraint, HasSamplingFields, Usage,
};
use crate::server::handlers::{select_executor, AppState, DecodeSlot};
use crate::server::route_stream::spawn_media_streams;
use crate::server::streaming::{StreamToken, TokenStream};

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
    if let Some(e) = placement.refusal(&gen_config) {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    if request.stream.unwrap_or(false) {
        let id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let model_name = request.model.clone();
        let Some(prompt) = prompts.into_iter().next() else {
            return error_response(
                StatusCode::BAD_REQUEST,
                "prompt must not be empty",
                "invalid_request_error",
            );
        };

        // The keep-alive applies once the last token is out, as on the
        // non-streaming path.
        let keep_alive_state = Arc::clone(&state);
        let keep_alive_model = model_name.clone();
        let keep_alive = request.keep_alive.clone();
        // The `n` choices run one after another, each sampled as the
        // non-streaming path samples it. The task holds the request's slot
        // until the last one stops.
        let configs = (0..request.choices_per_prompt())
            .map(|i| choice_config(&gen_config, i))
            .collect();
        let streams = spawn_media_streams(
            permit,
            placement,
            ImageRoute::Text,
            prompt.tokens,
            RequestMedia::default(),
            configs,
            async move {
                apply_keep_alive(&keep_alive_state, &keep_alive_model, &keep_alive).await;
            },
        );
        // `echo` sends the prompt text as the first chunk of each choice.
        let echoed = prompt.echo;
        let choices = streams
            .into_iter()
            .map(|tokens| -> TokenStream {
                if !echo {
                    return tokens;
                }
                let head = StreamToken::text(echoed.clone(), None);
                Box::pin(futures::stream::once(std::future::ready(head)).chain(tokens))
            })
            .collect();
        let include_usage = request.include_usage();
        create_completion_choices_stream(id, model_name, choices, include_usage).into_response()
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
    use serde_json::{json, Value};

    use crate::server::ollama_api::e2e_rig::{Rig, MODEL};
    use crate::server::streaming::sse_test::parse_events;

    /// The joined `delta.text` of each choice index in `events`.
    fn streamed_texts(events: &[Value], n: usize) -> Vec<String> {
        let mut texts = vec![String::new(); n];
        for event in events {
            let Some(choice) = event["choices"].get(0) else {
                continue;
            };
            let index = choice["index"].as_u64().unwrap_or(u64::MAX) as usize;
            if let (Some(text), Some(slot)) =
                (choice["delta"]["text"].as_str(), texts.get_mut(index))
            {
                slot.push_str(text);
            }
        }
        texts
    }

    /// A streamed `n = 2` request sends both choices in order at their own
    /// index, each as the non-streaming request samples it, and one usage
    /// chunk that counts the prompt once. `echo` opens each choice.
    #[tokio::test]
    async fn streaming_two_choices_matches_the_non_streaming_choices() -> Result<()> {
        let rig = Rig::new().await?;
        for echo in [false, true] {
            let mut body = json!({
                "model": MODEL,
                "prompt": "Hi",
                "n": 2,
                "max_tokens": 4,
                "temperature": 0.9,
                "seed": 7,
                "echo": echo,
            });
            let whole = rig.ok("/v1/completions", &body).await?;
            body["stream"] = json!(true);
            body["stream_options"] = json!({"include_usage": true});
            let (status, raw) = rig.post("/v1/completions", &body).await?;
            assert_eq!(status, 200, "{raw}");
            let events = parse_events(&raw)?;

            let want: Vec<String> = (0..2)
                .map(|i| {
                    whole["choices"][i]["text"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string()
                })
                .collect();
            assert_eq!(streamed_texts(&events, 2), want, "echo {echo}: {raw}");
            let indices: Vec<u64> = events
                .iter()
                .filter_map(|e| e["choices"].get(0)?["index"].as_u64())
                .collect();
            assert!(indices.windows(2).all(|w| w[0] <= w[1]), "{raw}");
            for index in [0u64, 1] {
                let finishes: Vec<&Value> = events
                    .iter()
                    .filter_map(|e| e["choices"].get(0))
                    .filter(|c| c["index"].as_u64() == Some(index))
                    .filter(|c| c["finish_reason"].is_string())
                    .collect();
                assert_eq!(finishes.len(), 1, "choice {index}: {raw}");
                let want = &whole["choices"][index as usize]["finish_reason"];
                assert_eq!(&finishes[0]["finish_reason"], want);
            }
            let usage: Vec<&Value> = events.iter().filter(|e| e["usage"].is_object()).collect();
            assert_eq!(usage.len(), 1, "{raw}");
            assert_eq!(usage[0]["choices"], json!([]));
            for field in ["prompt_tokens", "completion_tokens", "total_tokens"] {
                assert_eq!(usage[0]["usage"][field], whole["usage"][field], "{field}");
            }
            assert_eq!(events.last(), Some(&json!("[DONE]")));
        }
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
