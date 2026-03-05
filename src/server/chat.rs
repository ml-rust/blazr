//! Chat completion endpoint handler

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::generation::{
    apply_keep_alive, convert_logprobs, decode_context_prefix, error_response, overloaded_response,
    record_generation_metrics, stream_with_stop_sequences, validate_generation_params,
    HasSamplingFields, LogprobResult, ResponseFormat, SamplingParams, Usage,
};
use super::handlers::AppState;
use super::metrics;
use super::streaming::{create_chat_stream, StreamToken};
use super::tools::{
    build_tools_system_prompt, extract_tool_calls, request_msg_to_chat_msg, ChatRequestMessage,
    ChatResponseMessage, Tool,
};

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

    let prompt = if request.raw.unwrap_or(false) {
        request
            .messages
            .iter()
            .map(|m| m.content.as_deref().unwrap_or(""))
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
            match executor.generate_text(&prompt, &iter_config).await {
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
                        message: ChatResponseMessage {
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

// ── Request/Response types ──

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatRequestMessage>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    #[allow(dead_code)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub raw: Option<bool>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub context: Option<Vec<i64>>,
    #[serde(default)]
    pub think: Option<bool>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub keep_alive: Option<String>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<usize>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub mirostat: Option<u8>,
    #[serde(default)]
    pub mirostat_tau: Option<f32>,
    #[serde(default)]
    pub mirostat_eta: Option<f32>,
    #[serde(default)]
    pub dynatemp_range: Option<f32>,
    #[serde(default)]
    pub dynatemp_exponent: Option<f32>,
    #[serde(default)]
    pub dry_multiplier: Option<f32>,
    #[serde(default)]
    pub dry_base: Option<usize>,
    #[serde(default)]
    pub dry_allowed_length: Option<usize>,
    #[serde(default)]
    pub dry_sequence_breakers: Option<Vec<String>>,
    #[serde(default)]
    pub typical_p: Option<f32>,
    #[serde(default)]
    pub user: Option<String>,
}

impl HasSamplingFields for ChatRequest {
    fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            stop: self.stop.clone(),
            seed: self.seed,
            logit_bias: self.logit_bias.clone(),
            logprobs: self.logprobs,
            top_logprobs: self.top_logprobs,
            json_mode: self
                .response_format
                .as_ref()
                .is_some_and(|rf| rf.r#type == "json_object"),
            mirostat_mode: self.mirostat,
            mirostat_tau: self.mirostat_tau,
            mirostat_eta: self.mirostat_eta,
            dynatemp_range: self.dynatemp_range,
            dynatemp_exponent: self.dynatemp_exponent,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_sequence_breakers: self.dry_sequence_breakers.clone(),
            typical_p: self.typical_p,
        }
    }
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatResponseMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobResult>,
}
