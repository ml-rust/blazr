//! Shared generation logic for completion and chat handlers.
//!
//! Contains SamplingParams, stop sequence handling, metrics recording,
//! validation, and shared response types.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use super::metrics;
use super::streaming::StreamToken;
use crate::config::GenerationConfig;
use crate::engine::{parse_keep_alive, FinishReason};

use super::handlers::AppState;

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

/// OpenAI response_format parameter
#[derive(Deserialize, Default, Clone)]
pub struct ResponseFormat {
    /// "text" (default) or "json_object"
    #[serde(default = "default_response_format_type")]
    pub r#type: String,
}

fn default_response_format_type() -> String {
    "text".to_string()
}

/// Sampling parameters shared between completion and chat requests
pub struct SamplingParams {
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<u64>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<usize>,
    pub json_mode: bool,
    pub mirostat_mode: Option<u8>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub dynatemp_range: Option<f32>,
    pub dynatemp_exponent: Option<f32>,
    pub dry_multiplier: Option<f32>,
    pub dry_base: Option<usize>,
    pub dry_allowed_length: Option<usize>,
    pub dry_sequence_breakers: Option<Vec<String>>,
    pub typical_p: Option<f32>,
}

impl SamplingParams {
    pub fn into_gen_config(self) -> GenerationConfig {
        let logit_bias = self
            .logit_bias
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, v)))
            .collect();
        GenerationConfig {
            max_tokens: self.max_tokens.unwrap_or(256),
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k.unwrap_or(0),
            min_p: self.min_p.unwrap_or(0.05),
            repeat_penalty: self.repeat_penalty.unwrap_or(1.1),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            stop_sequences: self.stop.unwrap_or_default(),
            seed: self.seed,
            logit_bias,
            logprobs: self.logprobs.unwrap_or(false),
            top_logprobs: self.top_logprobs.unwrap_or(5).min(20),
            json_mode: self.json_mode,
            mirostat_mode: self.mirostat_mode.unwrap_or(0),
            mirostat_tau: self.mirostat_tau.unwrap_or(5.0),
            mirostat_eta: self.mirostat_eta.unwrap_or(0.1),
            dynatemp_range: self.dynatemp_range.unwrap_or(0.0),
            dynatemp_exponent: self.dynatemp_exponent.unwrap_or(1.0),
            dry_multiplier: self.dry_multiplier.unwrap_or(0.0),
            dry_base: self.dry_base.unwrap_or(2),
            dry_allowed_length: self.dry_allowed_length.unwrap_or(0),
            dry_sequence_breakers: self.dry_sequence_breakers.unwrap_or_default(),
            typical_p: self.typical_p.unwrap_or(0.0),
            ..Default::default()
        }
    }
}

/// Trait for request types that carry sampling fields.
/// Eliminates the duplicated SamplingParams { ... } construction.
pub trait HasSamplingFields {
    fn sampling_params(&self) -> SamplingParams;
}

/// Stream generation tokens with stop sequence checking.
pub async fn stream_with_stop_sequences(
    executor: Arc<crate::engine::Executor<ServerRuntime>>,
    prompt: String,
    gen_config: GenerationConfig,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) {
    let stop_sequences = gen_config.stop_sequences.clone();
    let stream = executor.generate(&prompt, &gen_config);
    let mut stream = std::pin::pin!(stream);
    let mut accumulated = String::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                accumulated.push_str(&token.text);

                // Check stop sequences
                let mut stopped = false;
                for stop in &stop_sequences {
                    if let Some(pos) = accumulated.find(stop.as_str()) {
                        let already_sent = accumulated.len() - token.text.len();
                        if pos >= already_sent {
                            let safe_text = &token.text[..pos - already_sent];
                            let _ = tx
                                .send(StreamToken {
                                    text: safe_text.to_string(),
                                    finish_reason: Some(FinishReason::Stop),
                                    error: None,
                                })
                                .await;
                        } else {
                            let _ = tx
                                .send(StreamToken {
                                    text: String::new(),
                                    finish_reason: Some(FinishReason::Stop),
                                    error: None,
                                })
                                .await;
                        }
                        stopped = true;
                        break;
                    }
                }

                if stopped {
                    break;
                }

                let is_final = token.finish_reason.is_some();
                if tx
                    .send(StreamToken {
                        text: token.text,
                        finish_reason: token.finish_reason,
                        error: None,
                    })
                    .await
                    .is_err()
                {
                    break;
                }
                if is_final {
                    break;
                }
            }
            Err(e) => {
                tracing::error!("Generation error during streaming: {}", e);
                let _ = tx
                    .send(StreamToken {
                        text: String::new(),
                        finish_reason: None,
                        error: Some(e.to_string()),
                    })
                    .await;
                break;
            }
        }
    }
}

/// Record generation metrics (tokens, TTFT, throughput) after a non-streaming request.
pub fn record_generation_metrics(
    model: &str,
    user: &Option<String>,
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt_eval_ms: u64,
    elapsed: std::time::Duration,
) {
    metrics::record_tokens(prompt_tokens, completion_tokens);
    metrics::record_ttft(prompt_eval_ms as f64 / 1000.0);
    let tps = if elapsed.as_secs_f64() > 0.0 {
        completion_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    metrics::record_tokens_per_second(tps);
    tracing::info!(
        model = model,
        user = user.as_deref().unwrap_or("-"),
        prompt_tokens = prompt_tokens,
        completion_tokens = completion_tokens,
        tokens_per_second = format!("{:.1}", tps).as_str(),
        duration_ms = elapsed.as_millis() as u64,
        "generation complete"
    );
}

/// Decode context tokens (from a previous turn) into a string prefix.
pub fn decode_context_prefix(
    executor: &Arc<crate::engine::Executor<ServerRuntime>>,
    context: &Option<Vec<i64>>,
) -> Result<String, String> {
    if let Some(ref ctx_tokens) = context {
        let token_ids: Vec<u32> = ctx_tokens.iter().map(|&t| t as u32).collect();
        executor
            .tokenizer()
            .decode(&token_ids)
            .map_err(|e| e.to_string())
    } else {
        Ok(String::new())
    }
}

/// Validate generation parameters, return error message if invalid
pub fn validate_generation_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
) -> Result<(), String> {
    if let Some(t) = temperature {
        if !(0.0..=2.0).contains(&t) {
            return Err(format!("temperature must be between 0 and 2, got {}", t));
        }
    }
    if let Some(p) = top_p {
        if !(0.0..=1.0).contains(&p) {
            return Err(format!("top_p must be between 0 and 1, got {}", p));
        }
    }
    if let Some(m) = max_tokens {
        if m == 0 {
            return Err("max_tokens must be greater than 0".to_string());
        }
    }
    Ok(())
}

/// Return a 503 Service Unavailable response with Retry-After header
pub fn overloaded_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [("retry-after", "5")],
        axum::Json(serde_json::json!({
            "error": {
                "message": "Server overloaded: in-flight token budget exceeded. Retry after a short delay.",
                "type": "server_overloaded"
            }
        })),
    )
        .into_response()
}

/// Convert executor token logprobs into API response format
pub fn convert_logprobs(tokens: &[crate::engine::GeneratedToken]) -> LogprobResult {
    LogprobResult {
        content: tokens
            .iter()
            .map(|t| LogprobEntry {
                token: t.text.clone(),
                logprob: t.logprob.unwrap_or(f32::NEG_INFINITY),
                top_logprobs: t
                    .top_logprobs
                    .as_ref()
                    .map(|tops| {
                        tops.iter()
                            .map(|tl| TopLogprob {
                                token: tl.text.clone(),
                                logprob: tl.logprob,
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            })
            .collect(),
    }
}

/// Apply per-request keep_alive to the model's scheduler entry
pub async fn apply_keep_alive(state: &AppState, model_name: &str, keep_alive: &Option<String>) {
    if let Some(ref ka) = keep_alive {
        let duration = parse_keep_alive(ka);
        state.scheduler.set_keep_alive(model_name, duration).await;
    }
}

/// Build a standard OpenAI error response
pub fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    (
        status,
        axum::Json(ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                r#type: error_type.to_string(),
            },
        }),
    )
        .into_response()
}

// ── Shared response types ──

/// OpenAI-compatible logprobs result
#[derive(Serialize, Clone)]
pub struct LogprobResult {
    pub content: Vec<LogprobEntry>,
}

/// Log probability for a single token position
#[derive(Serialize, Clone)]
pub struct LogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub top_logprobs: Vec<TopLogprob>,
}

/// A single alternative token with its log probability
#[derive(Serialize, Clone)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub total_duration_ms: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub prompt_eval_duration_ms: u64,
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub load_duration_ms: u64,
    #[serde(skip_serializing_if = "is_zero_f64")]
    pub tokens_per_second: f64,
}

fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
}
