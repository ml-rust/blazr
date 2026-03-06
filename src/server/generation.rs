//! Shared generation logic for completion and chat handlers.
//!
//! Contains SamplingParams, stop sequence handling, metrics recording,
//! and batched generation via the request scheduler.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;

use super::metrics;
use super::streaming::StreamToken;
use crate::config::GenerationConfig;
use crate::engine::FinishReason;

// Re-export types and utilities from gen_types so callers import from one place.
pub use super::gen_types::{
    apply_keep_alive, convert_logprobs, error_response, overloaded_response,
    validate_generation_params, LogprobResult, ResponseFormat, Usage,
};

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

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
    pub grammar: Option<String>,
    /// LoRA adapter name to activate for this request (must already be loaded).
    pub lora_adapter: Option<String>,
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
            grammar: self.grammar,
            lora_adapter: self.lora_adapter,
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
    let mut last_token_time = std::time::Instant::now();
    let mut token_count = 0usize;

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                token_count += 1;
                // Record inter-token latency (skip first token — that's TTFT)
                if token_count > 1 {
                    let itl = last_token_time.elapsed().as_secs_f64();
                    super::metrics::record_itl(itl);
                }
                last_token_time = std::time::Instant::now();
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

/// Result of a batched generation via RequestScheduler
pub struct BatchedGenerationResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
    pub prompt_eval_duration_ms: u64,
}

/// Submit a request to the RequestScheduler and collect all generated tokens.
/// Used by chat/completion handlers when continuous batching is enabled.
pub async fn generate_via_scheduler(
    request_scheduler: &crate::engine::RequestScheduler,
    executor: &Arc<crate::engine::Executor<ServerRuntime>>,
    prompt: &str,
    gen_config: &GenerationConfig,
) -> Result<BatchedGenerationResult, String> {
    let prompt_tokens = executor
        .tokenizer()
        .encode(prompt)
        .map_err(|e| format!("tokenization failed: {}", e))?;
    let prompt_len = prompt_tokens.len();
    let start = std::time::Instant::now();

    let mut handle = request_scheduler
        .submit(prompt_tokens, gen_config.clone())
        .map_err(|e| format!("scheduler submit failed: {}", e))?;

    let mut text = String::new();
    let mut completion_tokens = 0;
    let mut finish = FinishReason::Length;

    while let Some(result) = handle.token_rx.recv().await {
        match result {
            Ok(token) => {
                text.push_str(&token.text);
                completion_tokens += 1;
                if let Some(reason) = token.finish_reason {
                    finish = reason;
                    break;
                }
            }
            Err(e) => {
                return Err(format!("generation error: {}", e));
            }
        }
    }

    // Estimate prompt eval as time to first token (approximation)
    let prompt_eval_ms = start.elapsed().as_millis() as u64;

    Ok(BatchedGenerationResult {
        text,
        prompt_tokens: prompt_len,
        completion_tokens,
        finish_reason: finish,
        prompt_eval_duration_ms: prompt_eval_ms,
    })
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
