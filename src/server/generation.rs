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
    apply_keep_alive, chat_prompt_error_response, convert_logprobs, encode_with_allowed_special,
    error_response, error_response_with_code, overloaded_response, policy_error_response,
    validate_generation_params, LogprobResult, ResponseFormat, Usage,
};

use crate::runtime::ServerRuntime;

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
    /// The request's `lora_adapter`. The handler already chose the variant
    /// executor from it; the generation config only carries it.
    pub lora_adapter: Option<String>,
}

impl SamplingParams {
    /// The request's sampling values over `defaults`, the live
    /// `generation` section. Each value the request omits takes the default.
    pub fn into_gen_config(self, defaults: &GenerationConfig) -> GenerationConfig {
        let d = defaults.clone();
        let logit_bias = match self.logit_bias {
            Some(bias) => bias
                .into_iter()
                .filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, v)))
                .collect(),
            None => d.logit_bias,
        };
        GenerationConfig {
            max_tokens: self.max_tokens.unwrap_or(d.max_tokens),
            temperature: self.temperature.unwrap_or(d.temperature),
            top_p: self.top_p.unwrap_or(d.top_p),
            top_k: self.top_k.unwrap_or(d.top_k),
            min_p: self.min_p.unwrap_or(d.min_p),
            repeat_penalty: self.repeat_penalty.unwrap_or(d.repeat_penalty),
            frequency_penalty: self.frequency_penalty.unwrap_or(d.frequency_penalty),
            presence_penalty: self.presence_penalty.unwrap_or(d.presence_penalty),
            stop_sequences: self.stop.unwrap_or(d.stop_sequences),
            seed: self.seed.or(d.seed),
            logit_bias,
            logprobs: self.logprobs.unwrap_or(d.logprobs),
            top_logprobs: self.top_logprobs.unwrap_or(d.top_logprobs).min(20),
            json_mode: self.json_mode || d.json_mode,
            mirostat_mode: self.mirostat_mode.unwrap_or(d.mirostat_mode),
            mirostat_tau: self.mirostat_tau.unwrap_or(d.mirostat_tau),
            mirostat_eta: self.mirostat_eta.unwrap_or(d.mirostat_eta),
            dynatemp_range: self.dynatemp_range.unwrap_or(d.dynatemp_range),
            dynatemp_exponent: self.dynatemp_exponent.unwrap_or(d.dynatemp_exponent),
            dry_multiplier: self.dry_multiplier.unwrap_or(d.dry_multiplier),
            dry_base: self.dry_base.unwrap_or(d.dry_base),
            dry_allowed_length: self.dry_allowed_length.unwrap_or(d.dry_allowed_length),
            dry_sequence_breakers: self
                .dry_sequence_breakers
                .unwrap_or(d.dry_sequence_breakers),
            typical_p: self.typical_p.unwrap_or(d.typical_p),
            grammar: self.grammar.or(d.grammar),
            lora_adapter: self.lora_adapter.or(d.lora_adapter),
            repeat_last_n: d.repeat_last_n,
            verbose_prompt: d.verbose_prompt,
        }
    }
}

/// Trait for request types that carry sampling fields.
/// Eliminates the duplicated SamplingParams { ... } construction.
pub trait HasSamplingFields {
    fn sampling_params(&self) -> SamplingParams;
}

/// Stream generation tokens with stop sequence checking.
///
/// Takes the prompt ids the handler already validated — stop sequences are
/// matched against the *generated* text, so nothing here needs the prompt text.
pub async fn stream_with_stop_sequences(
    executor: Arc<crate::engine::Executor<ServerRuntime>>,
    prompt_tokens: Vec<u32>,
    gen_config: GenerationConfig,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) {
    let stream = executor.generate(&prompt_tokens, &gen_config);
    forward_with_stop_sequences(stream, &gen_config.stop_sequences, "Generation", tx).await;
}

/// Same as `stream_with_stop_sequences` over `generate_multimodal`, with the
/// decoded images and audio segments of the llava-style path.
pub async fn stream_multimodal_with_stop_sequences(
    executor: Arc<crate::engine::Executor<ServerRuntime>>,
    prompt_tokens: Vec<u32>,
    images: Vec<Vec<u8>>,
    audio_segments: Vec<Vec<f32>>,
    gen_config: GenerationConfig,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) {
    let stream =
        executor.generate_multimodal(&prompt_tokens, &images, &audio_segments, &gen_config);
    forward_with_stop_sequences(
        stream,
        &gen_config.stop_sequences,
        "Multimodal generation",
        tx,
    )
    .await;
}

/// Same as `stream_with_stop_sequences` over `generate_with_images`: the
/// prompt ids hold one image placeholder per entry of `images`.
pub async fn stream_images_with_stop_sequences(
    executor: Arc<crate::engine::Executor<ServerRuntime>>,
    prompt_tokens: Vec<u32>,
    images: Vec<Vec<u8>>,
    gen_config: GenerationConfig,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) {
    let stream = executor.generate_with_images(&prompt_tokens, &images, &gen_config);
    forward_with_stop_sequences(stream, &gen_config.stop_sequences, "Image generation", tx).await;
}

/// Forward `stream` to `tx`, cutting the text at the first stop sequence.
/// `label` names the path in the error log.
async fn forward_with_stop_sequences<S>(
    stream: S,
    stop_sequences: &[String],
    label: &str,
    tx: tokio::sync::mpsc::Sender<StreamToken>,
) where
    S: futures::Stream<Item = anyhow::Result<crate::engine::GeneratedToken>>,
{
    let mut stream = std::pin::pin!(stream);
    let mut accumulated = String::new();
    let start = std::time::Instant::now();
    let mut last_token_time = start;
    let mut token_count = 0usize;

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                token_count += 1;
                // The first token closes the TTFT. Each later one is an ITL sample.
                if token_count == 1 {
                    metrics::record_ttft(start.elapsed().as_secs_f64());
                } else {
                    metrics::record_itl(last_token_time.elapsed().as_secs_f64());
                }
                last_token_time = std::time::Instant::now();
                accumulated.push_str(&token.text);

                // Check stop sequences
                let mut stopped = false;
                for stop in stop_sequences {
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
                tracing::error!("{} error during streaming: {}", label, e);
                let _ = tx
                    .send(StreamToken {
                        text: String::new(),
                        finish_reason: None,
                        error: Some(e.to_string()),
                    })
                    .await;
                return;
            }
        }
    }
    if token_count > 0 {
        metrics::record_e2e(start.elapsed().as_secs_f64());
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
    metrics::record_e2e(elapsed.as_secs_f64());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SamplingParams {
        SamplingParams {
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            seed: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            json_mode: false,
            mirostat_mode: None,
            mirostat_tau: None,
            mirostat_eta: None,
            dynatemp_range: None,
            dynatemp_exponent: None,
            dry_multiplier: None,
            dry_base: None,
            dry_allowed_length: None,
            dry_sequence_breakers: None,
            typical_p: None,
            grammar: None,
            lora_adapter: None,
        }
    }

    #[test]
    fn omitted_values_take_the_live_defaults() {
        let defaults = GenerationConfig {
            max_tokens: 77,
            temperature: 0.3,
            repeat_last_n: 9,
            stop_sequences: vec!["END".into()],
            ..Default::default()
        };
        let config = params().into_gen_config(&defaults);
        assert_eq!(config.max_tokens, 77);
        assert_eq!(config.temperature, 0.3);
        assert_eq!(config.repeat_last_n, 9);
        assert_eq!(config.stop_sequences, ["END"]);
    }

    #[test]
    fn request_values_win_over_the_defaults() {
        let request = SamplingParams {
            max_tokens: Some(5),
            temperature: Some(0.0),
            stop: Some(Vec::new()),
            top_logprobs: Some(50),
            ..params()
        };
        let config = request.into_gen_config(&GenerationConfig::default());
        assert_eq!(config.max_tokens, 5);
        assert_eq!(config.temperature, 0.0);
        assert!(config.stop_sequences.is_empty());
        assert_eq!(config.top_logprobs, 20);
    }
}
