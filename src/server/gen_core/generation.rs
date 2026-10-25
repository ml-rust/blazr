//! Shared generation logic for completion and chat handlers.
//!
//! Contains SamplingParams, metrics recording and context decoding. The
//! stop-sequence streaming functions live in `stop_stream`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::GenerationConfig;
use crate::server::metrics;

// Re-export types and utilities from gen_types so callers import from one place.
pub use super::constraint_admission::admit_generation;
pub use super::gen_types::{
    apply_keep_alive, chat_prompt_error_response, convert_logprobs, encode_with_allowed_special,
    error_response, error_response_with_code, overloaded_response, policy_error_response,
    resolve_constraint, validate_generation_params, LogprobResult, ResponseFormat, Usage,
};
pub use super::request_fields::{
    check_choice_count, check_stop, choice_count, string_or_array, LogprobsRequest, StreamOptions,
};
pub use super::stop_stream::{
    stream_images_with_stop_sequences, stream_multimodal_with_stop_sequences,
    stream_with_stop_sequences,
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
    /// Recent tokens the repetition penalty covers.
    pub repeat_last_n: Option<usize>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<u64>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<usize>,
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
            // The handler resolves the request's constraint at admission.
            constraint: d.constraint,
            compiled: None,
            lora_adapter: self.lora_adapter.or(d.lora_adapter),
            repeat_last_n: self.repeat_last_n.unwrap_or(d.repeat_last_n),
            verbose_prompt: d.verbose_prompt,
        }
    }
}

/// The sampling config of choice `i`: `seed + i` when the request sets a
/// seed. The streamed and the
/// non-streaming paths of every endpoint sample each choice alike.
pub fn choice_config(base: &GenerationConfig, i: usize) -> GenerationConfig {
    let mut config = base.clone();
    if let Some(seed) = config.seed {
        config.seed = Some(seed.wrapping_add(i as u64));
    }
    config
}

/// Trait for request types that carry sampling fields.
/// Eliminates the duplicated SamplingParams { ... } construction.
pub trait HasSamplingFields {
    fn sampling_params(&self) -> SamplingParams;
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
            repeat_last_n: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            seed: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
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

    #[test]
    fn each_choice_offsets_the_seed_by_its_index() {
        let seeded = GenerationConfig {
            seed: Some(7),
            ..Default::default()
        };
        assert_eq!(choice_config(&seeded, 0).seed, Some(7));
        assert_eq!(choice_config(&seeded, 3).seed, Some(10));
        assert_eq!(choice_config(&GenerationConfig::default(), 3).seed, None);
    }

    #[test]
    fn repeat_last_n_takes_the_request_value_else_the_default() {
        let defaults = GenerationConfig {
            repeat_last_n: 9,
            ..Default::default()
        };
        let set = SamplingParams {
            repeat_last_n: Some(3),
            ..params()
        };
        assert_eq!(set.into_gen_config(&defaults).repeat_last_n, 3);
        assert_eq!(params().into_gen_config(&defaults).repeat_last_n, 9);
    }
}
