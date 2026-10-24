//! Request and response types for the text completions endpoint.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::completion_prompt::CompletionPrompt;
use crate::server::generation::{
    check_stop, string_or_array, validate_generation_params, HasSamplingFields, LogprobResult,
    LogprobsRequest, ResponseFormat, SamplingParams, StreamOptions, Usage,
};

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    /// A string, strings, token ids, or several token id lists.
    pub prompt: CompletionPrompt,
    /// Special tokens the caller permits `prompt` to spell out, tiktoken-style.
    ///
    /// Absent means none. A `prompt` containing `"<|im_start|>"` then encodes
    /// the literal string, not the control-token id.
    /// A caller names each control token it needs.
    /// Any other special token in the prompt is refused with a 400.
    #[serde(default)]
    pub allowed_special: Option<Vec<String>>,
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
    /// `include_usage` adds a usage chunk to a stream. See [`StreamOptions`].
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// One string or up to four.
    #[serde(default, deserialize_with = "string_or_array")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub n: Option<usize>,
    /// Text after the completion. The model fills the gap between `prompt` and `suffix`.
    #[serde(default)]
    pub suffix: Option<String>,
    /// Accepted when it does not exceed `n`. A larger value is a 400.
    #[serde(default)]
    pub best_of: Option<usize>,
    #[serde(default)]
    pub context: Option<Vec<i64>>,
    #[serde(default)]
    pub keep_alive: Option<String>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,
    /// A boolean, or an integer 0 to 5 for the top logprobs count.
    #[serde(default)]
    pub logprobs: Option<LogprobsRequest>,
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
    pub grammar: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    /// The registered LoRA adapter of `model` to run on. See `ChatRequest`.
    #[serde(default)]
    pub lora_adapter: Option<String>,
}

impl CompletionRequest {
    /// Choices per prompt: `n`, at least 1.
    pub fn choices_per_prompt(&self) -> usize {
        self.n.unwrap_or(1).max(1)
    }

    /// Whether a stream ends with a usage chunk: `stream_options.include_usage`.
    pub fn include_usage(&self) -> bool {
        StreamOptions::include_usage(self.stream_options.as_ref())
    }

    /// Check the fields that need no model. The handler answers 400 on an error.
    pub fn check_fields(&self) -> Result<(), String> {
        validate_generation_params(self.temperature, self.top_p, self.max_tokens)?;
        check_stop(self.stop.as_deref())?;
        if let Some(logprobs) = self.logprobs {
            logprobs.check()?;
        }
        let n = self.choices_per_prompt();
        if self.best_of.is_some_and(|b| b > n) {
            return Err("best_of is not supported. Omit it or set it to at most n".to_string());
        }
        if self.stream.unwrap_or(false) {
            if n > 1 {
                return Err("streaming supports n = 1".to_string());
            }
            if self.prompt.items().len() > 1 {
                return Err("streaming supports one prompt".to_string());
            }
        }
        Ok(())
    }
}

impl HasSamplingFields for CompletionRequest {
    fn sampling_params(&self) -> SamplingParams {
        let logprobs = self.logprobs.map(LogprobsRequest::resolve);
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
            logprobs: logprobs.map(|(on, _)| on),
            top_logprobs: logprobs.and_then(|(_, top)| top).or(self.top_logprobs),
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
            lora_adapter: self.lora_adapter.clone(),
        }
    }
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobResult>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn request(extra: Value) -> CompletionRequest {
        let mut body = json!({"model": "m", "prompt": "hi"});
        if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
            body.extend(extra.clone());
        }
        serde_json::from_value(body).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn stop_reads_as_a_string_or_an_array() {
        assert_eq!(
            request(json!({"stop": "END"})).stop,
            Some(vec!["END".to_string()])
        );
        assert_eq!(
            request(json!({"stop": ["a", "b"]})).stop,
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn more_than_four_stop_sequences_is_an_error() {
        let five = json!({"stop": ["1", "2", "3", "4", "5"]});
        assert!(request(five).check_fields().is_err());
        let four = json!({"stop": ["1", "2", "3", "4"]});
        assert!(request(four).check_fields().is_ok());
    }

    #[test]
    fn logprobs_integer_sets_the_top_count() {
        let params = request(json!({"logprobs": 3})).sampling_params();
        assert_eq!(params.logprobs, Some(true));
        assert_eq!(params.top_logprobs, Some(3));
    }

    #[test]
    fn logprobs_boolean_keeps_the_top_logprobs_field() {
        let params = request(json!({"logprobs": true, "top_logprobs": 7})).sampling_params();
        assert_eq!(params.logprobs, Some(true));
        assert_eq!(params.top_logprobs, Some(7));
        let off = request(json!({"logprobs": false})).sampling_params();
        assert_eq!(off.logprobs, Some(false));
        assert_eq!(request(json!({})).sampling_params().logprobs, None);
    }

    #[test]
    fn logprobs_out_of_range_is_an_error() {
        assert!(request(json!({"logprobs": 6})).check_fields().is_err());
        assert!(request(json!({"logprobs": -1})).check_fields().is_err());
        assert!(request(json!({"logprobs": 5})).check_fields().is_ok());
    }

    #[test]
    fn best_of_above_n_is_an_error() {
        let err = request(json!({"best_of": 2})).check_fields().unwrap_err();
        assert!(err.contains("best_of is not supported"));
        assert!(request(json!({"best_of": 2, "n": 1}))
            .check_fields()
            .is_err());
        assert!(request(json!({"best_of": 2, "n": 2}))
            .check_fields()
            .is_ok());
        assert!(request(json!({"best_of": 1})).check_fields().is_ok());
        assert!(request(json!({})).check_fields().is_ok());
    }

    #[test]
    fn streaming_refuses_n_above_one_and_several_prompts() {
        let many_n = request(json!({"stream": true, "n": 2}));
        assert!(many_n.check_fields().unwrap_err().contains("n = 1"));
        let many_prompts = request(json!({"stream": true, "prompt": ["a", "b"]}));
        assert!(many_prompts
            .check_fields()
            .unwrap_err()
            .contains("one prompt"));
        assert!(request(json!({"stream": true})).check_fields().is_ok());
        assert!(request(json!({"n": 2, "prompt": ["a", "b"]}))
            .check_fields()
            .is_ok());
    }

    #[test]
    fn stream_options_sets_include_usage() {
        let on = request(json!({"stream": true, "stream_options": {"include_usage": true}}));
        assert!(on.include_usage());
        assert!(!request(json!({"stream": true})).include_usage());
        // A non-streaming request accepts the field and ignores it.
        let unstreamed = request(json!({"stream_options": {"include_usage": true}}));
        assert!(unstreamed.check_fields().is_ok());
    }

    #[test]
    fn every_prompt_shape_deserializes() {
        for prompt in [
            json!("text"),
            json!(["a", "b"]),
            json!([1, 2, 3]),
            json!([[1, 2], [3]]),
        ] {
            let parsed = request(json!({ "prompt": prompt }));
            assert!(parsed.check_fields().is_ok());
        }
        assert_eq!(
            request(json!({"prompt": [[1], [2]]})).prompt.items().len(),
            2
        );
    }
}
