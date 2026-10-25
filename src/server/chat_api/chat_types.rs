//! Request and response types for the chat completions endpoint

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::tool_choice::ToolChoice;
use super::tool_constraint::chat_constraint;
use super::tools::{ChatRequestMessage, ChatResponseMessage, Tool};
use crate::config::{Constraint, ReasoningFormat};
use crate::model::chat_template::RenderOptions;
use crate::server::generation::{
    check_stop, resolve_constraint, string_or_array, validate_generation_params, HasSamplingFields,
    LogprobResult, ResponseFormat, SamplingParams, StreamOptions, Usage,
};

// ── Request/Response types ──

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatRequestMessage>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// `none`, `auto`, `required`, or a named function. See [`ToolChoice`].
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Alias of `max_tokens`. It wins when both are present.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
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
    /// Recent tokens the repetition penalty covers.
    #[serde(default)]
    pub repeat_last_n: Option<usize>,
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
    pub raw: Option<bool>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub context: Option<Vec<i64>>,
    #[serde(default)]
    pub think: Option<bool>,
    /// Template variable `reasoning_effort`, for templates that read one.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Extra template variables, passed through to a Jinja template as is.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// `deepseek` splits reasoning from the answer, `none` returns raw output.
    /// Default: the server's `--reasoning-format`.
    #[serde(default)]
    pub reasoning_format: Option<ReasoningFormat>,
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
    pub grammar: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
    /// The registered LoRA adapter of `model` to run on: the request runs
    /// on the variant `<model>:<lora_adapter>`, registered by `--lora` or
    /// `POST /v1/lora`. An unknown name is a 404. With `model` naming a
    /// variant, both must name the same adapter.
    #[serde(default)]
    pub lora_adapter: Option<String>,
    /// The code after the gap of a fill-in-the-middle request. With `raw`
    /// set, the joined message text is the prefix and the prompt takes the
    /// tokenizer's FIM markers. Set by a dialect adapter, never from JSON.
    #[serde(skip)]
    pub fim_suffix: Option<String>,
}

impl ChatRequest {
    /// The token limit: `max_completion_tokens` when set, else `max_tokens`.
    pub fn effective_max_tokens(&self) -> Option<usize> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    /// Whether a stream ends with a usage chunk: `stream_options.include_usage`.
    pub fn include_usage(&self) -> bool {
        StreamOptions::include_usage(self.stream_options.as_ref())
    }

    /// The number of choices: `n`, at least 1.
    pub fn choice_count(&self) -> usize {
        self.n.unwrap_or(1).max(1)
    }

    /// Check the sampling fields and `stop`. The handler answers 400 on an error.
    pub fn check_fields(&self) -> Result<(), String> {
        validate_generation_params(self.temperature, self.top_p, self.effective_max_tokens())?;
        check_stop(self.stop.as_deref())
    }

    /// Template variables for this request.
    ///
    /// `think` sets `enable_thinking`, `reasoning_effort` passes through, and
    /// `chat_template_kwargs` adds any other variable the template reads.
    /// Tools reach the template only when it renders them itself. `tools` is
    /// the list the model sees: [`ToolChoice::visible`].
    ///
    /// # Errors
    ///
    /// A tool that does not serialize is an error naming the tool.
    /// The handler answers 400.
    pub fn render_options(
        &self,
        native_tools: bool,
        tools: &[Tool],
    ) -> Result<RenderOptions, String> {
        let mut options = RenderOptions::default();
        if native_tools && !tools.is_empty() {
            let rendered = tools
                .iter()
                .map(|t| {
                    serde_json::to_value(t).map_err(|e| {
                        format!(
                            "tool {:?} does not serialize: {e}. Correct the tool definition",
                            t.function.name
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            options.tools = Some(rendered);
        }
        if let Some(kwargs) = &self.chat_template_kwargs {
            options
                .extra
                .extend(kwargs.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        if let Some(think) = self.think {
            options
                .extra
                .insert("enable_thinking".to_string(), think.into());
        }
        if let Some(effort) = &self.reasoning_effort {
            options
                .extra
                .insert("reasoning_effort".to_string(), effort.clone().into());
        }
        Ok(options)
    }
}

impl HasSamplingFields for ChatRequest {
    fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            max_tokens: self.effective_max_tokens(),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            stop: self.stop.clone(),
            seed: self.seed,
            logit_bias: self.logit_bias.clone(),
            logprobs: self.logprobs,
            top_logprobs: self.top_logprobs,
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

impl ChatRequest {
    /// The constraint `grammar`, `response_format` and a forced `choice` ask for.
    /// See [`resolve_constraint`] and [`chat_constraint`].
    pub fn constraint(&self, choice: &ToolChoice) -> Result<Option<Constraint>, String> {
        chat_constraint(
            choice,
            self.tools.as_deref().unwrap_or(&[]),
            self.grammar.as_deref(),
            self.response_format.as_ref(),
            resolve_constraint(self.grammar.as_deref(), self.response_format.as_ref()),
        )
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

#[cfg(test)]
mod tests {
    use super::super::tool_choice::ToolMode;
    use super::*;
    use serde_json::json;

    fn request(extra: serde_json::Value) -> ChatRequest {
        let mut body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "weather"}}],
        });
        if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
            body.extend(extra.clone());
        }
        serde_json::from_value(body).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn repeat_last_n_reaches_the_sampling_params() {
        assert_eq!(request(json!({})).sampling_params().repeat_last_n, None);
        let set = request(json!({"repeat_last_n": 16}));
        assert_eq!(set.sampling_params().repeat_last_n, Some(16));
    }

    #[test]
    fn max_completion_tokens_aliases_max_tokens() {
        assert_eq!(request(json!({})).effective_max_tokens(), None);
        let only_old = request(json!({"max_tokens": 7}));
        assert_eq!(only_old.effective_max_tokens(), Some(7));
        let only_new = request(json!({"max_completion_tokens": 9}));
        assert_eq!(only_new.effective_max_tokens(), Some(9));
        assert_eq!(only_new.sampling_params().max_tokens, Some(9));
    }

    #[test]
    fn max_completion_tokens_wins_when_both_are_present() {
        let both = request(json!({"max_tokens": 7, "max_completion_tokens": 9}));
        assert_eq!(both.effective_max_tokens(), Some(9));
        assert_eq!(both.sampling_params().max_tokens, Some(9));
    }

    #[test]
    fn zero_max_completion_tokens_is_an_error() {
        let req = request(json!({"max_tokens": 5, "max_completion_tokens": 0}));
        assert!(req.check_fields().is_err());
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
        assert_eq!(request(json!({})).stop, None);
    }

    #[test]
    fn more_than_four_stop_sequences_is_an_error() {
        let five = request(json!({"stop": ["1", "2", "3", "4", "5"]}));
        assert!(five.check_fields().is_err());
        let four = request(json!({"stop": ["1", "2", "3", "4"]}));
        assert!(four.check_fields().is_ok());
    }

    #[test]
    fn streaming_accepts_any_choice_count() {
        assert!(request(json!({"stream": true, "n": 2}))
            .check_fields()
            .is_ok());
        assert_eq!(request(json!({"stream": true, "n": 3})).choice_count(), 3);
        assert_eq!(request(json!({"n": 0})).choice_count(), 1);
        assert_eq!(request(json!({})).choice_count(), 1);
    }

    #[test]
    fn stream_options_sets_include_usage() {
        let on = request(json!({"stream": true, "stream_options": {"include_usage": true}}));
        assert!(on.include_usage());
        assert!(!request(json!({"stream": true})).include_usage());
        let off = request(json!({"stream_options": {"include_usage": false}}));
        assert!(!off.include_usage());
        // A non-streaming request accepts the field and ignores it.
        let unstreamed = request(json!({"stream_options": {"include_usage": true}}));
        assert!(unstreamed.check_fields().is_ok());
    }

    #[test]
    fn tool_choice_reads_as_a_json_value_in_every_form() {
        for value in [
            json!("none"),
            json!("auto"),
            json!("required"),
            json!({"type": "function", "function": {"name": "weather"}}),
        ] {
            let parsed = request(json!({ "tool_choice": value.clone() }));
            assert_eq!(parsed.tool_choice, Some(value));
        }
        assert!(request(json!({})).tool_choice.is_none());
    }

    #[test]
    fn hidden_tools_do_not_reach_the_template() {
        let req = request(json!({}));
        let tools = req.tools.clone().unwrap_or_default();
        let shown = req.render_options(true, &tools).unwrap();
        assert_eq!(shown.tools.map(|t| t.len()), Some(1));
        let none = ToolChoice::Mode(ToolMode::None);
        let hidden = req.render_options(true, none.visible(&tools)).unwrap();
        assert!(hidden.tools.is_none());
    }

    #[test]
    fn a_forced_choice_sets_the_constraint() {
        let req = request(json!({}));
        let required = ToolChoice::Mode(ToolMode::Required);
        assert!(matches!(
            req.constraint(&required),
            Ok(Some(Constraint::JsonSchema(_)))
        ));
        let auto = ToolChoice::Mode(ToolMode::Auto);
        assert_eq!(req.constraint(&auto), Ok(None));
    }
}
