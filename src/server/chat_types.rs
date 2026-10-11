//! Request and response types for the chat completions endpoint

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::generation::{HasSamplingFields, LogprobResult, ResponseFormat, SamplingParams, Usage};
use super::tools::{ChatRequestMessage, ChatResponseMessage, Tool};
use crate::config::ReasoningFormat;
use crate::model::chat_template::RenderOptions;

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
    /// LoRA adapter name to activate for this request.
    /// The adapter must be loaded beforehand via `POST /v1/lora`.
    #[serde(default)]
    pub lora_adapter: Option<String>,
}

impl ChatRequest {
    /// Template variables for this request.
    ///
    /// `think` sets `enable_thinking`, `reasoning_effort` passes through, and
    /// `chat_template_kwargs` adds any other variable the template reads.
    /// Tools reach the template only when it renders them itself.
    pub fn render_options(&self, native_tools: bool) -> RenderOptions {
        let mut options = RenderOptions::default();
        if native_tools {
            options.tools = self
                .tools
                .as_ref()
                .filter(|tools| !tools.is_empty())
                .map(|tools| {
                    tools
                        .iter()
                        .filter_map(|t| serde_json::to_value(t).ok())
                        .collect()
                });
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
        options
    }
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
            grammar: self.effective_grammar(),
            lora_adapter: self.lora_adapter.clone(),
        }
    }
}

impl ChatRequest {
    /// Resolve grammar from explicit `grammar` field or `response_format.json_schema`.
    fn effective_grammar(&self) -> Option<String> {
        if self.grammar.is_some() {
            return self.grammar.clone();
        }
        if let Some(ref rf) = self.response_format {
            if rf.r#type == "json_schema" {
                if let Some(ref schema) = rf.json_schema {
                    return crate::engine::grammar::json_schema_to_gbnf(schema).ok();
                }
            }
        }
        None
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
