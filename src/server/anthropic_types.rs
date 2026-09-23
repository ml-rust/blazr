//! Request and response types for the Anthropic Messages API.

use serde::{Deserialize, Serialize};

use super::anthropic_media::AnthropicImageSource;
use super::generation::{HasSamplingFields, SamplingParams};
use super::reasoning::split_reasoning_for_format;
use crate::config::ReasoningFormat;

/// Anthropic Messages API request
#[derive(Deserialize)]
pub struct AnthropicRequest {
    /// Model name
    pub model: String,
    /// Max tokens to generate (required in Anthropic API)
    pub max_tokens: usize,
    /// Message array
    pub messages: Vec<AnthropicMessage>,
    /// System prompt (string or array of content blocks)
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    /// Stop sequences
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    /// Whether to stream
    #[serde(default)]
    pub stream: Option<bool>,
    /// Temperature (0.0 - 1.0)
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Metadata
    #[serde(default)]
    #[allow(dead_code)]
    pub metadata: Option<serde_json::Value>,
}

/// Anthropic system can be a string or array of content blocks
#[derive(Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

impl AnthropicSystem {
    pub(super) fn text(&self) -> String {
        match self {
            AnthropicSystem::Text(s) => s.clone(),
            AnthropicSystem::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    AnthropicContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// Anthropic message
#[derive(Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    /// Content: string or array of content blocks
    pub content: AnthropicContent,
}

/// Content can be a string or array of content blocks
#[derive(Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// Anthropic content block
#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
}

impl HasSamplingFields for AnthropicRequest {
    fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            max_tokens: Some(self.max_tokens),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: self.stop_sequences.clone(),
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
}

/// Anthropic Messages API response
#[derive(Serialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub r#type: &'static str,
    pub role: &'static str,
    pub content: Vec<AnthropicResponseBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// A block of the assistant's reply.
#[derive(Serialize)]
#[serde(tag = "type")]
pub enum AnthropicResponseBlock {
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    #[serde(rename = "text")]
    Text { text: String },
}

/// Response blocks for one generation: a `thinking` block when the model
/// produced reasoning, then the `text` block.
pub(super) fn response_blocks(
    prompt: &str,
    text: String,
    format: ReasoningFormat,
) -> Vec<AnthropicResponseBlock> {
    let (reasoning, content) = split_reasoning_for_format(format, prompt, text);
    let mut blocks = Vec::with_capacity(2);
    if let Some(thinking) = reasoning {
        blocks.push(AnthropicResponseBlock::Thinking { thinking });
    }
    blocks.push(AnthropicResponseBlock::Text { text: content });
    blocks
}

#[derive(Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Anthropic token count response
#[derive(Serialize)]
pub struct TokenCountResponse {
    pub input_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::chat_vision::marked_text;

    #[test]
    fn test_anthropic_content_text() {
        let json = r#""hello world""#;
        let content: AnthropicContent = serde_json::from_str(json).unwrap();
        assert_eq!(marked_text(&content.media_parts(), None), "hello world");
    }

    #[test]
    fn test_anthropic_content_blocks() {
        let json = r#"[{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]"#;
        let content: AnthropicContent = serde_json::from_str(json).unwrap();
        assert_eq!(marked_text(&content.media_parts(), None), "hello\nworld");
    }

    #[test]
    fn test_anthropic_system_text() {
        let json = r#""You are helpful""#;
        let sys: AnthropicSystem = serde_json::from_str(json).unwrap();
        assert_eq!(sys.text(), "You are helpful");
    }

    #[test]
    fn test_anthropic_system_blocks() {
        let json = r#"[{"type": "text", "text": "You are helpful"}]"#;
        let sys: AnthropicSystem = serde_json::from_str(json).unwrap();
        assert_eq!(sys.text(), "You are helpful");
    }

    #[test]
    fn test_anthropic_request_deserialization() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.messages.len(), 1);
        assert_eq!(
            marked_text(&req.messages[0].content.media_parts(), None),
            "Hello"
        );
    }

    #[test]
    fn test_anthropic_request_with_system() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "system": "Be concise",
            "messages": [
                {"role": "user", "content": "Hi"}
            ]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system.unwrap().text(), "Be concise");
    }

    #[test]
    fn test_anthropic_request_with_content_blocks() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]}
            ]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            marked_text(&req.messages[0].content.media_parts(), None),
            "Hello\nWorld"
        );
    }

    #[test]
    fn test_anthropic_response_serialization() {
        let resp = AnthropicResponse {
            id: "msg_123".to_string(),
            r#type: "message",
            role: "assistant",
            content: vec![AnthropicResponseBlock::Text {
                text: "Hello!".to_string(),
            }],
            model: "test".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["usage"]["input_tokens"], 10);
    }

    #[test]
    fn test_response_blocks_split_reasoning() {
        let prompt = "<|im_start|>assistant\n<think>\n";
        let blocks = response_blocks(
            prompt,
            "why\n</think>\n\nbecause".to_string(),
            ReasoningFormat::Deepseek,
        );
        let json = serde_json::to_value(&blocks).unwrap();
        assert_eq!(json[0]["type"], "thinking");
        assert_eq!(json[0]["thinking"], "why\n");
        assert_eq!(json[1]["type"], "text");
        assert_eq!(json[1]["text"], "because");

        let raw = response_blocks(prompt, "plain".to_string(), ReasoningFormat::None);
        let json = serde_json::to_value(&raw).unwrap();
        assert_eq!(json.as_array().map(Vec::len), Some(1));
        assert_eq!(json[0]["text"], "plain");
    }
}
