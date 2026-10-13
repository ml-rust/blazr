//! Request and response types for the Responses API.

use serde::{Deserialize, Serialize};

use super::reasoning::split_reasoning_for_format;
use crate::config::ReasoningFormat;

/// OpenAI Responses API request
#[derive(Deserialize)]
pub struct ResponsesRequest {
    /// Model name
    pub model: String,
    /// Input: string prompt or array of input items
    pub input: ResponsesInput,
    /// Max output tokens. Unset takes the live `generation.max_tokens`.
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    /// Temperature
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Whether to include reasoning/thinking in output
    #[serde(default)]
    #[allow(dead_code)]
    pub reasoning: Option<ReasoningConfig>,
    /// Instructions (system prompt)
    #[serde(default)]
    pub instructions: Option<String>,
    /// Text response format
    #[serde(default)]
    #[allow(dead_code)]
    pub text: Option<TextFormat>,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    /// Metadata
    #[serde(default)]
    #[allow(dead_code)]
    pub metadata: Option<serde_json::Value>,
    /// The registered LoRA adapter of `model` to run on: the request runs
    /// on the variant `<model>:<lora_adapter>`. See `ChatRequest`.
    #[serde(default)]
    pub lora_adapter: Option<String>,
}

/// Input can be a string or array of input items
#[derive(Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

/// An input item (message)
#[derive(Deserialize)]
pub struct InputItem {
    /// Role: "user", "assistant", "system", "developer"
    pub role: String,
    /// Content: string or array of content parts
    pub content: ResponsesContent,
}

/// Content: string or array of content parts
#[derive(Deserialize)]
#[serde(untagged)]
pub enum ResponsesContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

/// Content part types
#[derive(Deserialize)]
#[serde(tag = "type")]
#[allow(clippy::enum_variant_names)]
pub enum ResponsesContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        /// A data URL (`data:image/png;base64,...`) or an `http(s)` URL.
        #[serde(default)]
        image_url: Option<String>,
        /// Requested detail level: "auto", "low", "high".
        #[serde(default)]
        #[allow(dead_code)]
        detail: Option<String>,
    },
    #[serde(rename = "input_audio")]
    InputAudio {
        #[allow(dead_code)]
        data: Option<String>,
    },
}

/// Reasoning configuration
#[derive(Deserialize)]
pub struct ReasoningConfig {
    /// Effort level: "low", "medium", "high"
    #[serde(default = "default_effort")]
    #[allow(dead_code)]
    pub effort: String,
}

fn default_effort() -> String {
    "medium".to_string()
}

/// Text format configuration
#[derive(Deserialize)]
pub struct TextFormat {
    /// "text" or "json_object" or "json_schema"
    #[serde(default = "default_text_format")]
    #[allow(dead_code)]
    pub format: TextFormatType,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum TextFormatType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema {
        #[allow(dead_code)]
        schema: Option<serde_json::Value>,
    },
}

fn default_text_format() -> TextFormatType {
    TextFormatType::Text
}

/// OpenAI Responses API response
#[derive(Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub model: String,
    pub output: Vec<OutputItem>,
    pub usage: ResponsesUsage,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    /// The model's reasoning, emitted before its message.
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        summary: Vec<ReasoningSummary>,
        status: &'static str,
    },
    #[serde(rename = "message")]
    Message {
        id: String,
        role: &'static str,
        content: Vec<OutputContent>,
        status: &'static str,
    },
}

#[derive(Serialize)]
pub struct ReasoningSummary {
    pub r#type: &'static str,
    pub text: String,
}

#[derive(Serialize)]
pub struct OutputContent {
    pub r#type: &'static str,
    pub text: String,
}

/// Output items for one generation: a reasoning item when the model
/// produced one, then the message.
pub(super) fn output_items(prompt: &str, text: String, format: ReasoningFormat) -> Vec<OutputItem> {
    let (reasoning, content) = split_reasoning_for_format(format, prompt, text);
    let mut items = Vec::with_capacity(2);
    if let Some(reasoning) = reasoning {
        items.push(OutputItem::Reasoning {
            id: format!("rs_{}", uuid::Uuid::new_v4()),
            summary: vec![ReasoningSummary {
                r#type: "summary_text",
                text: reasoning,
            }],
            status: "completed",
        });
    }
    items.push(OutputItem::Message {
        id: format!("msg_{}", uuid::Uuid::new_v4()),
        role: "assistant",
        content: vec![OutputContent {
            r#type: "output_text",
            text: content,
        }],
        status: "completed",
    });
    items
}

#[derive(Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::chat_vision::marked_text;

    #[test]
    fn test_responses_input_text() {
        let json = r#"{"model": "test", "input": "Hello"}"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("expected text input"),
        }
    }

    #[test]
    fn test_responses_input_items() {
        let json = r#"{
            "model": "test",
            "input": [
                {"role": "user", "content": "Hello"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Items(items) => {
                assert_eq!(items.len(), 1);
                assert_eq!(items[0].role, "user");
                assert_eq!(marked_text(&items[0].content.media_parts(), None), "Hello");
            }
            _ => panic!("expected items input"),
        }
    }

    #[test]
    fn test_responses_input_content_parts() {
        let json = r#"{
            "model": "test",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}, {"type": "input_text", "text": "World"}]}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Items(items) => {
                assert_eq!(
                    marked_text(&items[0].content.media_parts(), None),
                    "Hello\nWorld"
                );
            }
            _ => panic!("expected items input"),
        }
    }

    #[test]
    fn test_responses_with_instructions() {
        let json = r#"{
            "model": "test",
            "input": "Hi",
            "instructions": "You are helpful",
            "max_output_tokens": 100
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.instructions.unwrap(), "You are helpful");
        assert_eq!(req.max_output_tokens, Some(100));
    }

    #[test]
    fn test_responses_response_serialization() {
        let resp = ResponsesResponse {
            id: "resp_123".to_string(),
            object: "response",
            created_at: 1234567890,
            model: "test".to_string(),
            output: vec![OutputItem::Message {
                id: "msg_123".to_string(),
                role: "assistant",
                content: vec![OutputContent {
                    r#type: "output_text",
                    text: "Hello!".to_string(),
                }],
                status: "completed",
            }],
            usage: ResponsesUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
            },
            status: "completed",
            error: None,
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "response");
        assert_eq!(json["output"][0]["type"], "message");
        assert_eq!(json["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(json["usage"]["total_tokens"], 15);
        assert_eq!(json["status"], "completed");
    }

    #[test]
    fn test_output_items_split_reasoning() {
        let prompt = "<|im_start|>assistant\n<think>\n";
        let items = output_items(
            prompt,
            "why\n</think>\n\nbecause".to_string(),
            ReasoningFormat::Deepseek,
        );
        let json = serde_json::to_value(&items).unwrap();
        assert_eq!(json[0]["type"], "reasoning");
        assert_eq!(json[0]["summary"][0]["text"], "why\n");
        assert_eq!(json[1]["type"], "message");
        assert_eq!(json[1]["content"][0]["text"], "because");

        let raw = output_items(
            prompt,
            "why\n</think>\n\nbecause".to_string(),
            ReasoningFormat::None,
        );
        let json = serde_json::to_value(&raw).unwrap();
        assert_eq!(json.as_array().map(Vec::len), Some(1));
        assert_eq!(json[0]["content"][0]["text"], "why\n</think>\n\nbecause");
    }

    #[test]
    fn test_developer_role_mapping() {
        let json = r#"{
            "model": "test",
            "input": [
                {"role": "developer", "content": "System instructions"},
                {"role": "user", "content": "Hi"}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        match req.input {
            ResponsesInput::Items(items) => {
                assert_eq!(items[0].role, "developer");
                assert_eq!(items[1].role, "user");
            }
            _ => panic!("expected items"),
        }
    }
}
