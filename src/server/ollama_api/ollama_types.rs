//! Ollama request and response types for `/api/chat`, `/api/generate`,
//! `/api/embed` and `/api/embeddings`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::server::generation::string_or_array;

/// `POST /api/chat` request.
#[derive(Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    /// An empty list is a load request.
    #[serde(default)]
    pub messages: Vec<OllamaMessage>,
    /// Tool definitions in the OpenAI shape.
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub format: Option<OllamaFormat>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    /// Absent means true.
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub think: Option<OllamaThink>,
    #[serde(default)]
    pub keep_alive: Option<OllamaKeepAlive>,
}

/// `POST /api/generate` request.
#[derive(Deserialize)]
pub struct OllamaGenerateRequest {
    pub model: String,
    /// An empty or absent prompt is a load request.
    #[serde(default)]
    pub prompt: Option<String>,
    /// The code after the gap: a fill-in-the-middle request.
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub system: Option<String>,
    /// A built-in chat template name. Go templates are refused.
    #[serde(default)]
    pub template: Option<String>,
    /// Token ids of an earlier reply's `context`.
    #[serde(default)]
    pub context: Option<Vec<i64>>,
    #[serde(default)]
    pub raw: Option<bool>,
    /// Base64 images.
    #[serde(default)]
    pub images: Option<Vec<String>>,
    #[serde(default)]
    pub format: Option<OllamaFormat>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    #[serde(default)]
    pub think: Option<OllamaThink>,
    #[serde(default)]
    pub keep_alive: Option<OllamaKeepAlive>,
    /// Absent means true.
    #[serde(default)]
    pub stream: Option<bool>,
}

/// One chat message.
#[derive(Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    /// Base64 images.
    #[serde(default)]
    pub images: Option<Vec<String>>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    /// The tool a `tool` message answers.
    #[serde(default)]
    pub tool_name: Option<String>,
    /// The reasoning of an earlier assistant turn.
    #[serde(default)]
    pub thinking: Option<String>,
}

/// A tool call, in a request message or a reply.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct OllamaToolCall {
    /// Sent by some clients. A reply leaves it out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: OllamaToolCallFunction,
}

/// The function of a tool call. `arguments` is a JSON object.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct OllamaToolCallFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

/// `format`: `"json"` or a JSON schema object.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum OllamaFormat {
    Name(String),
    Schema(Map<String, Value>),
}

/// `think`: a flag, or an effort level `low`, `medium` or `high`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum OllamaThink {
    Flag(bool),
    Effort(String),
}

/// `keep_alive`: a duration string such as `"5m"`, or a number of seconds.
/// A negative value keeps the model forever, zero unloads it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum OllamaKeepAlive {
    Secs(f64),
    Str(String),
}

impl OllamaKeepAlive {
    /// The value in the form `parse_keep_alive` reads: a negative value is
    /// `"-1"`, a number is whole seconds rounded up.
    pub fn to_arg(&self) -> String {
        match self {
            Self::Str(s) if s.trim().starts_with('-') => "-1".to_string(),
            Self::Str(s) => s.trim().to_string(),
            Self::Secs(n) if *n < 0.0 => "-1".to_string(),
            Self::Secs(n) => format!("{}", n.ceil() as u64),
        }
    }
}

/// `options`: the sampling keys blazr maps. Every other key lands in
/// `ignored`.
#[derive(Debug, Default, Deserialize)]
pub struct OllamaOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub mirostat: Option<u8>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    #[serde(default, deserialize_with = "string_or_array")]
    pub stop: Option<Vec<String>>,
    pub num_predict: Option<i64>,
    pub num_ctx: Option<usize>,
    #[serde(flatten)]
    pub ignored: Map<String, Value>,
}

/// The durations, in nanoseconds, and token counts of a finished reply.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct OllamaTimings {
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: usize,
    pub prompt_eval_duration: u64,
    pub eval_count: usize,
    pub eval_duration: u64,
}

/// `input` of `/api/embed`: one string or a list.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum OllamaInput {
    One(String),
    Many(Vec<String>),
}

impl OllamaInput {
    /// The inputs, in order.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

/// `POST /api/embed` request.
#[derive(Deserialize)]
pub struct OllamaEmbedRequest {
    pub model: String,
    #[serde(default)]
    pub input: Option<OllamaInput>,
    /// Absent means true.
    #[serde(default)]
    pub truncate: Option<bool>,
    #[serde(default)]
    pub options: Option<Map<String, Value>>,
    #[serde(default)]
    pub keep_alive: Option<OllamaKeepAlive>,
    #[serde(default)]
    pub dimensions: Option<usize>,
}

/// `POST /api/embeddings` request, the legacy single-prompt form.
#[derive(Deserialize)]
pub struct OllamaEmbeddingsRequest {
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub options: Option<Map<String, Value>>,
    #[serde(default)]
    pub keep_alive: Option<OllamaKeepAlive>,
}

/// `POST /api/embed` reply.
#[derive(Serialize)]
pub struct OllamaEmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: usize,
}

/// `POST /api/embeddings` reply. A typed body keeps each value the
/// shortest decimal of its `f32`, as `/api/embed` writes it.
#[derive(Serialize)]
pub struct OllamaEmbeddingsResponse {
    pub embedding: Vec<f32>,
}

/// The current time in RFC 3339 with nanoseconds, the `created_at` form.
pub fn created_at() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat(body: Value) -> OllamaChatRequest {
        serde_json::from_value(body).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn stream_defaults_to_true() {
        let req = chat(json!({"model": "m", "messages": []}));
        assert_eq!(req.stream, None);
        let off = chat(json!({"model": "m", "stream": false}));
        assert_eq!(off.stream, Some(false));
    }

    #[test]
    fn keep_alive_reads_strings_and_numbers() {
        let cases = [
            (json!("5m"), "5m"),
            (json!(300), "300"),
            (json!(0), "0"),
            (json!(-1), "-1"),
            (json!(-5), "-1"),
            (json!("-5m"), "-1"),
            (json!(1.5), "2"),
        ];
        for (value, arg) in cases {
            let ka: OllamaKeepAlive = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(ka.to_arg(), arg, "{value}");
        }
    }

    #[test]
    fn format_reads_a_name_or_a_schema() {
        let name: OllamaFormat = serde_json::from_value(json!("json")).unwrap();
        assert_eq!(name, OllamaFormat::Name("json".into()));
        let schema: OllamaFormat = serde_json::from_value(json!({"type": "object"})).unwrap();
        assert!(matches!(schema, OllamaFormat::Schema(m) if m["type"] == "object"));
    }

    #[test]
    fn think_reads_a_flag_or_an_effort() {
        let flag: OllamaThink = serde_json::from_value(json!(true)).unwrap();
        assert_eq!(flag, OllamaThink::Flag(true));
        let effort: OllamaThink = serde_json::from_value(json!("high")).unwrap();
        assert_eq!(effort, OllamaThink::Effort("high".into()));
    }

    #[test]
    fn assistant_tool_calls_carry_object_arguments() {
        let req = chat(json!({"model": "m", "messages": [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Paris"}}}],
        }]}));
        let calls = req.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[0].function.arguments, json!({"city": "Paris"}));
        let reply = serde_json::to_value(&calls[0]).unwrap();
        assert_eq!(
            reply,
            json!({"function": {"name": "weather", "arguments": {"city": "Paris"}}})
        );
    }

    #[test]
    fn embed_input_reads_a_string_or_a_list() {
        let one: OllamaInput = serde_json::from_value(json!("a")).unwrap();
        assert_eq!(one.into_vec(), ["a"]);
        let many: OllamaInput = serde_json::from_value(json!(["a", "b"])).unwrap();
        assert_eq!(many.into_vec(), ["a", "b"]);
    }

    #[test]
    fn options_keep_unmapped_keys_apart() {
        let opts: OllamaOptions =
            serde_json::from_value(json!({"temperature": 0.5, "stop": "x", "num_gpu": 1})).unwrap();
        assert_eq!(opts.temperature, Some(0.5));
        assert_eq!(opts.stop, Some(vec!["x".to_string()]));
        assert_eq!(opts.ignored.keys().collect::<Vec<_>>(), ["num_gpu"]);
    }

    #[test]
    fn created_at_has_nanoseconds() {
        let at = created_at();
        let parsed = chrono::DateTime::parse_from_rfc3339(&at).unwrap();
        assert!(at.ends_with('Z'), "{at}");
        assert_eq!(at.split('.').nth(1).map(|f| f.len()), Some(10), "{at}");
        assert!(parsed.timestamp() > 0);
    }
}
