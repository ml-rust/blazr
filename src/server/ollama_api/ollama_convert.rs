//! Ollama requests as the chat request the OpenAI route runs.
//!
//! The conversion builds the chat request's JSON and deserializes it, so
//! the chat request's own checks apply unchanged.

use serde_json::{json, Map, Value};

use super::ollama_types::{
    OllamaChatRequest, OllamaFormat, OllamaGenerateRequest, OllamaKeepAlive, OllamaMessage,
    OllamaOptions, OllamaThink,
};
use crate::model::chat_template::{ChatTemplate, BUILTIN_NAMES};
use crate::server::chat_types::ChatRequest;
use crate::server::encoding::base64_decode;
use crate::server::multimodal::image_mime;

/// The base64 characters that cover the first 12 bytes, enough to sniff a WebP.
const SNIFF_CHARS: usize = 16;

/// The fields every Ollama generation request shares.
struct Shared<'a> {
    format: Option<&'a OllamaFormat>,
    options: Option<&'a OllamaOptions>,
    think: Option<&'a OllamaThink>,
    keep_alive: Option<&'a OllamaKeepAlive>,
}

impl Shared<'_> {
    /// Write the shared fields into `fields`.
    fn apply(&self, fields: &mut Map<String, Value>) -> Result<(), String> {
        // Reasoning always splits out, so thinking never lands in `content`.
        fields.insert("reasoning_format".into(), "deepseek".into());
        if let Some(format) = response_format(self.format)? {
            fields.insert("response_format".into(), format);
        }
        if let Some(options) = self.options {
            options.apply(fields)?;
        }
        match self.think {
            None => {}
            Some(OllamaThink::Flag(on)) => {
                fields.insert("think".into(), (*on).into());
            }
            Some(OllamaThink::Effort(effort)) => {
                if !matches!(effort.as_str(), "low" | "medium" | "high") {
                    return Err(format!(
                        "think must be true, false, \"low\", \"medium\" or \"high\", got {effort:?}"
                    ));
                }
                fields.insert("think".into(), true.into());
                fields.insert("reasoning_effort".into(), effort.clone().into());
            }
        }
        if let Some(keep_alive) = self.keep_alive {
            fields.insert("keep_alive".into(), keep_alive.to_arg().into());
        }
        Ok(())
    }
}

/// The OpenAI `response_format` of `format`.
fn response_format(format: Option<&OllamaFormat>) -> Result<Option<Value>, String> {
    match format {
        None => Ok(None),
        Some(OllamaFormat::Name(name)) if name.is_empty() => Ok(None),
        Some(OllamaFormat::Name(name)) if name == "json" => {
            Ok(Some(json!({"type": "json_object"})))
        }
        Some(OllamaFormat::Name(name)) => Err(format!(
            "format must be \"json\" or a JSON schema object, got {name:?}"
        )),
        Some(OllamaFormat::Schema(schema)) => Ok(Some(json!({
            "type": "json_schema",
            "json_schema": {"name": "format", "schema": schema},
        }))),
    }
}

/// The `image_url` of the base64 image `image`, the `index`th of its
/// message, as a data URI of its sniffed type. A data URI passes through.
fn image_url(image: &str, index: usize) -> Result<String, String> {
    let image = image.trim();
    if image.starts_with("data:") {
        return Ok(image.to_string());
    }
    let head: String = image
        .chars()
        .filter(|c| !c.is_whitespace())
        .take(SNIFF_CHARS)
        .collect();
    let mime = base64_decode(&head)
        .ok()
        .and_then(|bytes| image_mime(&bytes))
        .ok_or_else(|| format!("image {index} is not base64 of a PNG, JPEG, WebP or GIF image"))?;
    Ok(format!("data:{mime};base64,{image}"))
}

/// The chat `content` of `text` and `images`: the text alone, or the
/// images followed by the text as content parts.
fn content(text: &str, images: &[String]) -> Result<Value, String> {
    if images.is_empty() {
        return Ok(text.into());
    }
    let mut parts = images
        .iter()
        .enumerate()
        .map(|(i, image)| {
            Ok(json!({"type": "image_url", "image_url": {"url": image_url(image, i)?}}))
        })
        .collect::<Result<Vec<_>, String>>()?;
    if !text.is_empty() {
        parts.push(json!({"type": "text", "text": text}));
    }
    Ok(Value::Array(parts))
}

/// The chat messages of `messages`.
///
/// An assistant tool call takes the id `call_<message>_<call>` unless the
/// client sent one, and its object arguments become a JSON string. A
/// `tool` message answers the earliest open call to its `tool_name`.
fn messages(messages: &[OllamaMessage]) -> Result<Vec<Value>, String> {
    let mut open: Vec<(String, String)> = Vec::new();
    let mut out = Vec::with_capacity(messages.len());
    for (m, msg) in messages.iter().enumerate() {
        let text = msg.content.as_deref().unwrap_or("");
        let images = msg.images.as_deref().unwrap_or(&[]);
        let mut chat = Map::new();
        chat.insert("role".into(), msg.role.clone().into());
        chat.insert("content".into(), content(text, images)?);
        if let Some(thinking) = &msg.thinking {
            chat.insert("reasoning_content".into(), thinking.clone().into());
        }
        if let Some(calls) = msg.tool_calls.as_deref().filter(|c| !c.is_empty()) {
            let calls: Vec<Value> = calls
                .iter()
                .enumerate()
                .map(|(i, call)| {
                    let id = call.id.clone().unwrap_or_else(|| format!("call_{m}_{i}"));
                    open.push((id.clone(), call.function.name.clone()));
                    let arguments = match &call.function.arguments {
                        Value::String(s) => s.clone(),
                        Value::Null => "{}".to_string(),
                        other => other.to_string(),
                    };
                    json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": arguments},
                    })
                })
                .collect();
            chat.insert("tool_calls".into(), Value::Array(calls));
        }
        if msg.role == "tool" {
            if let Some(name) = &msg.tool_name {
                chat.insert("name".into(), name.clone().into());
                if let Some(at) = open.iter().position(|(_, n)| n == name) {
                    chat.insert("tool_call_id".into(), open.remove(at).0.into());
                }
            }
        }
        out.push(Value::Object(chat));
    }
    Ok(out)
}

/// `fields` as a chat request.
fn into_chat_request(fields: Map<String, Value>) -> Result<ChatRequest, String> {
    serde_json::from_value(Value::Object(fields)).map_err(|e| format!("invalid request: {e}"))
}

/// The chat request of `request`, run on the model `model`.
///
/// # Errors
///
/// A field that does not convert. The handler answers 400.
pub(super) fn chat_request(
    request: &OllamaChatRequest,
    model: &str,
) -> Result<ChatRequest, String> {
    let mut fields = Map::new();
    fields.insert("model".into(), model.into());
    fields.insert("messages".into(), messages(&request.messages)?.into());
    if let Some(tools) = &request.tools {
        fields.insert("tools".into(), tools.clone().into());
    }
    Shared {
        format: request.format.as_ref(),
        options: request.options.as_ref(),
        think: request.think.as_ref(),
        keep_alive: request.keep_alive.as_ref(),
    }
    .apply(&mut fields)?;
    into_chat_request(fields)
}

/// The chat request of `request`, run on the model `model`: one user
/// message, with `system`, `template`, `context` and `raw` as the chat
/// request reads them. A `suffix` makes it a fill-in-the-middle prompt on
/// the raw path.
///
/// # Errors
///
/// A field that does not convert, a template that is no built-in name, or
/// a `suffix` beside `raw`. The handler answers 400.
pub(super) fn generate_request(
    request: &OllamaGenerateRequest,
    model: &str,
) -> Result<ChatRequest, String> {
    let raw = request.raw.unwrap_or(false);
    let suffix = request.suffix.clone().filter(|s| !s.is_empty());
    if raw && suffix.is_some() {
        return Err("suffix cannot combine with raw. Send the prefix as prompt without raw".into());
    }
    let mut fields = Map::new();
    fields.insert("model".into(), model.into());
    let prompt = request.prompt.as_deref().unwrap_or("");
    let images = request.images.as_deref().unwrap_or(&[]);
    let message = json!({"role": "user", "content": content(prompt, images)?});
    fields.insert("messages".into(), json!([message]));
    fields.insert("raw".into(), (raw || suffix.is_some()).into());
    if let Some(system) = &request.system {
        fields.insert("system".into(), system.clone().into());
    }
    if let Some(template) = &request.template {
        if ChatTemplate::builtin(template).is_none() {
            return Err(format!(
                "template must be a built-in name ({}); Go templates are unsupported",
                BUILTIN_NAMES.join(", ")
            ));
        }
        fields.insert("template".into(), template.clone().into());
    }
    if let Some(context) = &request.context {
        fields.insert("context".into(), json!(context));
    }
    Shared {
        format: request.format.as_ref(),
        options: request.options.as_ref(),
        think: request.think.as_ref(),
        keep_alive: request.keep_alive.as_ref(),
    }
    .apply(&mut fields)?;
    let mut chat = into_chat_request(fields)?;
    chat.fim_suffix = suffix;
    Ok(chat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Constraint, ReasoningFormat};
    use crate::server::multimodal::{ContentPart, MessageContent};
    use crate::server::tool_choice::ToolChoice;

    fn chat(body: Value) -> Result<ChatRequest, String> {
        let request: OllamaChatRequest = serde_json::from_value(body).map_err(|e| e.to_string())?;
        chat_request(&request, "resolved")
    }

    fn generate(body: Value) -> Result<ChatRequest, String> {
        let request: OllamaGenerateRequest =
            serde_json::from_value(body).map_err(|e| e.to_string())?;
        generate_request(&request, "resolved")
    }

    /// The `image_url` of the first part of the first message.
    fn first_image(req: &ChatRequest) -> String {
        match &req.messages[0].content {
            Some(MessageContent::Parts(parts)) => match &parts[0] {
                ContentPart::ImageUrl { image_url } => image_url.url.clone(),
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn images_become_data_uris_of_their_sniffed_type() {
        use crate::server::encoding::base64_encode;
        let cases: [(&[u8], &str); 4] = [
            (
                &[0x89, b'P', b'N', b'G', 0, 0, 0, 0, 0, 0, 0, 0],
                "image/png",
            ),
            (
                &[0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0, 0, 0, 0],
                "image/jpeg",
            ),
            (b"RIFF\0\0\0\0WEBPVP8 ", "image/webp"),
            (b"GIF89a\0\0\0\0\0\0", "image/gif"),
        ];
        for (bytes, mime) in cases {
            let b64 = base64_encode(bytes);
            let req = chat(json!({"model": "m", "messages": [
                {"role": "user", "content": "what?", "images": [b64]},
            ]}))
            .unwrap();
            assert_eq!(first_image(&req), format!("data:{mime};base64,{b64}"));
        }
        let bad = chat(json!({"model": "m", "messages": [
            {"role": "user", "content": "x", "images": [base64_encode(b"plain text here")]},
        ]}));
        assert!(bad.err().unwrap_or_default().contains("image 0"));
    }

    #[test]
    fn tool_calls_round_trip_with_ids_and_string_arguments() {
        let req = chat(json!({"model": "m", "messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "weather", "arguments": {"city": "Paris"}}},
            ]},
            {"role": "tool", "content": "20C", "tool_name": "weather"},
        ], "tools": [{"type": "function", "function": {"name": "weather"}}]}))
        .unwrap();
        let call = &req.messages[1].tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.id, "call_1_0");
        assert_eq!(call.function.arguments, r#"{"city":"Paris"}"#);
        let tool = &req.messages[2];
        assert_eq!(tool.name.as_deref(), Some("weather"));
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_1_0"));
        assert_eq!(req.tools.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn format_sets_the_constraint() {
        let none = ToolChoice::parse(None, None).unwrap();
        let json = chat(json!({"model": "m", "messages": [], "format": "json"})).unwrap();
        assert_eq!(json.constraint(&none), Ok(Some(Constraint::JsonObject)));
        let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let typed = chat(json!({"model": "m", "messages": [], "format": schema.clone()})).unwrap();
        assert_eq!(
            typed.constraint(&none),
            Ok(Some(Constraint::JsonSchema(schema)))
        );
        assert!(chat(json!({"model": "m", "format": "yaml"})).is_err());
    }

    #[test]
    fn think_sets_the_flag_and_the_effort() {
        let on = chat(json!({"model": "m", "think": true})).unwrap();
        assert_eq!((on.think, on.reasoning_effort), (Some(true), None));
        let high = chat(json!({"model": "m", "think": "high"})).unwrap();
        assert_eq!(high.think, Some(true));
        assert_eq!(high.reasoning_effort.as_deref(), Some("high"));
        assert!(chat(json!({"model": "m", "think": "max"})).is_err());
        assert_eq!(on.reasoning_format, Some(ReasoningFormat::Deepseek));
    }

    #[test]
    fn generate_without_raw_is_one_user_message_with_system() {
        let req = generate(json!({
            "model": "m:latest", "prompt": "Hi", "system": "Be brief", "context": [1, 2],
            "keep_alive": 0,
        }))
        .unwrap();
        assert_eq!(req.model, "resolved");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(
            req.messages[0].content.as_ref().map(|c| c.text()),
            Some("Hi".into())
        );
        assert_eq!(req.system.as_deref(), Some("Be brief"));
        assert_eq!(req.raw, Some(false));
        assert_eq!(req.context, Some(vec![1, 2]));
        assert_eq!(req.keep_alive.as_deref(), Some("0"));
        assert_eq!(req.fim_suffix, None);
    }

    #[test]
    fn raw_and_suffix_take_the_raw_path() {
        let raw = generate(json!({"model": "m", "prompt": "x", "raw": true})).unwrap();
        assert_eq!(raw.raw, Some(true));
        let fim = generate(json!({"model": "m", "prompt": "fn a(", "suffix": ")"})).unwrap();
        assert_eq!(fim.raw, Some(true));
        assert_eq!(fim.fim_suffix.as_deref(), Some(")"));
        let both = generate(json!({"model": "m", "prompt": "a", "suffix": "b", "raw": true}));
        assert!(both.err().unwrap_or_default().contains("raw"));
    }

    #[test]
    fn a_template_must_be_a_built_in_name() {
        let named = generate(json!({"model": "m", "prompt": "x", "template": "chatml"})).unwrap();
        assert_eq!(named.template.as_deref(), Some("chatml"));
        let go = generate(json!({"model": "m", "prompt": "x", "template": "{{ .Prompt }}"}));
        let err = go.err().unwrap_or_default();
        assert!(
            err.contains("Go templates are unsupported") && err.contains("chatml"),
            "{err}"
        );
    }
}
