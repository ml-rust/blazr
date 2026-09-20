//! The message shape every chat template renders.

/// A chat message with role and content.
///
/// `name`, `tool_calls` and `reasoning_content` are set only by callers that
/// hand a Jinja template the structured turn. Built-in formats read `role`
/// and `content` alone.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Tool name on a `tool` turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// OpenAI-shaped tool calls on an `assistant` turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Model reasoning that preceded `content` on an `assistant` turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    /// A plain `role`/`content` turn.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            ..Self::default()
        }
    }

    /// The turn as a Jinja template sees it.
    ///
    /// A tool call's `arguments` arrives as a JSON string on the OpenAI wire;
    /// Hugging Face templates iterate it as a mapping, so a string that parses
    /// as JSON is handed over parsed.
    pub fn template_value(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("role".to_string(), self.role.clone().into());
        map.insert("content".to_string(), self.content.clone().into());
        if let Some(name) = &self.name {
            map.insert("name".to_string(), name.clone().into());
        }
        if let Some(calls) = &self.tool_calls {
            let calls = calls.iter().cloned().map(parse_arguments).collect();
            map.insert("tool_calls".to_string(), serde_json::Value::Array(calls));
        }
        if let Some(reasoning) = &self.reasoning_content {
            map.insert("reasoning_content".to_string(), reasoning.clone().into());
        }
        serde_json::Value::Object(map)
    }
}

/// Replace a `function.arguments` JSON string with the value it encodes.
fn parse_arguments(mut call: serde_json::Value) -> serde_json::Value {
    let Some(arguments) = call
        .get_mut("function")
        .and_then(|f| f.get_mut("arguments"))
    else {
        return call;
    };
    if let Some(text) = arguments.as_str() {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
            if parsed.is_object() {
                *arguments = parsed;
            }
        }
    }
    call
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_value_parses_argument_strings() {
        let msg = ChatMessage {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: Some(vec![serde_json::json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "f", "arguments": "{\"x\": 1}"}
            })]),
            ..ChatMessage::default()
        };
        let value = msg.template_value();
        assert_eq!(
            value["tool_calls"][0]["function"]["arguments"],
            serde_json::json!({"x": 1})
        );
        assert!(value.get("name").is_none());
        assert!(value.get("reasoning_content").is_none());
    }

    #[test]
    fn template_value_keeps_non_json_arguments() {
        let msg = ChatMessage {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: Some(vec![serde_json::json!({
                "function": {"name": "f", "arguments": "not json"}
            })]),
            ..ChatMessage::default()
        };
        let value = msg.template_value();
        assert_eq!(value["tool_calls"][0]["function"]["arguments"], "not json");
    }
}
