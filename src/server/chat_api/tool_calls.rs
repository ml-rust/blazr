//! Tool calls parsed out of a complete model output, and the two call spellings.

use super::tool_parser::ToolCallParser;
use super::tools::{Tool, ToolCall, ToolCallFunction};

/// Extract tool calls from a complete model output.
///
/// The output goes through [`ToolCallParser`] in one piece, so a complete
/// response and a stream read calls the same way. The parser documents the
/// spellings it reads.
///
/// - With no call, the content is `text` unchanged.
/// - With a call, the content is the text before the first call, trimmed.
///   It is `None` when that text is empty.
pub fn extract_tool_calls(
    text: &str,
    declared: &[Tool],
) -> (Option<String>, Option<Vec<ToolCall>>) {
    let mut parser = ToolCallParser::new(declared.to_vec());
    let mut events = parser.feed(text);
    let tail = parser.finish();
    events.content.push_str(&tail.content);
    events.calls.extend(tail.calls);
    if events.calls.is_empty() {
        return (Some(events.content), None);
    }
    let content = events.content.trim();
    let content = (!content.is_empty()).then(|| content.to_string());
    (content, Some(events.calls))
}

/// True when `call` names a tool in `declared`. An empty `declared` admits every call.
pub(super) fn is_declared(declared: &[Tool], call: &ToolCall) -> bool {
    declared.is_empty()
        || declared
            .iter()
            .any(|t| t.function.name == call.function.name)
}

/// A fresh `call_` id.
fn new_call_id() -> String {
    format!(
        "call_{}",
        &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
    )
}

/// Parse `{"name": ..., "arguments": ...}` as a tool call.
///
/// Object `arguments` are serialized compact. String `arguments` are kept as they are.
pub(super) fn parse_tool_call_json(json_str: &str) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let obj = v.as_object()?;
    let name = obj.get("name")?.as_str()?;
    let arguments = obj.get("arguments")?;

    let args_str = if let Some(s) = arguments.as_str() {
        s.to_string()
    } else {
        serde_json::to_string(arguments).ok()?
    };

    Some(ToolCall {
        id: new_call_id(),
        r#type: "function".to_string(),
        function: ToolCallFunction {
            name: name.to_string(),
            arguments: args_str,
        },
    })
}

/// Parse the `<function=name><parameter=k>v</parameter></function>` spelling.
///
/// A parameter value that parses as JSON is kept as that value. Any other is
/// a string. The result is the same `arguments` JSON object the JSON spelling
/// carries.
pub(super) fn parse_tool_call_xml(body: &str) -> Option<ToolCall> {
    let body = body.trim();
    let open = body.strip_prefix("<function=")?;
    let name_end = open.find('>')?;
    let name = open[..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let inner = open[name_end + 1..]
        .trim()
        .strip_suffix("</function>")?
        .trim();

    let mut arguments = serde_json::Map::new();
    let mut rest = inner;
    while let Some(start) = rest.find("<parameter=") {
        let after = &rest[start + "<parameter=".len()..];
        let key_end = after.find('>')?;
        let key = after[..key_end].trim().to_string();
        let value_start = &after[key_end + 1..];
        let value_end = value_start.find("</parameter>")?;
        let raw = value_start[..value_end].trim();
        let value = serde_json::from_str::<serde_json::Value>(raw)
            .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
        arguments.insert(key, value);
        rest = &value_start[value_end + "</parameter>".len()..];
    }

    Some(ToolCall {
        id: new_call_id(),
        r#type: "function".to_string(),
        function: ToolCallFunction {
            name: name.to_string(),
            arguments: serde_json::Value::Object(arguments).to_string(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_tool_calls_tagged() {
        let text = r#"I'll look that up for you.
<tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>"#;
        let (content, calls) = extract_tool_calls(text, &[]);
        assert_eq!(content.as_deref(), Some("I'll look that up for you."));
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].r#type, "function");
        assert!(calls[0].function.arguments.contains("London"));
    }

    #[test]
    fn test_extract_tool_calls_multiple_tags() {
        let text = r#"<tool_call>{"name": "fn_a", "arguments": {}}</tool_call>
<tool_call>{"name": "fn_b", "arguments": {"x": 1}}</tool_call>"#;
        let (content, calls) = extract_tool_calls(text, &[]);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "fn_a");
        assert_eq!(calls[1].function.name, "fn_b");
        assert!(content.is_none() || content.as_deref() == Some(""));
    }

    #[test]
    fn test_extract_tool_calls_bare_json() {
        let text = r#"Sure, let me check.
{"name": "search", "arguments": {"query": "rust lang"}}"#;
        let (content, calls) = extract_tool_calls(text, &[]);
        assert_eq!(content.as_deref(), Some("Sure, let me check."));
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_extract_tool_calls_no_tools() {
        let text = "Just a regular response with no tool calls.";
        let (content, calls) = extract_tool_calls(text, &[]);
        assert_eq!(content.as_deref(), Some(text));
        assert!(calls.is_none());
    }

    #[test]
    fn test_extract_tool_calls_invalid_json_ignored() {
        let text = r#"<tool_call>not valid json</tool_call>"#;
        let (content, calls) = extract_tool_calls(text, &[]);
        assert!(calls.is_none());
        assert!(content.is_some());
    }

    #[test]
    fn test_extract_tool_calls_function_xml() {
        let text = "Checking.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=note>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = extract_tool_calls(text, &[]);
        assert_eq!(content.as_deref(), Some("Checking."));
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "Paris");
        assert_eq!(args["days"], 3);
        assert_eq!(args["note"], "line one\nline two");
    }

    #[test]
    fn test_function_xml_without_parameters() {
        let (_, calls) =
            extract_tool_calls("<tool_call><function=ping></function></tool_call>", &[]);
        let calls = calls.unwrap();
        assert_eq!(calls[0].function.name, "ping");
        assert_eq!(calls[0].function.arguments, "{}");
    }

    fn declared(names: &[&str]) -> Vec<Tool> {
        names
            .iter()
            .map(|name| Tool {
                r#type: "function".to_string(),
                function: super::super::tools::ToolFunction {
                    name: name.to_string(),
                    description: None,
                    parameters: None,
                },
            })
            .collect()
    }

    #[test]
    fn an_undeclared_name_is_dropped_in_every_spelling() {
        let tools = declared(&["known"]);
        let tagged = r#"<tool_call>{"name": "ghost", "arguments": {}}</tool_call>"#;
        let bare = r#"{"name": "ghost", "arguments": {}}"#;
        let xml = "<tool_call><function=ghost></function></tool_call>";
        for text in [tagged, bare, xml] {
            let (content, calls) = extract_tool_calls(text, &tools);
            assert!(calls.is_none(), "{text}");
            assert_eq!(content.as_deref(), Some(text));
        }
    }

    #[test]
    fn a_declared_call_survives_beside_an_undeclared_one() {
        let tools = declared(&["known"]);
        let text = r#"<tool_call>{"name": "ghost", "arguments": {}}</tool_call>
<tool_call>{"name": "known", "arguments": {"x": 1}}</tool_call>"#;
        let (_, calls) = extract_tool_calls(text, &tools);
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "known");
    }

    #[test]
    fn a_whole_text_call_has_no_content() {
        let tools = declared(&["known"]);
        let text = r#"  {"name":"known","arguments":{"note":"a {b} c"}}"#;
        let (content, calls) = extract_tool_calls(text, &tools);
        assert!(content.is_none());
        assert_eq!(
            calls.unwrap()[0].function.arguments,
            r#"{"note":"a {b} c"}"#
        );
    }

    #[test]
    fn bare_json_mid_line_stays_content() {
        let text = r#"Use {"name": "search", "arguments": {}} to search."#;
        let (content, calls) = extract_tool_calls(text, &[]);
        assert!(calls.is_none());
        assert_eq!(content.as_deref(), Some(text));
    }

    #[test]
    fn a_call_id_is_call_and_24_hex_digits() {
        let (_, calls) = extract_tool_calls(r#"{"name":"f","arguments":{}}"#, &[]);
        let id = calls.unwrap()[0].id.clone();
        let hex = id.strip_prefix("call_").unwrap();
        assert_eq!(hex.len(), 24, "{id}");
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()), "{id}");
    }
}
