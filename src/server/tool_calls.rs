//! Tool calls parsed out of model output.

use super::tools::{ToolCall, ToolCallFunction};

/// Extract tool calls from model-generated text.
///
/// Supported spellings:
/// 1. `<tool_call>{"name": "fn", "arguments": {...}}</tool_call>`
/// 2. `<tool_call>\n<function=fn>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>`
/// 3. `{"name": "fn", "arguments": {...}}` (bare JSON object with name+arguments keys)
pub fn extract_tool_calls(text: &str) -> (Option<String>, Option<Vec<ToolCall>>) {
    let mut calls = Vec::new();
    let tag_start = "<tool_call>";
    let tag_end = "</tool_call>";

    if text.contains(tag_start) {
        let mut search = text;
        while let Some(start) = search.find(tag_start) {
            let after_tag = &search[start + tag_start.len()..];
            if let Some(end) = after_tag.find(tag_end) {
                let body = after_tag[..end].trim();
                if let Some(call) = parse_tool_call_json(body).or_else(|| parse_tool_call_xml(body))
                {
                    calls.push(call);
                }
                search = &after_tag[end + tag_end.len()..];
            } else {
                break;
            }
        }
        if !calls.is_empty() {
            let remaining = text
                .split(tag_start)
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            let content = if remaining.is_empty() {
                None
            } else {
                Some(remaining)
            };
            return (content, Some(calls));
        }
    }

    // Bare JSON object with "name" and "arguments" keys: scan backwards for
    // each '{' and try parsing as a tool call.
    let mut search_from = text.len();
    while search_from > 0 {
        let prefix = &text[..search_from];
        let Some(start) = prefix.rfind('{') else {
            break;
        };
        let candidate = &text[start..];
        let mut depth = 0;
        let mut end_pos = None;
        for (i, ch) in candidate.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end_pos = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(end) = end_pos {
            let json_str = &candidate[..end];
            if let Some(call) = parse_tool_call_json(json_str) {
                let content_before = text[..start].trim();
                let content = if content_before.is_empty() {
                    None
                } else {
                    Some(content_before.to_string())
                };
                return (content, Some(vec![call]));
            }
        }
        search_from = start;
    }

    (Some(text.to_string()), None)
}

/// A fresh `call_` id.
fn new_call_id() -> String {
    format!(
        "call_{}",
        &uuid::Uuid::new_v4().to_string().replace('-', "")[..24]
    )
}

/// Parse a JSON string as a tool call
fn parse_tool_call_json(json_str: &str) -> Option<ToolCall> {
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
/// A parameter value that parses as JSON is kept as that value; any other is
/// a string. The result is the same `arguments` JSON object the JSON spelling
/// carries.
fn parse_tool_call_xml(body: &str) -> Option<ToolCall> {
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
        let (content, calls) = extract_tool_calls(text);
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
        let (content, calls) = extract_tool_calls(text);
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
        let (content, calls) = extract_tool_calls(text);
        assert_eq!(content.as_deref(), Some("Sure, let me check."));
        let calls = calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_extract_tool_calls_no_tools() {
        let text = "Just a regular response with no tool calls.";
        let (content, calls) = extract_tool_calls(text);
        assert_eq!(content.as_deref(), Some(text));
        assert!(calls.is_none());
    }

    #[test]
    fn test_extract_tool_calls_invalid_json_ignored() {
        let text = r#"<tool_call>not valid json</tool_call>"#;
        let (content, calls) = extract_tool_calls(text);
        assert!(calls.is_none());
        assert!(content.is_some());
    }

    #[test]
    fn test_extract_tool_calls_function_xml() {
        let text = "Checking.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=note>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        let (content, calls) = extract_tool_calls(text);
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
        let (_, calls) = extract_tool_calls("<tool_call><function=ping></function></tool_call>");
        let calls = calls.unwrap();
        assert_eq!(calls[0].function.name, "ping");
        assert_eq!(calls[0].function.arguments, "{}");
    }
}
