//! The `delta.tool_calls` entries one parsed tool call streams as.

use serde::Serialize;

use crate::server::tools::ToolCall;

/// Most bytes of `arguments` one fragment carries. A fragment ends on a
/// char boundary, so one can run a few bytes longer.
const ARGUMENT_FRAGMENT_BYTES: usize = 32;

/// One entry of a chunk's `delta.tool_calls`.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallDelta {
    /// Position of the call among the calls of this response.
    pub index: usize,
    /// Set on the first entry of a call only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// `function`, on the first entry of a call only.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    pub function: FunctionDelta,
}

/// The `function` member of a [`ToolCallDelta`].
#[derive(Debug, Clone, Serialize)]
pub struct FunctionDelta {
    /// Set on the first entry of a call only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The next piece of the arguments JSON. Empty on the first entry.
    pub arguments: String,
}

/// The entries `call` streams as, at `index`.
///
/// The first entry carries `id`, `type`, `function.name` and empty
/// `arguments`. Each later entry carries one fragment of `arguments`. The
/// fragments concatenate to `call.function.arguments`.
pub(super) fn call_deltas(index: usize, call: ToolCall) -> Vec<ToolCallDelta> {
    let mut deltas = vec![ToolCallDelta {
        index,
        id: Some(call.id),
        kind: Some(call.r#type),
        function: FunctionDelta {
            name: Some(call.function.name),
            arguments: String::new(),
        },
    }];
    deltas.extend(
        fragments(&call.function.arguments, ARGUMENT_FRAGMENT_BYTES).map(|piece| ToolCallDelta {
            index,
            id: None,
            kind: None,
            function: FunctionDelta {
                name: None,
                arguments: piece.to_string(),
            },
        }),
    );
    deltas
}

/// `text` in pieces of `max` bytes, each cut moved forward to the next char boundary.
fn fragments(text: &str, max: usize) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let mut cut = rest.len().min(max.max(1));
        while !rest.is_char_boundary(cut) {
            cut += 1;
        }
        let (piece, tail) = rest.split_at(cut);
        rest = tail;
        Some(piece)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::tools::ToolCallFunction;

    #[test]
    fn fragments_cut_on_char_boundaries_and_rejoin() {
        let text = "{\"q\":\"\u{4e2d}\u{6587}\u{1f600}\"}";
        let pieces: Vec<&str> = fragments(text, 3).collect();
        assert!(pieces.len() > 1, "{pieces:?}");
        assert_eq!(pieces.concat(), text);
        assert_eq!(fragments("", 3).count(), 0);
    }

    #[test]
    fn the_first_delta_names_the_call_and_the_rest_carry_arguments() {
        let arguments = format!("{{\"text\":\"{}\"}}", "x".repeat(80));
        let call = ToolCall {
            id: "call_1".to_string(),
            r#type: "function".to_string(),
            function: ToolCallFunction {
                name: "echo".to_string(),
                arguments: arguments.clone(),
            },
        };
        let deltas = call_deltas(2, call);
        let first = serde_json::to_value(&deltas[0]).unwrap();
        assert_eq!(
            first,
            serde_json::json!({"index": 2, "id": "call_1", "type": "function",
                "function": {"name": "echo", "arguments": ""}})
        );
        let rest: Vec<serde_json::Value> = deltas[1..]
            .iter()
            .map(|d| serde_json::to_value(d).unwrap())
            .collect();
        assert!(rest.len() > 1);
        for delta in &rest {
            assert_eq!(delta["index"], 2);
            assert!(delta.get("id").is_none() && delta.get("type").is_none());
            assert!(delta["function"].get("name").is_none());
        }
        let joined: String = rest
            .iter()
            .map(|d| d["function"]["arguments"].as_str().unwrap())
            .collect();
        assert_eq!(joined, arguments);
    }
}
