//! One chat token's text through its choice's parsers: the reasoning and
//! content halves, and the tool calls the token releases.

use super::reasoning::ReasoningParser;
use crate::server::tool_parser::ToolCallParser;
use crate::server::tools::ToolCall;

/// What one token's text releases.
#[derive(Debug)]
pub(super) struct TokenText {
    /// The `delta.reasoning_content` text.
    pub reasoning: String,
    /// The `delta.content` text.
    pub content: String,
    /// Calls whose boundary arrived with this token, in output order.
    pub calls: Vec<ToolCall>,
}

/// Split `text` with `reasoning`, then run its content half through `tools`.
///
/// Without a reasoning parser, all of `text` is content. Without a tool
/// parser, no call is released. A final token also releases what each
/// parser holds back.
pub(super) fn split_token_text(
    reasoning: Option<&mut ReasoningParser>,
    tools: Option<&mut ToolCallParser>,
    text: String,
    is_final: bool,
) -> TokenText {
    let (reasoning, content) = match reasoning {
        Some(parser) => {
            let (mut reasoning, mut content) = parser.feed(&text);
            if is_final {
                let (tail_reasoning, tail_content) = parser.finish();
                reasoning.push_str(&tail_reasoning);
                content.push_str(&tail_content);
            }
            (reasoning, content)
        }
        None => (String::new(), text),
    };
    let (content, calls) = match tools {
        Some(parser) => {
            let mut events = parser.feed(&content);
            if is_final {
                let tail = parser.finish();
                events.content.push_str(&tail.content);
                events.calls.extend(tail.calls);
            }
            (events.content, events.calls)
        }
        None => (content, Vec::new()),
    };
    TokenText {
        reasoning,
        content,
        calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::reasoning::ReasoningTags;
    use crate::server::tools::{Tool, ToolFunction};

    fn tools() -> ToolCallParser {
        ToolCallParser::new(vec![Tool {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: "weather".to_string(),
                description: None,
                parameters: None,
            },
        }])
    }

    #[test]
    fn without_parsers_the_text_is_content() {
        let got = split_token_text(None, None, "plain".to_string(), false);
        assert_eq!(
            (got.reasoning.as_str(), got.content.as_str()),
            ("", "plain")
        );
        assert!(got.calls.is_empty());
    }

    #[test]
    fn reasoning_splits_first_and_the_content_half_releases_the_call() {
        let mut reasoning = ReasoningParser::new(ReasoningTags::THINK, false);
        let mut parser = tools();
        let text = r#"<think>plan</think>{"name":"weather","arguments":{}}"#;
        let got = split_token_text(
            Some(&mut reasoning),
            Some(&mut parser),
            text.to_string(),
            true,
        );
        assert_eq!(got.reasoning, "plan");
        assert_eq!(got.content, "");
        assert_eq!(got.calls.len(), 1);
        assert_eq!(got.calls[0].function.name, "weather");
    }

    /// An open block holds its text until the final token releases it.
    #[test]
    fn a_final_token_releases_held_text() {
        let mut reasoning = ReasoningParser::new(ReasoningTags::THINK, false);
        let held = split_token_text(Some(&mut reasoning), None, "<think>pla".to_string(), false);
        let tail = split_token_text(Some(&mut reasoning), None, "n".to_string(), true);
        let joined = format!("{}{}", held.reasoning, tail.reasoning);
        assert_eq!(joined, "plan");
        assert_eq!(tail.content, "");
    }
}
