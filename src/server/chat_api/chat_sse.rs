//! The SSE response of a streamed chat completion.

use std::pin::Pin;

use axum::response::{IntoResponse, Response};
use futures::stream::Stream;

use super::tool_parser::ToolCallParser;
use super::tools::Tool;
use crate::server::chat_stream::create_chat_stream;
use crate::server::reasoning::ReasoningParser;
use crate::server::streaming::StreamToken;

/// The chat stream of `tokens`.
///
/// With `visible_tools` non-empty, tool calls in the output stream as
/// `delta.tool_calls`, for every `tool_choice` that shows the model tools.
/// A forced call is bare JSON at the start of the output and parses the
/// same way.
pub(super) fn chat_sse(
    id: String,
    model: String,
    tokens: Pin<Box<dyn Stream<Item = StreamToken> + Send>>,
    reasoning: Option<ReasoningParser>,
    visible_tools: &[Tool],
    include_usage: bool,
) -> Response {
    let has_tools = !visible_tools.is_empty();
    let tool_parser = has_tools.then(|| ToolCallParser::new(visible_tools.to_vec()));
    create_chat_stream(id, model, tokens, reasoning, tool_parser, include_usage).into_response()
}

#[cfg(test)]
mod tests {
    use super::super::chat_types::ChatRequest;
    use super::super::tool_choice::ToolChoice;
    use super::*;
    use crate::engine::FinishReason;
    use crate::server::streaming::sse_test::{events, feed};
    use serde_json::{json, Value};

    /// The events of a stream of `pieces` for a request with `fields` and one `weather` tool.
    async fn stream(fields: Value, pieces: &[&str]) -> anyhow::Result<Vec<Value>> {
        let mut body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "weather"}}],
        });
        if let (Some(body), Some(fields)) = (body.as_object_mut(), fields.as_object()) {
            body.extend(fields.clone());
        }
        let request: ChatRequest = serde_json::from_value(body)?;
        let tools = request.tools.as_deref().unwrap_or(&[]);
        let choice = ToolChoice::parse(request.tool_choice.as_ref(), Some(tools))
            .map_err(anyhow::Error::msg)?;
        let items = pieces
            .iter()
            .enumerate()
            .map(|(i, piece)| {
                let last = (i + 1 == pieces.len()).then_some(FinishReason::Eos);
                StreamToken::text(piece.to_string(), last)
            })
            .collect();
        let sse = chat_sse(
            "c".into(),
            "m".into(),
            feed(items),
            None,
            choice.visible(tools),
            false,
        );
        events(sse).await
    }

    const CALL: [&str; 2] = [
        "<tool_call>{\"name\":\"weather\",",
        "\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
    ];

    #[tokio::test]
    async fn tools_without_tool_choice_stream_tool_call_deltas() -> anyhow::Result<()> {
        let got = stream(json!({}), &CALL).await?;
        let head = &got[1]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(head["function"]["name"], "weather", "{got:?}");
        assert_eq!(head["type"], "function");
        let finish = &got[got.len() - 2]["choices"][0]["finish_reason"];
        assert_eq!(*finish, "tool_calls", "{got:?}");
        assert!(got
            .iter()
            .all(|e| e["choices"][0]["delta"].get("content").is_none()));
        Ok(())
    }

    #[tokio::test]
    async fn tool_choice_none_leaves_the_call_text_as_content() -> anyhow::Result<()> {
        let got = stream(json!({"tool_choice": "none"}), &CALL).await?;
        let content: String = got
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(content, CALL.concat());
        assert_eq!(got[got.len() - 2]["choices"][0]["finish_reason"], "stop");
        Ok(())
    }
}
