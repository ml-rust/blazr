//! The chat SSE stream of a forced tool call: buffer the output, send it as `delta.tool_calls`.

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use futures::StreamExt;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::pin::Pin;

use super::tool_calls::extract_tool_calls;
use super::tools::Tool;
use crate::server::reasoning::ReasoningParser;
use crate::server::streaming::{error_event, StreamToken};

/// One `chat.completion.chunk` with a single choice.
fn chunk(id: &str, model: &str, created: i64, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
}

/// The chunks that follow the role chunk, for the complete output `text`.
///
/// - A declared call becomes one `delta.tool_calls` chunk, then a chunk with
///   `finish_reason` `tool_calls`. Text before the call comes first as content.
/// - Output with no declared call (for example one cut short by the token
///   limit) becomes one content chunk, then a chunk with `finish`.
pub fn tool_call_chunks(
    id: &str,
    model: &str,
    created: i64,
    text: &str,
    finish: &str,
    declared: &[Tool],
) -> Vec<Value> {
    let (content, calls) = extract_tool_calls(text, declared);
    let make = |delta, finish| chunk(id, model, created, delta, finish);
    let mut chunks = Vec::new();
    if let Some(content) = content.filter(|c| !c.is_empty()) {
        chunks.push(make(json!({"content": content}), None));
    }
    match calls {
        Some(calls) => {
            let deltas: Vec<Value> = calls
                .iter()
                .enumerate()
                .map(|(index, call)| {
                    json!({
                        "index": index,
                        "id": call.id,
                        "type": call.r#type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    })
                })
                .collect();
            chunks.push(make(json!({"tool_calls": deltas}), None));
            chunks.push(make(json!({}), Some("tool_calls")));
        }
        None => chunks.push(make(json!({}), Some(finish))),
    }
    chunks
}

/// Create the SSE stream of a forced tool call.
///
/// The stream holds every token until the output ends, then sends the tool
/// calls it holds as `delta.tool_calls`. Logprobs are not sent. A reasoning
/// split drops the reasoning text from the calls.
pub fn create_tool_call_stream(
    id: String,
    model: String,
    tokens: Pin<Box<dyn Stream<Item = StreamToken> + Send>>,
    reasoning: Option<ReasoningParser>,
    declared: Vec<Tool>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();
        let mut tokens = tokens;
        let mut reasoning = reasoning;
        let role = chunk(&id, &model, created, json!({"role": "assistant"}), None);
        yield Ok(Event::default().data(role.to_string()));

        let mut text = String::new();
        let mut finish = "stop".to_string();
        let mut failed = false;
        while let Some(token) = tokens.next().await {
            if let Some(ref err) = token.error {
                if let Some(evt) = error_event(err) {
                    yield Ok(evt);
                }
                failed = true;
                break;
            }
            let is_final = token.finish_reason.is_some();
            if let Some(reason) = token.finish_reason {
                finish = reason.as_str().to_string();
            }
            match reasoning.as_mut() {
                Some(parser) => {
                    text.push_str(&parser.feed(&token.text).1);
                    if is_final {
                        text.push_str(&parser.finish().1);
                    }
                }
                None => text.push_str(&token.text),
            }
            if is_final {
                break;
            }
        }

        if !failed {
            for part in tool_call_chunks(&id, &model, created, &text, &finish, &declared) {
                yield Ok(Event::default().data(part.to_string()));
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

#[cfg(test)]
mod tests {
    use super::super::tools::ToolFunction;
    use super::*;
    use crate::engine::FinishReason;
    use crate::server::streaming::sse_test::{events, feed};

    fn declared() -> Vec<Tool> {
        vec![Tool {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: "weather".to_string(),
                description: None,
                parameters: None,
            },
        }]
    }

    #[test]
    fn a_declared_call_becomes_a_tool_calls_delta_and_finish() {
        let text = r#"{"name":"weather","arguments":{"city":"Paris"}}"#;
        let got = tool_call_chunks("c", "m", 1, text, "stop", &declared());
        assert_eq!(got.len(), 2, "{got:?}");
        let call = &got[0]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "weather");
        assert_eq!(call["function"]["arguments"], r#"{"city":"Paris"}"#);
        assert_eq!(got[1]["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn output_without_a_declared_call_keeps_its_finish_reason() {
        let text = r#"{"name":"weather","argum"#;
        let got = tool_call_chunks("c", "m", 1, text, "length", &declared());
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0]["choices"][0]["delta"]["content"], text);
        assert_eq!(got[1]["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn the_stream_buffers_tokens_and_ends_with_done() -> anyhow::Result<()> {
        let items = vec![
            StreamToken::text(r#"{"name":"weather","#.into(), None),
            StreamToken::text(r#""arguments":{}}"#.into(), Some(FinishReason::Stop)),
        ];
        let sse = create_tool_call_stream("c".into(), "m".into(), feed(items), None, declared());
        let got = events(sse).await?;
        assert_eq!(got.len(), 4, "{got:?}");
        assert_eq!(got[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            got[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "weather"
        );
        assert_eq!(got[2]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(got[3], json!("[DONE]"));
        Ok(())
    }
}
