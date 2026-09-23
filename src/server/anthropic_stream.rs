//! Server-sent events for the Anthropic Messages route.
//!
//! The route streams the event sequence an Anthropic client expects:
//! `message_start`, a `content_block_start`/`content_block_delta`/
//! `content_block_stop` run per block, then `message_delta` and
//! `message_stop`. Reasoning text opens a `thinking` block, reply text a
//! `text` block, each at its own index.

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use futures::StreamExt;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::pin::Pin;

use super::reasoning::ReasoningParser;
use super::streaming::StreamToken;
use crate::engine::FinishReason;

/// The Anthropic name for a finish reason.
fn stop_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Length => "max_tokens",
        _ => "end_turn",
    }
}

/// Turns generated tokens into Anthropic events, one block at a time.
struct EventBuilder {
    id: String,
    model: String,
    input_tokens: usize,
    output_tokens: usize,
    reasoning: Option<ReasoningParser>,
    thinking_index: Option<usize>,
    text_index: Option<usize>,
    next_index: usize,
}

impl EventBuilder {
    fn new(
        id: String,
        model: String,
        input_tokens: usize,
        reasoning: Option<ReasoningParser>,
    ) -> Self {
        Self {
            id,
            model,
            input_tokens,
            output_tokens: 0,
            reasoning,
            thinking_index: None,
            text_index: None,
            next_index: 0,
        }
    }

    /// The opening event, sent before any token.
    fn start(&self) -> (&'static str, Value) {
        (
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                }
            }),
        )
    }

    /// Opens `kind` at a fresh index, or reuses the index it already holds.
    fn open_block(&mut self, kind: &'static str, events: &mut Vec<(&'static str, Value)>) -> usize {
        let held = match kind {
            "thinking" => self.thinking_index,
            _ => self.text_index,
        };
        if let Some(index) = held {
            return index;
        }
        let index = self.next_index;
        self.next_index += 1;
        let block = if kind == "thinking" {
            self.thinking_index = Some(index);
            json!({"type": "thinking", "thinking": ""})
        } else {
            self.text_index = Some(index);
            json!({"type": "text", "text": ""})
        };
        events.push((
            "content_block_start",
            json!({"type": "content_block_start", "index": index, "content_block": block}),
        ));
        index
    }

    /// The events one generated token produces.
    fn feed(&mut self, token: &StreamToken) -> Vec<(&'static str, Value)> {
        let mut events = Vec::new();
        self.output_tokens += 1;
        let is_final = token.finish_reason.is_some();
        let (thinking, text) = match self.reasoning.as_mut() {
            Some(parser) => {
                let (mut r, mut c) = parser.feed(&token.text);
                if is_final {
                    let (tail_r, tail_c) = parser.finish();
                    r.push_str(&tail_r);
                    c.push_str(&tail_c);
                }
                (r, c)
            }
            None => (String::new(), token.text.clone()),
        };
        if !thinking.is_empty() {
            let index = self.open_block("thinking", &mut events);
            events.push((
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": thinking},
                }),
            ));
        }
        if !text.is_empty() {
            if let Some(index) = self.thinking_index.take() {
                events.push((
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": index}),
                ));
            }
            let index = self.open_block("text", &mut events);
            events.push((
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": text},
                }),
            ));
        }
        events
    }

    /// The closing events for a run that ended with `reason`.
    fn finish(&mut self, reason: Option<FinishReason>) -> Vec<(&'static str, Value)> {
        let mut events = Vec::new();
        for index in [self.thinking_index.take(), self.text_index.take()]
            .into_iter()
            .flatten()
        {
            events.push((
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            ));
        }
        events.push((
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": reason.map(stop_reason),
                    "stop_sequence": Value::Null,
                },
                "usage": {"output_tokens": self.output_tokens},
            }),
        ));
        events.push(("message_stop", json!({"type": "message_stop"})));
        events
    }
}

/// An `error` event carrying a generation error.
fn error_event(message: &str) -> (&'static str, Value) {
    (
        "error",
        json!({
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }),
    )
}

/// Serializes one named event.
fn sse(name: &'static str, payload: &Value) -> Option<Event> {
    serde_json::to_string(payload)
        .ok()
        .map(|data| Event::default().event(name).data(data))
}

/// Streams a generation as Anthropic Messages events.
///
/// With `reasoning` set, reasoning text goes to a `thinking` block and the
/// reply to a `text` block; without it, every token is reply text.
pub(super) fn create_message_stream(
    id: String,
    model: String,
    input_tokens: usize,
    tokens: Pin<Box<dyn Stream<Item = StreamToken> + Send>>,
    reasoning: Option<ReasoningParser>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let mut builder = EventBuilder::new(id, model, input_tokens, reasoning);
        let mut tokens = tokens;

        let (name, payload) = builder.start();
        if let Some(event) = sse(name, &payload) {
            yield Ok(event);
        }

        let mut reason = None;
        while let Some(token) = tokens.next().await {
            if let Some(ref err) = token.error {
                let (name, payload) = error_event(err);
                if let Some(event) = sse(name, &payload) {
                    yield Ok(event);
                }
                return;
            }
            reason = token.finish_reason;
            for (name, payload) in builder.feed(&token) {
                if let Some(event) = sse(name, &payload) {
                    yield Ok(event);
                }
            }
            if reason.is_some() {
                break;
            }
        }

        for (name, payload) in builder.finish(reason) {
            if let Some(event) = sse(name, &payload) {
                yield Ok(event);
            }
        }
    };

    Sse::new(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::reasoning::ReasoningTags;

    fn token(text: &str, finish: Option<FinishReason>) -> StreamToken {
        StreamToken {
            text: text.to_string(),
            finish_reason: finish,
            error: None,
        }
    }

    fn names(events: &[(&'static str, Value)]) -> Vec<&'static str> {
        events.iter().map(|(name, _)| *name).collect()
    }

    /// A reply with no reasoning opens one text block and closes it.
    #[test]
    fn test_text_only_block_sequence() {
        let mut builder = EventBuilder::new("msg_1".into(), "m".into(), 7, None);
        assert_eq!(builder.start().0, "message_start");
        let first = builder.feed(&token("Hello", None));
        assert_eq!(
            names(&first),
            ["content_block_start", "content_block_delta"]
        );
        assert_eq!(first[0].1["index"], 0);
        let second = builder.feed(&token(" there", None));
        assert_eq!(names(&second), ["content_block_delta"]);
        assert_eq!(second[0].1["delta"]["text"], " there");
        let end = builder.finish(Some(FinishReason::Stop));
        assert_eq!(
            names(&end),
            ["content_block_stop", "message_delta", "message_stop"]
        );
        assert_eq!(end[1].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(end[1].1["usage"]["output_tokens"], 2);
    }

    /// Reasoning text opens a thinking block that closes when the reply
    /// starts, and the reply takes the next index.
    #[test]
    fn test_thinking_then_text_blocks() {
        let parser = ReasoningParser::new(ReasoningTags::THINK, true);
        let mut builder = EventBuilder::new("msg_2".into(), "m".into(), 3, Some(parser));
        let thought = builder.feed(&token("why", None));
        assert_eq!(
            names(&thought),
            ["content_block_start", "content_block_delta"]
        );
        assert_eq!(thought[0].1["content_block"]["type"], "thinking");
        assert_eq!(thought[1].1["delta"]["thinking"], "why");
        let reply = builder.feed(&token("</think>done", None));
        assert_eq!(
            names(&reply),
            [
                "content_block_stop",
                "content_block_start",
                "content_block_delta"
            ]
        );
        assert_eq!(reply[0].1["index"], 0);
        assert_eq!(reply[1].1["index"], 1);
        assert_eq!(reply[2].1["delta"]["text"], "done");
        let end = builder.finish(Some(FinishReason::Length));
        assert_eq!(
            names(&end),
            ["content_block_stop", "message_delta", "message_stop"]
        );
        assert_eq!(end[0].1["index"], 1);
        assert_eq!(end[1].1["delta"]["stop_reason"], "max_tokens");
    }

    /// A run that ends inside the thinking block still closes it.
    #[test]
    fn test_finish_closes_open_thinking_block() {
        let parser = ReasoningParser::new(ReasoningTags::THINK, true);
        let mut builder = EventBuilder::new("msg_3".into(), "m".into(), 1, Some(parser));
        builder.feed(&token("still thinking", None));
        let end = builder.finish(Some(FinishReason::Length));
        assert_eq!(
            names(&end),
            ["content_block_stop", "message_delta", "message_stop"]
        );
        assert_eq!(end[0].1["index"], 0);
    }

    /// The opening event carries the prompt's token count.
    #[test]
    fn test_message_start_reports_input_tokens() {
        let builder = EventBuilder::new("msg_4".into(), "model-x".into(), 42, None);
        let (name, payload) = builder.start();
        assert_eq!(name, "message_start");
        assert_eq!(payload["message"]["usage"]["input_tokens"], 42);
        assert_eq!(payload["message"]["model"], "model-x");
        assert_eq!(payload["message"]["id"], "msg_4");
    }
}
