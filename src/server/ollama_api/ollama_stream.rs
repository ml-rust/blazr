//! The NDJSON stream of `/api/chat` and `/api/generate`.
//!
//! Each token goes through the reasoning split, then its content half
//! through the tool-call parser. Reasoning goes in `thinking`, never in
//! `content`. A tool call goes out whole, with object arguments, once its
//! boundary arrives. The final token's text goes out in its own chunk,
//! then the done object with empty output.

use std::pin::Pin;
use std::time::Instant;

use futures::stream::{Stream, StreamExt};
use serde_json::Value;

use super::ollama_render::{Clock, Piece, Render};
use super::ollama_types::{OllamaToolCall, OllamaToolCallFunction};
use crate::server::reasoning::ReasoningParser;
use crate::server::streaming::StreamToken;
use crate::server::tool_parser::ToolCallParser;
use crate::server::tools::ToolCall;

/// `call` with its arguments as a JSON object. Arguments that do not parse
/// stay a string.
pub(super) fn ollama_call(call: ToolCall) -> OllamaToolCall {
    let arguments = serde_json::from_str(&call.function.arguments)
        .unwrap_or(Value::String(call.function.arguments));
    OllamaToolCall {
        id: None,
        function: OllamaToolCallFunction {
            name: call.function.name,
            arguments,
        },
    }
}

/// Splits output into thinking, content and tool calls, one piece at a time.
pub(super) struct Splitter {
    reasoning: Option<ReasoningParser>,
    tools: Option<ToolCallParser>,
}

impl Splitter {
    pub(super) fn new(reasoning: Option<ReasoningParser>, tools: Option<ToolCallParser>) -> Self {
        Self { reasoning, tools }
    }

    /// The piece `text` releases. A final piece also releases what the
    /// parsers hold.
    pub(super) fn split(&mut self, text: &str, is_final: bool) -> Piece {
        let (mut thinking, mut content) = match self.reasoning.as_mut() {
            Some(parser) => parser.feed(text),
            None => (String::new(), text.to_string()),
        };
        if let (true, Some(parser)) = (is_final, self.reasoning.as_mut()) {
            let (tail_thinking, tail_content) = parser.finish();
            thinking.push_str(&tail_thinking);
            content.push_str(&tail_content);
        }
        let Some(parser) = self.tools.as_mut() else {
            return Piece {
                content,
                thinking,
                calls: Vec::new(),
            };
        };
        let mut events = parser.feed(&content);
        if is_final {
            let tail = parser.finish();
            events.content.push_str(&tail.content);
            events.calls.extend(tail.calls);
        }
        Piece {
            content: events.content,
            thinking,
            calls: events.calls.into_iter().map(ollama_call).collect(),
        }
    }
}

/// What a stream renders with.
pub(super) struct StreamSetup {
    /// The model name the client sent.
    pub model: String,
    pub splitter: Splitter,
    pub render: Render,
    pub clock: Clock,
}

/// The NDJSON items of `tokens`.
///
/// An error token is an `Err` item: the stream ends without a done object,
/// and a held tool call is dropped. A stream that closes before a final
/// token ends the same way.
pub(super) fn ndjson_items(
    setup: StreamSetup,
    tokens: Pin<Box<dyn Stream<Item = StreamToken> + Send>>,
) -> impl Stream<Item = Result<Value, String>> + Send {
    async_stream::stream! {
        let StreamSetup { model, mut splitter, render, clock } = setup;
        let mut tokens = tokens;
        let mut first = None;
        let mut generated = String::new();
        let mut finished = false;
        while let Some(token) = tokens.next().await {
            if let Some(error) = token.error {
                yield Err(error);
                finished = true;
                break;
            }
            first.get_or_insert_with(Instant::now);
            if render.context.is_some() {
                generated.push_str(&token.text);
            }
            let piece = splitter.split(&token.text, token.finish_reason.is_some());
            if !piece.is_empty() {
                yield Ok(render.chunk(&model, piece));
            }
            if let Some(reason) = token.finish_reason {
                let (prompt, completion) = match token.usage {
                    Some(usage) => (usage.prompt_tokens, usage.completion_tokens),
                    None => {
                        tracing::error!("Ollama stream: the final item has no token counts");
                        (0, 0)
                    }
                };
                let timings = clock.streamed(first, Instant::now(), prompt, completion);
                yield Ok(render.done(&model, Piece::default(), reason.as_str(), timings, &generated));
                finished = true;
                break;
            }
        }
        if !finished {
            yield Err("the generation ended without a final token".to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ndjson::{ndjson_response, ndjson_test};
    use super::super::ollama_render::Kind;
    use super::*;
    use crate::engine::FinishReason;
    use crate::server::reasoning::ReasoningTags;
    use crate::server::stream_usage::StreamUsage;
    use crate::server::streaming::sse_test::feed;
    use crate::server::tools::{Tool, ToolFunction};
    use serde_json::json;

    fn weather() -> ToolCallParser {
        ToolCallParser::new(vec![Tool {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: "weather".to_string(),
                description: None,
                parameters: None,
            },
        }])
    }

    fn setup(kind: Kind, reasoning: bool, tools: bool) -> StreamSetup {
        let now = Instant::now();
        StreamSetup {
            model: "m:latest".into(),
            splitter: Splitter::new(
                reasoning.then(|| ReasoningParser::new(ReasoningTags::THINK, false)),
                tools.then(weather),
            ),
            render: Render::new(kind),
            clock: Clock {
                start: now,
                load: std::time::Duration::ZERO,
                gen_start: now,
            },
        }
    }

    /// `pieces` as stream items, the last with `finish` and counts 7 and 3.
    fn items(pieces: &[&str], finish: FinishReason) -> Vec<StreamToken> {
        pieces
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let token = StreamToken::text(p.to_string(), None);
                if i + 1 == pieces.len() {
                    StreamToken::text(p.to_string(), Some(finish))
                        .with_usage(StreamUsage::new(7, 3))
                } else {
                    token
                }
            })
            .collect()
    }

    async fn run(setup: StreamSetup, items: Vec<StreamToken>) -> anyhow::Result<Vec<Value>> {
        let response = ndjson_response(ndjson_items(setup, feed(items)));
        let raw = ndjson_test::body(response).await?;
        assert!(!raw.contains("data:"), "{raw}");
        assert!(raw.ends_with('\n'), "{raw}");
        Ok(raw
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?)
    }

    #[tokio::test]
    async fn chat_lines_end_with_a_done_object_that_carries_the_counts() -> anyhow::Result<()> {
        let got = run(
            setup(Kind::Chat, false, false),
            items(&["Hel", "lo"], FinishReason::Eos),
        )
        .await?;
        assert_eq!(got.len(), 3, "{got:?}");
        assert_eq!(got[0]["message"]["content"], "Hel");
        assert_eq!(got[0]["model"], "m:latest");
        assert_eq!(got[0]["done"], false);
        let done = &got[2];
        assert_eq!(done["done"], true);
        assert_eq!(done["done_reason"], "stop");
        assert_eq!(done["message"]["content"], "");
        assert_eq!(done["prompt_eval_count"], 7);
        assert_eq!(done["eval_count"], 3);
        for key in [
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        ] {
            assert!(done[key].is_u64(), "{key}: {done}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_split_tool_call_goes_out_whole_with_object_arguments() -> anyhow::Result<()> {
        let pieces = [
            "<tool_call>{\"name\":\"weather\",",
            "\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
        ];
        let got = run(
            setup(Kind::Chat, false, true),
            items(&pieces, FinishReason::Stop),
        )
        .await?;
        let calls: Vec<&Value> = got
            .iter()
            .filter_map(|line| line["message"].get("tool_calls"))
            .collect();
        assert_eq!(calls.len(), 1, "{got:?}");
        assert_eq!(
            *calls[0],
            json!([{"function": {"name": "weather", "arguments": {"city": "Paris"}}}])
        );
        let done = got.last().unwrap();
        assert_eq!(done["done_reason"], "stop");
        assert!(got.iter().all(|l| l["message"]["content"] == ""), "{got:?}");
        Ok(())
    }

    #[tokio::test]
    async fn reasoning_goes_in_thinking_never_in_content() -> anyhow::Result<()> {
        let pieces = ["<think>plan", "</think>Answer"];
        let got = run(
            setup(Kind::Generate, true, false),
            items(&pieces, FinishReason::Length),
        )
        .await?;
        let thinking: String = got.iter().filter_map(|l| l["thinking"].as_str()).collect();
        let response: String = got.iter().filter_map(|l| l["response"].as_str()).collect();
        assert_eq!(thinking, "plan");
        assert_eq!(response, "Answer");
        assert_eq!(got.last().unwrap()["done_reason"], "length");
        Ok(())
    }

    #[tokio::test]
    async fn an_error_token_ends_the_stream_without_a_done_object() -> anyhow::Result<()> {
        let items = vec![
            StreamToken::text("Hi\n".into(), None),
            StreamToken::text("{\"name\":\"weather\",\"arguments\":{".into(), None),
            StreamToken::error("decode error".into()),
        ];
        let got = run(setup(Kind::Chat, false, true), items).await?;
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0]["message"]["content"], "Hi");
        assert_eq!(got[1], json!({"error": "decode error"}));
        assert!(got.iter().all(|l| l.get("done") != Some(&json!(true))));
        assert!(got.iter().all(|l| l["message"].get("tool_calls").is_none()));
        Ok(())
    }

    #[test]
    fn unparsed_arguments_stay_a_string() {
        let call = ToolCall {
            id: "c".into(),
            r#type: "function".into(),
            function: crate::server::tools::ToolCallFunction {
                name: "w".into(),
                arguments: "not json".into(),
            },
        };
        assert_eq!(ollama_call(call).function.arguments, json!("not json"));
    }
}
