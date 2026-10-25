//! The chat completions SSE stream of one or more choices.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use futures::StreamExt;

use super::chat_chunks::ChatChunker;
use super::chat_stream::{ChatStreamChoice, ChatStreamChunk, ChatStreamDelta, CHAT_CHUNK_OBJECT};
use super::reasoning::ReasoningParser;
use super::stream_usage::{usage_event, ChoicesUsage, ChunkUsage};
use super::streaming::{error_event, TokenStream};
use crate::server::tool_parser::ToolCallParser;

/// One choice of a chat stream: its items and the parsers its text goes through.
///
/// Each choice holds its own parsers, so a held tag or call never crosses
/// into another choice.
pub(in crate::server) struct ChoiceStream {
    pub tokens: TokenStream,
    pub reasoning: Option<ReasoningParser>,
    pub tools: Option<ToolCallParser>,
}

/// The SSE stream of `choices`, choice `i` at `index` `i`.
///
/// The choices stream in order. Each one opens with its own `role` chunk
/// and ends with its own `finish_reason` chunk. A choice ends when its items
/// end or its final item arrives.
///
/// With `reasoning` set, each token is split into `delta.reasoning_content`
/// and `delta.content`; without it, every token is `delta.content`.
///
/// With `tool_parser` set, the content half goes through it. Content streams
/// live. A tool call is held until its boundary arrives, then streams as
/// `delta.tool_calls`: one entry with `id`, `type`, `function.name` and
/// empty `arguments`, then entries with fragments of `arguments`. A stream
/// that sent a call finishes with `tool_calls`, whatever the upstream finish
/// reason. A call still open when the output ends goes out as content.
///
/// An item with logprobs puts one entry in the `logprobs.content` of its choice.
/// The stop-sequence matcher, the reasoning parser and the tool-call parser
/// can hold text back behind a token. That token's entry still goes in the
/// chunk sent when the token arrives, with an empty or partial `delta`. The
/// held text follows in a later chunk. No entry is dropped or sent twice.
/// The final token's entry goes out in the first chunk that token sends.
///
/// With `include_usage` set, every chunk carries `"usage": null`. After the
/// last choice, one more chunk with empty `choices` carries the token counts.
/// `prompt_tokens` counts the prompt once. `completion_tokens` sums every
/// choice. A choice that ends without its counts drops the usage chunk.
///
/// An error item sends one error event and `[DONE]`. The later choices are
/// dropped unread, held calls are dropped, and no usage chunk goes out.
pub(in crate::server) fn create_choices_stream(
    id: String,
    model: String,
    choices: Vec<ChoiceStream>,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();
        let usage = ChunkUsage::for_request(include_usage);
        let chunk = |choice| ChatStreamChunk {
            id: id.clone(),
            object: CHAT_CHUNK_OBJECT.to_string(),
            created,
            model: model.clone(),
            choices: vec![choice],
            usage,
        };
        let mut counts = ChoicesUsage::default();
        let mut failed = false;

        'choices: for (index, choice) in choices.into_iter().enumerate() {
            let ChoiceStream { mut tokens, reasoning, tools } = choice;
            let mut chunker = ChatChunker::new(reasoning, tools);
            let role = ChatStreamDelta {
                role: Some("assistant".to_string()),
                ..ChatStreamDelta::default()
            };
            let mut opening = ChatStreamChoice::new(role, None, None);
            opening.index = index;
            if let Ok(data) = serde_json::to_string(&chunk(opening)) {
                yield Ok(Event::default().data(data));
            }

            let mut finished = false;
            while let Some(token) = tokens.next().await {
                if let Some(ref err) = token.error {
                    if let Some(evt) = error_event(err) {
                        yield Ok(evt);
                    }
                    failed = true;
                    break 'choices;
                }

                let is_final = token.finish_reason.is_some();
                let final_usage = token.usage;
                for mut out in chunker.token(token) {
                    out.index = index;
                    if let Ok(data) = serde_json::to_string(&chunk(out)) {
                        yield Ok(Event::default().data(data));
                    }
                }

                if is_final {
                    counts.finished(final_usage);
                    finished = true;
                    break;
                }
            }
            if !finished {
                counts.unfinished();
            }
        }

        if include_usage && !failed {
            let final_usage = counts.total();
            if let Some(evt) = usage_event(&id, CHAT_CHUNK_OBJECT, created, &model, final_usage) {
                yield Ok(evt);
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

#[cfg(test)]
mod tests {
    use super::super::stream_usage::StreamUsage;
    use super::super::streaming::StreamToken;
    use super::*;
    use crate::engine::FinishReason;
    use crate::server::streaming::sse_test::{assert_usage_tail, events, feed};
    use serde_json::json;

    /// A choice that streams `pieces`, the last with `completion` counts over a 5-token prompt.
    fn choice(pieces: &[&str], completion: usize) -> ChoiceStream {
        let items = pieces
            .iter()
            .enumerate()
            .map(|(i, piece)| {
                if i + 1 == pieces.len() {
                    StreamToken::text(piece.to_string(), Some(FinishReason::Stop))
                        .with_usage(StreamUsage::new(5, completion))
                } else {
                    StreamToken::text(piece.to_string(), None)
                }
            })
            .collect();
        ChoiceStream {
            tokens: feed(items),
            reasoning: None,
            tools: None,
        }
    }

    #[tokio::test]
    async fn each_choice_streams_at_its_index_and_usage_sums_once() -> anyhow::Result<()> {
        let choices = vec![choice(&["a", "b"], 2), choice(&["c"], 1)];
        let sse = create_choices_stream("c".into(), "m".into(), choices, true);
        let got = events(sse).await?;

        assert_eq!(got.len(), 7, "{got:?}");
        let indices: Vec<_> = got[..5]
            .iter()
            .map(|e| e["choices"][0]["index"].clone())
            .collect();
        assert_eq!(indices, [json!(0), json!(0), json!(0), json!(1), json!(1)]);
        assert_eq!(got[0]["choices"][0]["delta"], json!({"role": "assistant"}));
        assert_eq!(got[3]["choices"][0]["delta"], json!({"role": "assistant"}));
        assert_eq!(got[2]["choices"][0]["finish_reason"], "stop");
        assert_eq!(got[4]["choices"][0]["delta"], json!({"content": "c"}));
        assert_eq!(got[4]["choices"][0]["finish_reason"], "stop");
        assert_usage_tail(&got, "chat.completion.chunk", StreamUsage::new(5, 3));
        Ok(())
    }

    #[tokio::test]
    async fn an_error_in_one_choice_ends_the_stream_without_usage() -> anyhow::Result<()> {
        let broken = ChoiceStream {
            tokens: feed(vec![StreamToken::error("decode error".into())]),
            reasoning: None,
            tools: None,
        };
        let choices = vec![choice(&["a"], 1), broken, choice(&["never"], 1)];
        let sse = create_choices_stream("c".into(), "m".into(), choices, true);
        let got = events(sse).await?;

        assert_eq!(got.len(), 5, "{got:?}");
        assert_eq!(got[2]["choices"][0]["index"], 1);
        assert_eq!(got[3]["error"]["message"], "decode error");
        assert_eq!(got[4], json!("[DONE]"));
        Ok(())
    }

    #[tokio::test]
    async fn a_choice_without_counts_drops_the_usage_chunk() -> anyhow::Result<()> {
        let bare = ChoiceStream {
            tokens: feed(vec![StreamToken::text("x".into(), Some(FinishReason::Eos))]),
            reasoning: None,
            tools: None,
        };
        let choices = vec![choice(&["a"], 1), bare];
        let sse = create_choices_stream("c".into(), "m".into(), choices, true);
        let got = events(sse).await?;

        assert_eq!(got.len(), 5, "{got:?}");
        assert!(got.iter().all(|e| e.get("choices") != Some(&json!([]))));
        assert_eq!(got[4], json!("[DONE]"));
        Ok(())
    }
}
