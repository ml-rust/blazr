//! The completions SSE stream of one or more choices.

use std::convert::Infallible;

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use futures::StreamExt;
use serde::Serialize;

use super::gen_types::LogprobResult;
use super::stream_usage::{usage_event, ChoicesUsage, ChunkUsage};
use super::streaming::{error_event, TokenStream};
use crate::engine::FinishReason;

/// SSE delta for streaming completions
#[derive(Serialize)]
pub struct StreamDelta {
    pub text: String,
}

#[derive(Serialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: StreamDelta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobResult>,
}

#[derive(Serialize)]
pub struct StreamCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "ChunkUsage::is_absent")]
    pub usage: ChunkUsage,
}

/// The `object` of a completions stream chunk.
const COMPLETION_CHUNK_OBJECT: &str = "text_completion.chunk";

/// The one-choice case of [`create_completion_choices_stream`].
pub fn create_completion_stream(
    id: String,
    model: String,
    tokens: TokenStream,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    create_completion_choices_stream(id, model, vec![tokens], include_usage)
}

/// The SSE stream of `choices`, choice `i` at `index` `i`.
///
/// The choices stream in order, with no interleaving. Each item becomes one
/// chunk of its choice. A choice ends when its items end or its final item
/// arrives. Items that end with no final item end the choice with one more
/// chunk: empty text and `length`, as a complete response reports them.
///
/// An item with logprobs puts one entry in `choices[0].logprobs.content`.
/// A stop-sequence prefix holds text back, so a chunk can carry a token's
/// logprob with empty `delta.text`. The held text arrives in a later chunk.
/// Every token's entry appears once, in the chunk of the item that carries it.
///
/// With `include_usage` set, every chunk carries `"usage": null`. After the
/// last choice, one more chunk with empty `choices` carries the token counts.
/// `prompt_tokens` counts the prompt once. `completion_tokens` sums every
/// choice. A choice that ends without its counts drops the usage chunk.
///
/// An error item sends one error event and `[DONE]`. The later choices are
/// dropped unread, and no usage chunk goes out.
pub fn create_completion_choices_stream(
    id: String,
    model: String,
    choices: Vec<TokenStream>,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();
        let usage = ChunkUsage::for_request(include_usage);
        let mut counts = ChoicesUsage::default();
        let mut failed = false;

        'choices: for (index, mut tokens) in choices.into_iter().enumerate() {
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
                let logprobs = token.logprobs();
                let chunk = StreamCompletionChunk {
                    id: id.clone(),
                    object: COMPLETION_CHUNK_OBJECT.to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![StreamChoice {
                        index,
                        delta: StreamDelta { text: token.text },
                        finish_reason: token.finish_reason.map(|r| r.as_str().to_string()),
                        logprobs,
                    }],
                    usage,
                };
                if let Ok(data) = serde_json::to_string(&chunk) {
                    yield Ok(Event::default().data(data));
                }

                if is_final {
                    counts.finished(final_usage);
                    finished = true;
                    break;
                }
            }
            if !finished {
                // Items that end with no final one still close the choice.
                let chunk = StreamCompletionChunk {
                    id: id.clone(),
                    object: COMPLETION_CHUNK_OBJECT.to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![StreamChoice {
                        index,
                        delta: StreamDelta { text: String::new() },
                        finish_reason: Some(FinishReason::Length.as_str().to_string()),
                        logprobs: None,
                    }],
                    usage,
                };
                if let Ok(data) = serde_json::to_string(&chunk) {
                    yield Ok(Event::default().data(data));
                }
                counts.unfinished();
            }
        }

        if include_usage && !failed {
            let evt = usage_event(&id, COMPLETION_CHUNK_OBJECT, created, &model, counts.total());
            if let Some(evt) = evt {
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
    use crate::engine::{GeneratedToken, TokenLogprob};
    use crate::server::streaming::sse_test::{
        assert_usage_tail, body, entry, events, expected_entry, feed, token,
    };
    use serde_json::json;

    /// The second token completes a stop prefix the matcher held: its chunk
    /// has empty text and its own entry. The release token carries the text.
    #[tokio::test]
    async fn completion_chunks_carry_entries_beside_held_back_text() -> anyhow::Result<()> {
        let items = vec![
            token(1, "a", None),
            StreamToken::text(String::new(), None).with_logprobs(
                &GeneratedToken {
                    token_id: 2,
                    text: "E".into(),
                    logprob: Some(-1.0),
                    top_logprobs: Some(vec![TokenLogprob {
                        token_id: 2,
                        text: "E".into(),
                        logprob: -1.0,
                    }]),
                    finish_reason: None,
                },
                true,
            ),
            token(3, "Eb", Some(FinishReason::Length)),
        ];
        let sse = create_completion_stream("c".into(), "m".into(), feed(items), false);
        let got = events(sse).await?;

        assert_eq!(got.len(), 4, "{got:?}");
        assert_eq!(got[0]["choices"][0]["delta"]["text"], "a");
        assert_eq!(entry(&got[0]), Some(&expected_entry(1, "a")));
        assert_eq!(got[1]["choices"][0]["delta"]["text"], "");
        assert_eq!(entry(&got[1]), Some(&expected_entry(2, "E")));
        assert_eq!(got[2]["choices"][0]["delta"]["text"], "Eb");
        assert_eq!(got[2]["choices"][0]["finish_reason"], "length");
        assert_eq!(entry(&got[2]), Some(&expected_entry(3, "Eb")));
        assert_eq!(got[3], json!("[DONE]"));
        Ok(())
    }

    #[tokio::test]
    async fn completion_without_logprobs_has_no_member() -> anyhow::Result<()> {
        let items = vec![StreamToken::text("a".into(), Some(FinishReason::Eos))];
        let sse = create_completion_stream("c".into(), "m".into(), feed(items), false);
        let got = events(sse).await?;
        assert_eq!(got.len(), 2);
        assert!(got[0]["choices"][0].get("logprobs").is_none());
        assert_eq!(got[0]["choices"][0]["finish_reason"], "stop");
        Ok(())
    }

    #[tokio::test]
    async fn an_error_item_sends_one_error_event_and_ends_the_stream() -> anyhow::Result<()> {
        let items = vec![
            token(1, "a", None),
            StreamToken::error("decode error".into()),
            token(2, "never sent", None),
        ];
        let sse = create_completion_stream("c".into(), "m".into(), feed(items), true);
        let got = events(sse).await?;
        assert_eq!(
            got.len(),
            3,
            "an error stream sends no usage chunk: {got:?}"
        );
        assert_eq!(got[1]["error"]["message"], "decode error");
        assert_eq!(got[1]["error"]["type"], "server_error");
        assert_eq!(got[2], json!("[DONE]"));
        Ok(())
    }

    /// Two content chunks, the final one with the counts its producer set.
    fn finished_items() -> Vec<StreamToken> {
        vec![
            token(1, "a", None),
            token(2, "b", Some(FinishReason::Stop)).with_usage(StreamUsage::new(5, 3)),
        ]
    }

    #[tokio::test]
    async fn include_usage_adds_a_usage_chunk_before_done() -> anyhow::Result<()> {
        let sse = create_completion_stream("c".into(), "m".into(), feed(finished_items()), true);
        let got = events(sse).await?;
        assert_eq!(got.len(), 4, "{got:?}");
        assert_eq!(got[1]["choices"][0]["finish_reason"], "stop");
        assert_usage_tail(&got, "text_completion.chunk", StreamUsage::new(5, 3));
        assert_eq!(got[2]["id"], "c");
        assert_eq!(got[2]["model"], "m");
        assert_eq!(got[2]["created"], got[0]["created"]);
        Ok(())
    }

    #[tokio::test]
    async fn without_include_usage_no_chunk_names_usage() -> anyhow::Result<()> {
        let sse = create_completion_stream("c".into(), "m".into(), feed(finished_items()), false);
        let raw = body(sse).await?;
        assert!(!raw.contains("usage"), "{raw}");
        assert_eq!(raw.matches("data:").count(), 3, "{raw}");
        Ok(())
    }

    /// A choice that streams `pieces`, the last with `completion` counts over a 5-token prompt.
    fn choice(pieces: &[&str], completion: usize) -> TokenStream {
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
        feed(items)
    }

    #[tokio::test]
    async fn each_choice_streams_at_its_index_and_usage_sums_once() -> anyhow::Result<()> {
        let choices = vec![choice(&["a", "b"], 2), choice(&["c"], 1)];
        let sse = create_completion_choices_stream("c".into(), "m".into(), choices, true);
        let got = events(sse).await?;

        assert_eq!(got.len(), 5, "{got:?}");
        let indices: Vec<_> = got[..3]
            .iter()
            .map(|e| e["choices"][0]["index"].clone())
            .collect();
        assert_eq!(indices, [json!(0), json!(0), json!(1)]);
        assert_eq!(got[1]["choices"][0]["finish_reason"], "stop");
        assert_eq!(got[2]["choices"][0]["delta"]["text"], "c");
        assert_eq!(got[2]["choices"][0]["finish_reason"], "stop");
        assert_usage_tail(&got, "text_completion.chunk", StreamUsage::new(5, 3));
        Ok(())
    }

    #[tokio::test]
    async fn an_error_in_one_choice_ends_the_stream_without_usage() -> anyhow::Result<()> {
        let broken = feed(vec![StreamToken::error("decode error".into())]);
        let choices = vec![choice(&["a"], 1), broken, choice(&["never"], 1)];
        let sse = create_completion_choices_stream("c".into(), "m".into(), choices, true);
        let got = events(sse).await?;

        assert_eq!(got.len(), 3, "{got:?}");
        assert_eq!(got[0]["choices"][0]["index"], 0);
        assert_eq!(got[1]["error"]["message"], "decode error");
        assert_eq!(got[2], json!("[DONE]"));
        Ok(())
    }

    #[tokio::test]
    async fn a_choice_without_a_final_item_drops_the_usage_chunk() -> anyhow::Result<()> {
        let cut = feed(vec![StreamToken::text("x".into(), None)]);
        let choices = vec![choice(&["a"], 1), cut];
        let sse = create_completion_choices_stream("c".into(), "m".into(), choices, true);
        let got = events(sse).await?;

        // "a", "x", the closing chunk, [DONE]. No usage chunk.
        assert_eq!(got.len(), 4, "{got:?}");
        assert!(got.iter().all(|e| e.get("choices") != Some(&json!([]))));
        assert!(got[1]["choices"][0]["finish_reason"].is_null());
        assert_eq!(got[2]["choices"][0]["index"], 1);
        assert_eq!(got[2]["choices"][0]["delta"]["text"], "");
        assert_eq!(got[2]["choices"][0]["finish_reason"], "length");
        assert_eq!(got[3], json!("[DONE]"));
        Ok(())
    }
}
