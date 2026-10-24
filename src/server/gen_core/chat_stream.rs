//! The chat completions SSE stream, with reasoning split and logprobs.

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use futures::StreamExt;
use serde::Serialize;
use std::convert::Infallible;
use std::pin::Pin;

use super::gen_types::LogprobResult;
use super::reasoning::ReasoningParser;
use super::stream_usage::{usage_event, ChunkUsage};
use super::streaming::{error_event, StreamToken};

/// Chat completion streaming delta
#[derive(Serialize)]
pub struct ChatStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Serialize)]
pub struct ChatStreamChoice {
    pub index: usize,
    pub delta: ChatStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobResult>,
}

impl ChatStreamChoice {
    /// The only choice of a chunk, at index 0.
    fn new(
        delta: ChatStreamDelta,
        finish_reason: Option<String>,
        logprobs: Option<LogprobResult>,
    ) -> Self {
        Self {
            index: 0,
            delta,
            finish_reason,
            logprobs,
        }
    }
}

#[derive(Serialize)]
pub struct ChatStreamChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
    #[serde(skip_serializing_if = "ChunkUsage::is_absent")]
    pub usage: ChunkUsage,
}

/// The `object` of a chat stream chunk.
pub(in crate::server) const CHAT_CHUNK_OBJECT: &str = "chat.completion.chunk";

/// Create SSE stream for chat completions (true streaming, no buffering).
///
/// With `reasoning` set, each token is split into `delta.reasoning_content`
/// and `delta.content`; without it, every token is `delta.content`.
///
/// An item with logprobs puts one entry in `choices[0].logprobs.content`.
/// The stop-sequence matcher and the reasoning parser can hold text back
/// behind a token. That token's entry still goes in the chunk sent when the
/// token arrives, with an empty or partial `delta`. The held text follows
/// in a later chunk. No entry is dropped or sent twice. The final chunk
/// carries the last token's entry beside its `finish_reason`.
///
/// With `include_usage` set, every chunk carries `"usage": null`. After the
/// final chunk, one more chunk with empty `choices` carries the token counts.
/// A stream that ends on an error sends no usage chunk.
pub fn create_chat_stream(
    id: String,
    model: String,
    tokens: Pin<Box<dyn Stream<Item = StreamToken> + Send>>,
    reasoning: Option<ReasoningParser>,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();
        let mut tokens = tokens;
        let mut reasoning = reasoning;
        let usage = ChunkUsage::for_request(include_usage);
        let chunk = |delta, finish_reason, logprobs| ChatStreamChunk {
            id: id.clone(),
            object: CHAT_CHUNK_OBJECT.to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatStreamChoice::new(delta, finish_reason, logprobs)],
            usage,
        };

        // First chunk with role
        let role = ChatStreamDelta {
            role: Some("assistant".to_string()),
            content: None,
            reasoning_content: None,
        };
        if let Ok(data) = serde_json::to_string(&chunk(role, None, None)) {
            yield Ok(Event::default().data(data));
        }

        // Content chunks
        while let Some(token) = tokens.next().await {
            if let Some(ref err) = token.error {
                if let Some(evt) = error_event(err) {
                    yield Ok(evt);
                }
                break;
            }

            let is_final = token.finish_reason.is_some();
            let finish_str = token.finish_reason.map(|r| r.as_str().to_string());
            let logprobs = token.logprobs();

            let (reasoning_text, content_text) = match reasoning.as_mut() {
                Some(parser) => {
                    let (mut r, mut c) = parser.feed(&token.text);
                    if is_final {
                        let (tail_r, tail_c) = parser.finish();
                        r.push_str(&tail_r);
                        c.push_str(&tail_c);
                    }
                    (r, c)
                }
                None => (String::new(), token.text),
            };
            // Held-back text sends nothing, unless the token's logprob must go out now.
            let held_back = reasoning_text.is_empty() && content_text.is_empty();
            if !is_final && held_back && reasoning.is_some() && logprobs.is_none() {
                continue;
            }

            let delta = ChatStreamDelta {
                role: None,
                content: (!content_text.is_empty()).then_some(content_text),
                reasoning_content: (!reasoning_text.is_empty()).then_some(reasoning_text),
            };
            if let Ok(data) = serde_json::to_string(&chunk(delta, finish_str, logprobs)) {
                yield Ok(Event::default().data(data));
            }

            if is_final {
                if include_usage {
                    let evt = usage_event(&id, CHAT_CHUNK_OBJECT, created, &model, token.usage);
                    if let Some(evt) = evt {
                        yield Ok(evt);
                    }
                }
                break;
            }
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::FinishReason;
    use crate::server::reasoning::ReasoningTags;
    use crate::server::stream_usage::StreamUsage;
    use crate::server::streaming::sse_test::{
        assert_usage_tail, body, entry, events, expected_entry, feed, token,
    };
    use serde_json::json;

    /// `<th` and `</` sit inside the reasoning tags, so the parser holds
    /// them back. Their chunks still carry their entries.
    #[tokio::test]
    async fn chat_chunks_carry_each_entry_when_its_token_arrives() -> anyhow::Result<()> {
        let pieces = ["<th", "ink>Plan", "</", "think>Hi", "!"];
        let items = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let last = i + 1 == pieces.len();
                token(i as u32 + 1, p, last.then_some(FinishReason::Stop))
            })
            .collect();
        let parser = ReasoningParser::new(ReasoningTags::THINK, false);
        let sse = create_chat_stream("c".into(), "m".into(), feed(items), Some(parser), false);
        let got = events(sse).await?;

        assert_eq!(got.len(), 2 + pieces.len(), "{got:?}");
        assert_eq!(got[0]["choices"][0]["delta"], json!({"role": "assistant"}));
        assert!(entry(&got[0]).is_none());
        let deltas = [
            json!({}),
            json!({"reasoning_content": "Plan"}),
            json!({}),
            json!({"content": "Hi"}),
            json!({"content": "!"}),
        ];
        for (i, (piece, delta)) in pieces.iter().zip(deltas).enumerate() {
            let chunk = &got[i + 1];
            assert_eq!(chunk["choices"][0]["delta"], delta, "{chunk}");
            assert_eq!(entry(chunk), Some(&expected_entry(i as u32 + 1, piece)));
        }
        assert_eq!(got[pieces.len()]["choices"][0]["finish_reason"], "stop");
        assert_eq!(got[pieces.len() + 1], json!("[DONE]"));
        Ok(())
    }

    #[tokio::test]
    async fn chat_without_logprobs_skips_held_back_tokens() -> anyhow::Result<()> {
        let items = vec![
            StreamToken::text("<th".into(), None),
            StreamToken::text("ink>a</think>b".into(), Some(FinishReason::Eos)),
        ];
        let parser = ReasoningParser::new(ReasoningTags::THINK, false);
        let sse = create_chat_stream("c".into(), "m".into(), feed(items), Some(parser), false);
        let got = events(sse).await?;
        assert_eq!(got.len(), 3, "{got:?}");
        assert_eq!(
            got[1]["choices"][0]["delta"],
            json!({"reasoning_content": "a", "content": "b"})
        );
        assert!(got[1]["choices"][0].get("logprobs").is_none());
        Ok(())
    }

    #[tokio::test]
    async fn chat_error_item_sends_one_error_event_and_ends_the_stream() -> anyhow::Result<()> {
        let items = vec![
            token(1, "a", None),
            StreamToken::error("decode error".into()),
        ];
        let sse = create_chat_stream("c".into(), "m".into(), feed(items), None, true);
        let got = events(sse).await?;
        assert_eq!(
            got.len(),
            4,
            "an error stream sends no usage chunk: {got:?}"
        );
        assert_eq!(got[2]["error"]["message"], "decode error");
        assert_eq!(got[3], json!("[DONE]"));
        Ok(())
    }

    /// A held-back reasoning prefix, then a final item with its producer's counts.
    fn finished_items() -> Vec<StreamToken> {
        vec![
            StreamToken::text("<th".into(), None),
            StreamToken::text("ink>a</think>b".into(), Some(FinishReason::Stop))
                .with_usage(StreamUsage::new(9, 2)),
        ]
    }

    #[tokio::test]
    async fn include_usage_adds_a_usage_chunk_before_done() -> anyhow::Result<()> {
        let parser = ReasoningParser::new(ReasoningTags::THINK, false);
        let items = feed(finished_items());
        let sse = create_chat_stream("c".into(), "m".into(), items, Some(parser), true);
        let got = events(sse).await?;
        assert_eq!(got.len(), 4, "{got:?}");
        assert_eq!(got[0]["choices"][0]["delta"], json!({"role": "assistant"}));
        assert_eq!(got[1]["choices"][0]["finish_reason"], "stop");
        assert_usage_tail(&got, "chat.completion.chunk", StreamUsage::new(9, 2));
        Ok(())
    }

    #[tokio::test]
    async fn without_include_usage_no_chunk_names_usage() -> anyhow::Result<()> {
        let parser = ReasoningParser::new(ReasoningTags::THINK, false);
        let items = feed(finished_items());
        let raw = body(create_chat_stream(
            "c".into(),
            "m".into(),
            items,
            Some(parser),
            false,
        ))
        .await?;
        assert!(!raw.contains("usage"), "{raw}");
        assert_eq!(raw.matches("data:").count(), 3, "{raw}");
        Ok(())
    }
}
