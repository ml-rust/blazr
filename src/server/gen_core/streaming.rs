//! The item a streamed generation sends, and the completions SSE stream.
//!
//! The chat stream lives in `chat_stream`.

use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use futures::StreamExt;
use serde::Serialize;
use std::convert::Infallible;
use std::pin::Pin;

use super::gen_types::{logprob_entry, LogprobResult};
use crate::engine::{FinishReason, GeneratedToken, TokenLogprob};

/// Token with optional finish reason, sent through the streaming channel
pub struct StreamToken {
    pub text: String,
    pub finish_reason: Option<FinishReason>,
    /// If set, indicates an error occurred during generation
    pub error: Option<String>,
    /// The engine token this item carries, with its id, text and logprob.
    /// Set only when the request asks for logprobs.
    pub logprob: Option<TokenLogprob>,
    /// The top alternatives of `logprob`'s token, when the request asks.
    pub top_logprobs: Option<Vec<TokenLogprob>>,
}

impl StreamToken {
    /// A text item with no logprobs.
    pub fn text(text: String, finish_reason: Option<FinishReason>) -> Self {
        Self {
            text,
            finish_reason,
            error: None,
            logprob: None,
            top_logprobs: None,
        }
    }

    /// An item that reports a generation error and ends the stream.
    pub fn error(message: String) -> Self {
        Self {
            error: Some(message),
            ..Self::text(String::new(), None)
        }
    }

    /// `self` with the logprobs of `token` when `asked` is set.
    ///
    /// A missing logprob reads as negative infinity, as in the
    /// non-streaming response.
    pub fn with_logprobs(mut self, token: &GeneratedToken, asked: bool) -> Self {
        if asked {
            self.logprob = Some(TokenLogprob {
                token_id: token.token_id,
                text: token.text.clone(),
                logprob: token.logprob.unwrap_or(f32::NEG_INFINITY),
            });
            self.top_logprobs = token.top_logprobs.clone();
        }
        self
    }

    /// The `logprobs` member of the chunk that carries this item: one entry,
    /// in the format of the non-streaming response.
    pub(super) fn logprobs(&self) -> Option<LogprobResult> {
        self.logprob.as_ref().map(|lp| LogprobResult {
            content: vec![logprob_entry(
                &lp.text,
                Some(lp.logprob),
                self.top_logprobs.as_deref(),
            )],
        })
    }
}

/// Build an SSE error event from an error message
pub(in crate::server) fn error_event(err: &str) -> Option<Event> {
    let payload = serde_json::json!({
        "error": {
            "message": err,
            "type": "server_error"
        }
    });
    serde_json::to_string(&payload)
        .ok()
        .map(|data| Event::default().data(data))
}

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
}

/// Create SSE stream from an async token stream (true streaming, no buffering)
///
/// Each item becomes one chunk. An item with logprobs puts one entry in
/// `choices[0].logprobs.content`. A stop-sequence prefix holds text back,
/// so a chunk can carry a token's logprob with empty `delta.text`. The
/// held text arrives in a later chunk. Every token's entry appears once,
/// in the chunk of the item that carries it.
pub fn create_completion_stream(
    id: String,
    model: String,
    tokens: Pin<Box<dyn Stream<Item = StreamToken> + Send>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();
        let mut tokens = tokens;

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

            let chunk = StreamCompletionChunk {
                id: id.clone(),
                object: "text_completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta { text: token.text },
                    finish_reason: finish_str,
                    logprobs,
                }],
            };

            if let Ok(data) = serde_json::to_string(&chunk) {
                yield Ok(Event::default().data(data));
            }

            if is_final {
                break;
            }
        }

        // Send [DONE] marker
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

/// Helpers that build stream items and read an SSE response back as JSON,
/// for the stream tests.
#[cfg(test)]
pub(crate) mod sse_test {
    use std::pin::Pin;

    use axum::response::IntoResponse;
    use futures::stream::Stream;
    use serde_json::{json, Value};

    use super::StreamToken;
    use crate::engine::{FinishReason, GeneratedToken, TokenLogprob};

    /// An item for engine token `id` with text `text` and logprob `-0.5 * id`.
    pub(crate) fn token(id: u32, text: &str, finish: Option<FinishReason>) -> StreamToken {
        let engine = GeneratedToken {
            token_id: id,
            text: text.to_string(),
            logprob: Some(-0.5 * id as f32),
            top_logprobs: Some(vec![TokenLogprob {
                token_id: id,
                text: text.to_string(),
                logprob: -0.5 * id as f32,
            }]),
            finish_reason: finish,
        };
        StreamToken::text(text.to_string(), finish).with_logprobs(&engine, true)
    }

    /// `items` as the stream a builder takes.
    pub(crate) fn feed(items: Vec<StreamToken>) -> Pin<Box<dyn Stream<Item = StreamToken> + Send>> {
        Box::pin(futures::stream::iter(items))
    }

    /// The one logprob entry of `chunk`, or `None` when it carries none.
    pub(crate) fn entry(chunk: &Value) -> Option<&Value> {
        let content = chunk["choices"][0].get("logprobs")?["content"].as_array()?;
        assert_eq!(content.len(), 1, "one entry per chunk: {chunk}");
        content.first()
    }

    /// The entry a [`token`] item with `id` and `text` serializes to.
    pub(crate) fn expected_entry(id: u32, text: &str) -> Value {
        let bytes: Vec<u8> = text.as_bytes().to_vec();
        let logprob = -0.5 * id as f32;
        json!({
            "token": text,
            "logprob": logprob,
            "bytes": bytes,
            "top_logprobs": [{"token": text, "logprob": logprob, "bytes": bytes}],
        })
    }

    /// Each `data:` payload of `sse`, in order. `[DONE]` reads as a JSON string.
    pub(crate) async fn events(sse: impl IntoResponse) -> anyhow::Result<Vec<Value>> {
        let body = axum::body::to_bytes(sse.into_response().into_body(), usize::MAX).await?;
        let text = String::from_utf8(body.to_vec())?;
        text.split("\n\n")
            .filter(|event| !event.trim().is_empty())
            .map(|event| {
                let data = event
                    .lines()
                    .find_map(|line| line.strip_prefix("data:"))
                    .map(str::trim_start)
                    .ok_or_else(|| anyhow::anyhow!("SSE event {event:?} has no data line"))?;
                if data == "[DONE]" {
                    return Ok(Value::String(data.to_string()));
                }
                Ok(serde_json::from_str(data)?)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::sse_test::{entry, events, expected_entry, feed, token};
    use super::*;
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
        let sse = create_completion_stream("c".into(), "m".into(), feed(items));
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
        let sse = create_completion_stream("c".into(), "m".into(), feed(items));
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
        let sse = create_completion_stream("c".into(), "m".into(), feed(items));
        let got = events(sse).await?;
        assert_eq!(got.len(), 3, "{got:?}");
        assert_eq!(got[1]["error"]["message"], "decode error");
        assert_eq!(got[1]["error"]["type"], "server_error");
        assert_eq!(got[2], json!("[DONE]"));
        Ok(())
    }
}
