//! The item a streamed generation sends, and the SSE error event.
//!
//! The completions stream lives in `completion_stream`, the chat stream in
//! `chat_choices`.

use axum::response::sse::Event;
use futures::stream::Stream;
use std::pin::Pin;

use super::gen_types::{logprob_entry, LogprobResult};
use super::stream_usage::StreamUsage;
use crate::engine::{FinishReason, GeneratedToken, TokenLogprob};

/// The items one generation streams.
pub(in crate::server) type TokenStream = Pin<Box<dyn Stream<Item = StreamToken> + Send>>;

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
    /// The request's token counts. Every producer sets it on the item that
    /// carries `finish_reason`, and on no other item.
    pub usage: Option<StreamUsage>,
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
            usage: None,
        }
    }

    /// `self` with the request's token counts `usage`.
    pub fn with_usage(mut self, usage: StreamUsage) -> Self {
        self.usage = Some(usage);
        self
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

    /// The raw SSE body of `sse`.
    pub(crate) async fn body(sse: impl IntoResponse) -> anyhow::Result<String> {
        let body = axum::body::to_bytes(sse.into_response().into_body(), usize::MAX).await?;
        Ok(String::from_utf8(body.to_vec())?)
    }

    /// Check the tail of a stream that asked for usage.
    ///
    /// `got` ends with the usage chunk, then `[DONE]`. The usage chunk has
    /// `object`, empty `choices` and `usage` equal to `want`. Every chunk
    /// before it carries `"usage": null`.
    pub(crate) fn assert_usage_tail(got: &[Value], object: &str, want: super::StreamUsage) {
        let [chunks @ .., usage, done] = got else {
            panic!("a usage stream has at least two events: {got:?}");
        };
        assert_eq!(*done, json!("[DONE]"));
        assert_eq!(usage["object"], object, "{usage}");
        assert_eq!(usage["choices"], json!([]), "{usage}");
        assert_eq!(
            usage["usage"],
            json!({
                "prompt_tokens": want.prompt_tokens,
                "completion_tokens": want.completion_tokens,
                "total_tokens": want.prompt_tokens + want.completion_tokens,
            })
        );
        assert!(!chunks.is_empty(), "content chunks precede the usage chunk");
        for chunk in chunks {
            assert_eq!(chunk.get("usage"), Some(&Value::Null), "{chunk}");
        }
    }

    /// Each `data:` payload of `sse`, in order. `[DONE]` reads as a JSON string.
    pub(crate) async fn events(sse: impl IntoResponse) -> anyhow::Result<Vec<Value>> {
        parse_events(&body(sse).await?)
    }

    /// Each `data:` payload of the raw SSE body `text`, as [`events`] reads it.
    pub(crate) fn parse_events(text: &str) -> anyhow::Result<Vec<Value>> {
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
