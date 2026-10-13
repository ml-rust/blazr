use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use futures::StreamExt;
use serde::Serialize;
use std::convert::Infallible;
use std::pin::Pin;

use super::reasoning::ReasoningParser;
use crate::engine::FinishReason;

/// Token with optional finish reason, sent through the streaming channel
pub struct StreamToken {
    pub text: String,
    pub finish_reason: Option<FinishReason>,
    /// If set, indicates an error occurred during generation
    pub error: Option<String>,
}

/// Build an SSE error event from an error message
fn error_event(err: &str) -> Option<Event> {
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

            let chunk = StreamCompletionChunk {
                id: id.clone(),
                object: "text_completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta { text: token.text },
                    finish_reason: finish_str,
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
}

#[derive(Serialize)]
pub struct ChatStreamChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatStreamChoice>,
}

/// Create SSE stream for chat completions (true streaming, no buffering).
///
/// With `reasoning` set, each token is split into `delta.reasoning_content`
/// and `delta.content`; without it, every token is `delta.content`.
pub fn create_chat_stream(
    id: String,
    model: String,
    tokens: Pin<Box<dyn Stream<Item = StreamToken> + Send>>,
    reasoning: Option<ReasoningParser>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();
        let mut tokens = tokens;
        let mut reasoning = reasoning;

        // First chunk with role
        let first_chunk = ChatStreamChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatStreamDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    reasoning_content: None,
                },
                finish_reason: None,
            }],
        };

        if let Ok(data) = serde_json::to_string(&first_chunk) {
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
            let held_back = reasoning_text.is_empty() && content_text.is_empty();
            if !is_final && held_back && reasoning.is_some() {
                continue;
            }

            let chunk = ChatStreamChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChatStreamChoice {
                    index: 0,
                    delta: ChatStreamDelta {
                        role: None,
                        content: if content_text.is_empty() { None } else { Some(content_text) },
                        reasoning_content: if reasoning_text.is_empty() {
                            None
                        } else {
                            Some(reasoning_text)
                        },
                    },
                    finish_reason: finish_str,
                }],
            };

            if let Ok(data) = serde_json::to_string(&chunk) {
                yield Ok(Event::default().data(data));
            }

            if is_final {
                break;
            }
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}
