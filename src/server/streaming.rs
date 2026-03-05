use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use futures::StreamExt;
use serde::Serialize;
use std::convert::Infallible;
use std::pin::Pin;

use crate::engine::FinishReason;

/// Token with optional finish reason, sent through the streaming channel
pub struct StreamToken {
    pub text: String,
    pub finish_reason: Option<FinishReason>,
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

/// Create SSE stream for chat completions (true streaming, no buffering)
pub fn create_chat_stream(
    id: String,
    model: String,
    tokens: Pin<Box<dyn Stream<Item = StreamToken> + Send>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();
        let mut tokens = tokens;

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
                },
                finish_reason: None,
            }],
        };

        if let Ok(data) = serde_json::to_string(&first_chunk) {
            yield Ok(Event::default().data(data));
        }

        // Content chunks
        while let Some(token) = tokens.next().await {
            let is_final = token.finish_reason.is_some();
            let finish_str = token.finish_reason.map(|r| r.as_str().to_string());

            let chunk = ChatStreamChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChatStreamChoice {
                    index: 0,
                    delta: ChatStreamDelta {
                        role: None,
                        content: if token.text.is_empty() { None } else { Some(token.text) },
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
