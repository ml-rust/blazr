use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use serde::Serialize;
use std::convert::Infallible;

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

/// Create SSE stream from token iterator
pub fn create_completion_stream(
    id: String,
    model: String,
    tokens: impl Iterator<Item = String> + Send + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();

        for token in tokens {
            let chunk = StreamCompletionChunk {
                id: id.clone(),
                object: "text_completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: StreamDelta { text: token },
                    finish_reason: None,
                }],
            };

            let data = serde_json::to_string(&chunk).unwrap_or_default();
            yield Ok(Event::default().data(data));
        }

        // Send final chunk with finish_reason
        let final_chunk = StreamCompletionChunk {
            id: id.clone(),
            object: "text_completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta { text: String::new() },
                finish_reason: Some("stop".to_string()),
            }],
        };

        let data = serde_json::to_string(&final_chunk).unwrap_or_default();
        yield Ok(Event::default().data(data));

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

/// Create SSE stream for chat completions
pub fn create_chat_stream(
    id: String,
    model: String,
    tokens: impl Iterator<Item = String> + Send + 'static,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = async_stream::stream! {
        let created = chrono::Utc::now().timestamp();

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

        let data = serde_json::to_string(&first_chunk).unwrap_or_default();
        yield Ok(Event::default().data(data));

        // Content chunks
        for token in tokens {
            let chunk = ChatStreamChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChatStreamChoice {
                    index: 0,
                    delta: ChatStreamDelta {
                        role: None,
                        content: Some(token),
                    },
                    finish_reason: None,
                }],
            };

            let data = serde_json::to_string(&chunk).unwrap_or_default();
            yield Ok(Event::default().data(data));
        }

        // Final chunk
        let final_chunk = ChatStreamChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatStreamDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };

        let data = serde_json::to_string(&final_chunk).unwrap_or_default();
        yield Ok(Event::default().data(data));

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}
