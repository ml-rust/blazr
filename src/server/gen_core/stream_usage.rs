//! Token counts of a streamed generation and the `stream_options.include_usage` chunk.

use axum::response::sse::Event;
use serde::{Serialize, Serializer};
use serde_json::json;

/// The token counts of one streamed generation.
///
/// `prompt_tokens` counts the prompt ids the model reads. `completion_tokens`
/// counts every token the engine generates, held-back stop text included.
/// The non-streaming `usage` counts the same way.
#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl StreamUsage {
    /// The counts of `prompt_tokens` prompt ids and `completion_tokens` generated tokens.
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    /// These counts with one more choice of the same prompt: the prompt
    /// counts once, and `next`'s completion adds to the sum.
    pub fn with_choice(self, next: StreamUsage) -> Self {
        Self::new(
            self.prompt_tokens,
            self.completion_tokens + next.completion_tokens,
        )
    }
}

/// The `usage` member of a content chunk.
///
/// `Absent` omits the member. `Null` writes `"usage": null`, as an OpenAI
/// stream does on every chunk but the usage chunk when the request sets
/// `stream_options.include_usage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkUsage {
    Absent,
    Null,
}

impl ChunkUsage {
    /// `Null` when the request asks for a usage chunk, else `Absent`.
    pub fn for_request(include_usage: bool) -> Self {
        if include_usage {
            Self::Null
        } else {
            Self::Absent
        }
    }

    /// Whether the member is left out. Use with `skip_serializing_if`.
    pub fn is_absent(&self) -> bool {
        *self == Self::Absent
    }
}

impl Serialize for ChunkUsage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_none()
    }
}

/// The usage chunk: `choices` empty and `usage` set to the counts of the
/// final item.
///
/// `final_usage` is the `usage` of the item that carries the finish reason.
/// Every producer sets it. An item without it logs an error and sends no
/// usage chunk, so the stream never reports an invented count.
pub(in crate::server) fn usage_event(
    id: &str,
    object: &str,
    created: i64,
    model: &str,
    final_usage: Option<StreamUsage>,
) -> Option<Event> {
    let Some(usage) = final_usage else {
        tracing::error!(
            "stream {id}: the final item has no token counts; the usage chunk is not sent"
        );
        return None;
    };
    let chunk = json!({
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage,
    });
    Some(Event::default().data(chunk.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn usage_totals_prompt_and_completion() {
        let usage = StreamUsage::new(7, 4);
        assert_eq!(usage.total_tokens, 11);
        let value = serde_json::to_value(usage).unwrap_or_default();
        assert_eq!(
            value,
            json!({"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11})
        );
    }

    #[test]
    fn a_second_choice_adds_its_completion_and_not_its_prompt() {
        let both = StreamUsage::new(7, 4).with_choice(StreamUsage::new(7, 3));
        assert_eq!(both, StreamUsage::new(7, 7));
    }

    #[derive(Serialize)]
    struct Chunk {
        #[serde(skip_serializing_if = "ChunkUsage::is_absent")]
        usage: ChunkUsage,
    }

    #[test]
    fn chunk_usage_is_null_or_left_out() {
        let null = serde_json::to_value(Chunk {
            usage: ChunkUsage::for_request(true),
        })
        .unwrap_or_default();
        assert_eq!(null, json!({"usage": Value::Null}));
        let absent = serde_json::to_string(&Chunk {
            usage: ChunkUsage::for_request(false),
        })
        .unwrap_or_default();
        assert_eq!(absent, "{}");
    }

    #[test]
    fn a_final_item_without_counts_sends_no_usage_chunk() {
        assert!(usage_event("c", "o", 1, "m", None).is_none());
        assert!(usage_event("c", "o", 1, "m", Some(StreamUsage::new(1, 1))).is_some());
    }
}
