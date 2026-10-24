//! The Ollama reply bodies of `/api/chat` and `/api/generate`, and their timings.

use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use super::ollama_types::{created_at, OllamaTimings, OllamaToolCall};

/// The endpoint a reply answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    /// `/api/chat`: the output goes in `message`.
    Chat,
    /// `/api/generate`: the output goes in `response`.
    Generate,
}

/// The `context` ids of a generate reply from its generated text.
pub(super) type ContextFn = Box<dyn Fn(&str) -> Vec<u32> + Send + Sync>;

/// How a reply renders its chunks and its done object.
pub(super) struct Render {
    pub kind: Kind,
    /// Set for a generate reply that returns `context`.
    pub context: Option<ContextFn>,
}

/// One released piece of output.
#[derive(Debug, Default, PartialEq)]
pub(super) struct Piece {
    pub content: String,
    pub thinking: String,
    pub calls: Vec<OllamaToolCall>,
}

impl Piece {
    /// Whether the piece carries nothing.
    pub(super) fn is_empty(&self) -> bool {
        self.content.is_empty() && self.thinking.is_empty() && self.calls.is_empty()
    }
}

impl Render {
    /// A render of `kind` with no `context`.
    pub(super) fn new(kind: Kind) -> Self {
        Self {
            kind,
            context: None,
        }
    }

    /// The `model`, `created_at`, output and `done` members of a body.
    fn body(&self, model: &str, piece: Piece, done: bool) -> Map<String, Value> {
        let mut out = Map::new();
        out.insert("model".into(), model.into());
        out.insert("created_at".into(), created_at().into());
        match self.kind {
            Kind::Chat => {
                let mut message = Map::new();
                message.insert("role".into(), "assistant".into());
                message.insert("content".into(), piece.content.into());
                if !piece.thinking.is_empty() {
                    message.insert("thinking".into(), piece.thinking.into());
                }
                if !piece.calls.is_empty() {
                    message.insert("tool_calls".into(), json!(piece.calls));
                }
                out.insert("message".into(), Value::Object(message));
            }
            Kind::Generate => {
                out.insert("response".into(), piece.content.into());
                if !piece.thinking.is_empty() {
                    out.insert("thinking".into(), piece.thinking.into());
                }
            }
        }
        out.insert("done".into(), done.into());
        out
    }

    /// A stream chunk of `piece`, with `done` false.
    pub(super) fn chunk(&self, model: &str, piece: Piece) -> Value {
        Value::Object(self.body(model, piece, false))
    }

    /// The done object: `piece`, `done_reason`, the timings and, for a
    /// generate reply with `context` set, the context ids of `generated`.
    pub(super) fn done(
        &self,
        model: &str,
        piece: Piece,
        done_reason: &str,
        timings: OllamaTimings,
        generated: &str,
    ) -> Value {
        let mut out = self.body(model, piece, true);
        out.insert("done_reason".into(), done_reason.into());
        if let Some(context) = &self.context {
            out.insert("context".into(), json!(context(generated)));
        }
        if let Value::Object(timings) = json!(timings) {
            out.extend(timings);
        }
        Value::Object(out)
    }

    /// The reply of a load or unload request: empty output, `done_reason`
    /// `load` or `unload`, no timings.
    pub(super) fn lifecycle(&self, model: &str, done_reason: &str) -> Value {
        let mut out = self.body(model, Piece::default(), true);
        out.insert("done_reason".into(), done_reason.into());
        Value::Object(out)
    }
}

/// `duration` in whole nanoseconds.
pub(super) fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// The instants a reply's timings come from.
#[derive(Debug, Clone, Copy)]
pub(super) struct Clock {
    /// The request arrived.
    pub start: Instant,
    /// Time spent selecting, and if needed loading, the model.
    pub load: Duration,
    /// Generation started.
    pub gen_start: Instant,
}

impl Clock {
    /// The timings of a stream whose first token arrived at `first` and
    /// whose final token arrived at `end`.
    pub(super) fn streamed(
        &self,
        first: Option<Instant>,
        end: Instant,
        prompt_eval_count: usize,
        eval_count: usize,
    ) -> OllamaTimings {
        let first = first.unwrap_or(end);
        OllamaTimings {
            total_duration: nanos(end.saturating_duration_since(self.start)),
            load_duration: nanos(self.load),
            prompt_eval_count,
            prompt_eval_duration: nanos(first.saturating_duration_since(self.gen_start)),
            eval_count,
            eval_duration: nanos(end.saturating_duration_since(first)),
        }
    }

    /// The timings of a whole generation that ended at `end`, whose prompt
    /// took `prompt_eval_ms`.
    pub(super) fn completed(
        &self,
        end: Instant,
        prompt_eval_count: usize,
        eval_count: usize,
        prompt_eval_ms: u64,
    ) -> OllamaTimings {
        let generation = end.saturating_duration_since(self.gen_start);
        let prompt = Duration::from_millis(prompt_eval_ms).min(generation);
        OllamaTimings {
            total_duration: nanos(end.saturating_duration_since(self.start)),
            load_duration: nanos(self.load),
            prompt_eval_count,
            prompt_eval_duration: nanos(prompt),
            eval_count,
            eval_duration: nanos(generation - prompt),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_and_generate_put_the_output_in_their_own_member() {
        let piece = || Piece {
            content: "Hi".into(),
            thinking: "plan".into(),
            calls: Vec::new(),
        };
        let chat = Render::new(Kind::Chat).chunk("m:latest", piece());
        assert_eq!(chat["model"], "m:latest");
        assert_eq!(
            chat["message"],
            json!({"role": "assistant", "content": "Hi", "thinking": "plan"})
        );
        assert_eq!(chat["done"], false);
        let generate = Render::new(Kind::Generate).chunk("m", piece());
        assert_eq!(generate["response"], "Hi");
        assert_eq!(generate["thinking"], "plan");
        assert!(generate.get("message").is_none());
    }

    #[test]
    fn done_carries_the_reason_timings_and_context() {
        let render = Render {
            kind: Kind::Generate,
            context: Some(Box::new(|text: &str| vec![1, text.len() as u32])),
        };
        let timings = OllamaTimings {
            prompt_eval_count: 3,
            eval_count: 2,
            ..OllamaTimings::default()
        };
        let done = render.done("m", Piece::default(), "length", timings, "abc");
        assert_eq!(done["done"], true);
        assert_eq!(done["done_reason"], "length");
        assert_eq!(done["context"], json!([1, 3]));
        assert_eq!(done["prompt_eval_count"], 3);
        assert_eq!(done["eval_count"], 2);
        assert_eq!(done["response"], "");
        let load = Render::new(Kind::Chat).lifecycle("m", "load");
        assert_eq!(load["message"]["content"], "");
        assert!(load.get("total_duration").is_none());
    }

    #[test]
    fn completed_timings_split_the_generation_at_the_prompt() {
        let start = Instant::now();
        let clock = Clock {
            start,
            load: Duration::from_millis(2),
            gen_start: start,
        };
        let end = start + Duration::from_millis(10);
        let t = clock.completed(end, 4, 5, 3);
        assert_eq!(t.load_duration, 2_000_000);
        assert_eq!(t.prompt_eval_duration, 3_000_000);
        assert_eq!(t.eval_duration, 7_000_000);
        assert_eq!(t.total_duration, 10_000_000);
        assert_eq!((t.prompt_eval_count, t.eval_count), (4, 5));
    }
}
