//! Generation plumbing shared by the endpoints: sampling fields, streaming, reasoning.

pub(super) mod chat_stream;
pub(super) mod constraint_admission;
pub(super) mod gen_types;
pub(super) mod generation;
pub(super) mod reasoning;
pub(super) mod request_fields;
pub(super) mod stop_stream;
pub(super) mod stream_metrics;
pub(super) mod streaming;

pub use chat_stream::create_chat_stream;
pub use reasoning::{
    prompt_opens_reasoning, split_reasoning, split_reasoning_for_format, ReasoningParser,
    ReasoningTags,
};
pub use streaming::{create_completion_stream, StreamToken};
