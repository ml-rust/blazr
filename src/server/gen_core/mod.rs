//! Generation plumbing shared by the endpoints: sampling fields, streaming, reasoning.

pub(super) mod gen_types;
pub(super) mod generation;
pub(super) mod reasoning;
pub(super) mod streaming;

pub use reasoning::{
    prompt_opens_reasoning, split_reasoning, split_reasoning_for_format, ReasoningParser,
    ReasoningTags,
};
pub use streaming::{create_chat_stream, create_completion_stream, StreamToken};
