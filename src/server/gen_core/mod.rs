//! Generation plumbing shared by the endpoints: sampling fields, streaming, reasoning.

pub(super) mod chat_choices;
pub(super) mod chat_chunks;
pub(super) mod chat_stream;
pub(super) mod chat_token_text;
pub(super) mod completion_stream;
pub(super) mod constraint_admission;
pub(super) mod gen_types;
pub(super) mod generation;
pub(super) mod reasoning;
pub(super) mod request_fields;
pub(super) mod stop_forwarder;
pub(super) mod stop_stream;
pub(super) mod stream_metrics;
pub(super) mod stream_usage;
pub(super) mod streaming;
pub(super) mod tag_prefix;
pub(super) mod tool_call_delta;

pub use completion_stream::create_completion_stream;
pub use reasoning::{
    prompt_opens_reasoning, split_reasoning, split_reasoning_for_format, ReasoningParser,
    ReasoningTags,
};
pub use streaming::StreamToken;
