//! OpenAI chat completions: request types, vision inputs, and tool calls.

pub(super) mod chat;
pub(super) mod chat_core;
pub(super) mod chat_sse;
pub(super) mod chat_types;
pub(super) mod chat_vision;
pub(super) mod json_scan;
pub(super) mod route_stream;
pub(super) mod tool_calls;
pub(super) mod tool_choice;
pub(super) mod tool_constraint;
pub(super) mod tool_parser;
pub(super) mod tools;
