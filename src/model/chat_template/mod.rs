//! Chat templates: per-model message formatting.
//!
//! A model's own Jinja template is rendered as is. Built-in formats cover
//! models that ship none, and a caller can name one explicitly.

pub mod builtin;
pub mod encode;
pub mod jinja;
pub mod jinja_json;
pub mod message;
pub mod source;
pub mod template;

pub use encode::{encode_chat_prompt, ChatPromptError};
pub use jinja::{JinjaTemplate, RenderOptions, TemplateError, TemplateOrigin};
pub use message::ChatMessage;
pub use source::ModelTemplate;
pub use template::{ChatTemplate, BUILTIN_NAMES};
