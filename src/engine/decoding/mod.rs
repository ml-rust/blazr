//! Token selection: sampling, mirostat, grammar constraints, speculative decoding.

pub mod grammar;
pub(crate) mod grammar_json;
pub(crate) mod grammar_parser;
pub(crate) mod mirostat;
pub(crate) mod sampling;
pub mod speculative;
mod stop_matcher;

pub use stop_matcher::{StopMatcher, StopStep};
