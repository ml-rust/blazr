//! Token selection: sampling, mirostat, grammar constraints, speculative decoding.

pub mod grammar_cache;
pub(crate) mod mirostat;
pub(crate) mod sampling;
pub mod speculative;
mod stop_matcher;
mod token_budget;

pub use grammar_cache::{compile_constraint, GrammarCache};
pub use stop_matcher::{StopMatcher, StopStep};
pub use token_budget::token_budget;
