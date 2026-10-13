//! Token sampling strategies
//!
//! Contains temperature-based sampling, mirostat v2, DRY penalty,
//! typical filtering, logprob computation, and logit bias application.

mod host_filters;
mod logits_to_token;
mod token;

pub use super::mirostat::MirostatState;
