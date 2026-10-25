//! Token sampling: the fused device sampler, mirostat dispatch, logprobs,
//! and the glue to boostr's host-side logit shaping.

mod host_params;
mod logits_to_token;
mod prepare;
mod token;

pub use host_params::dry_params;
