//! Token sampling: the fused device sampler, mirostat dispatch, logprobs,
//! and the glue to boostr's host-side logit shaping.

mod host_params;
mod logits_to_token;
mod token;
