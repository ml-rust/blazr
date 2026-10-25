//! Delivery of one lane token to its caller, shared by both lane engines.

use boostr::inference::scheduler::SequenceId;
use splintr::AnyTokenizer;

use crate::engine::request_registry::RequestRegistry;
use crate::engine::seq_sampler::SampledToken;
use crate::engine::types::GeneratedToken;

/// Send one sampled token to its caller and record it in the request's
/// history.
///
/// The request's stop sequences cut the text as on every other path.
/// Returns `false` when the request is over: the token finished it, it
/// completed a stop sequence, or the caller's receiver is gone.
pub(super) async fn deliver_token(
    registry: &RequestRegistry,
    tokenizer: &AnyTokenizer,
    seq_id: SequenceId,
    sampled: SampledToken,
) -> bool {
    let cut = registry.token_text(seq_id, tokenizer, sampled.id, sampled.finish);
    registry.push_history(seq_id, sampled.id);
    let token = GeneratedToken {
        token_id: sampled.id,
        text: cut.text,
        logprob: sampled.logprob,
        top_logprobs: sampled.top_logprobs,
        finish_reason: cut.finish,
    };
    registry.send_token(seq_id, token).await && cut.finish.is_none()
}
