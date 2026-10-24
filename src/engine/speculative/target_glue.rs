//! The parts of a speculative target that do not depend on the KV cache: the
//! client bound, the logit bias, the greedy read and the verify call.

use anyhow::anyhow;

use boostr::error::{Error, Result};
use boostr::inference::sampling::LogitBias;
use boostr::inference::speculative::{verify_logits, Draw, SampleRule, SpeculativeClient, Verdict};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, GrammarDfaOps, NormalizationOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::engine::executor::Executor;

/// The device client a speculative target needs.
pub trait TargetClient<R: Runtime<DType = DType>>:
    TensorOps<R>
    + ScalarOps<R>
    + ConvOps<R>
    + NormalizationOps<R>
    + UnaryOps<R>
    + ActivationOps<R>
    + BinaryOps<R>
    + TypeConversionOps<R>
    + SamplingOps<R>
    + GrammarDfaOps<R>
    + ModelClient<R>
    + boostr::quant::traits::DequantOps<R>
    + SpeculativeClient<R>
{
}

impl<R, C> TargetClient<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::quant::traits::DequantOps<R>
        + SpeculativeClient<R>,
{
}

/// Read access to the logits of a target's last extend.
///
/// The decode loop reads logprobs through it, whatever cache the target uses.
pub(in crate::engine) trait LastLogits<R: Runtime<DType = DType>> {
    /// The `[1, M, vocab]` logits of the last extend before the bias, if any.
    ///
    /// Row `i` scores the prefix that ends with fed token `i`. The plain
    /// decode loops report logprobs from unbiased logits too.
    fn last_logits(&self) -> Option<&Tensor<R>>;
}

/// An anyhow error as a boostr inference error.
pub(super) fn inference_error(e: anyhow::Error) -> Error {
    Error::InferenceError {
        reason: format!("{e:#}"),
    }
}

/// The `[1, 1, vocab]` logit bias of `gen`, or none when it sets no bias.
///
/// # Errors
///
/// An error when the bias tensor cannot be built.
pub(super) fn bias_tensor<R>(
    exec: &Executor<R>,
    gen: &GenerationConfig,
) -> anyhow::Result<Option<Tensor<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R>,
{
    if gen.logit_bias.is_empty() {
        return Ok(None);
    }
    let bias = LogitBias::new(&gen.logit_bias).to_tensor::<R>(
        exec.vocab_size(),
        DType::F32,
        exec.device(),
    )?;
    Ok(Some(bias))
}

/// Check that `logits` has one row per fed token, then add the bias.
///
/// Returns the logits and the biased logits. The bias casts to the logits
/// dtype on first use.
///
/// # Errors
///
/// An error when the logits shape has no row per token, or the cast or the
/// add fails.
pub(super) fn biased_rows<R>(
    exec: &Executor<R>,
    bias: &mut Option<Tensor<R>>,
    logits: Tensor<R>,
    rows: usize,
) -> anyhow::Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R>,
{
    let shape = logits.shape();
    if shape.len() != 3 || shape[1] != rows {
        return Err(anyhow!(
            "the forward over {rows} tokens returned logits of shape {shape:?}; \
             speculative verification needs one row per token"
        ));
    }
    let Some(bias) = bias.as_mut() else {
        return Ok((logits.clone(), logits));
    };
    if bias.dtype() != logits.dtype() {
        *bias = R::default_client(exec.device())
            .cast(bias, logits.dtype())
            .map_err(|e| anyhow!("casting logit_bias to {:?} failed: {e}", logits.dtype()))?;
    }
    let biased = logits
        .add(bias)
        .map_err(|e| anyhow!("logit_bias add failed: {e}"))?;
    Ok((logits, biased))
}

/// The greedy id of every row of `logits`, read to the host.
///
/// # Errors
///
/// An error when the argmax or the host read fails, or an id is outside the
/// `u32` range.
pub(super) fn greedy_ids<R>(exec: &Executor<R>, logits: &Tensor<R>) -> anyhow::Result<Vec<u32>>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R>,
{
    // [1, n, vocab] -> [1, n] ids on the device; n ids cross to the host.
    let ids = exec
        .argmax_ids(logits, 2)?
        .contiguous()?
        .try_to_vec::<i64>()?;
    ids.into_iter()
        .map(|id| {
            u32::try_from(id)
                .map_err(|_| anyhow!("argmax returned id {id}, outside the u32 token range"))
        })
        .collect()
}

/// The error of a verify forward that got no tokens.
pub(super) fn no_verify_tokens() -> Error {
    Error::InvalidArgument {
        arg: "tokens",
        reason: "a verify forward needs the last committed token; got no tokens".into(),
    }
}

/// Verify the drafts of `tokens` against the biased `rows` of their forward.
///
/// `fed` is the target's token history, with `tokens` already appended. Row 0
/// scores the cached tokens plus the last committed token. `draft_probs` is
/// the `[1, drafts, V]` draft distributions of a draft model, or `None` for
/// one-hot drafts.
///
/// # Errors
///
/// An error when the verify fails.
pub(super) fn verify_rows<R>(
    exec: &Executor<R>,
    biased: &Tensor<R>,
    fed: &[u32],
    tokens: &[u32],
    rule: &SampleRule,
    draw: Draw,
    draft_probs: Option<&Tensor<R>>,
) -> Result<Verdict>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R>,
{
    let drafts = tokens.get(1..).unwrap_or_default();
    let context = &fed[..fed.len().saturating_sub(drafts.len())];
    let client = R::default_client(exec.device());
    verify_logits(&client, biased, context, drafts, rule, draw, draft_probs)
}
