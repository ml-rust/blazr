//! One request's sampling over logits rows a remote pipeline returns.
//!
//! A swarm leader gets one host row of `vocab` logits per step from the
//! last pipeline stage. [`RowSampler`] uploads the row as `[1, 1, vocab]`
//! and samples it through [`Executor::sample_begin`] and
//! [`Executor::sample_finish`], so the grammar, logit bias, penalties, DRY,
//! mirostat, the seed and the logprobs apply as on one process.

use std::sync::Arc;

use anyhow::{bail, Result};

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, GrammarDfaOps, NormalizationOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;

use super::super::executor::Executor;
use super::super::types::{FinishReason, TokenLogprob};
use super::sampler_state::SamplerState;

/// One token sampled from a logits row.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SampledRow {
    pub token_id: u32,
    /// Set when the request asked for logprobs.
    pub logprob: Option<f32>,
    pub top_logprobs: Option<Vec<TokenLogprob>>,
    /// `Eos` or `Length`, as [`Executor::decode_finish_reason`] decides.
    pub finish: Option<FinishReason>,
}

/// The sampling state of one request whose model runs elsewhere.
pub(crate) struct RowSampler<R: Runtime<DType = DType>> {
    executor: Arc<Executor<R>>,
    state: SamplerState<R>,
    gen_config: GenerationConfig,
    /// Tokens the request samples at most.
    max_tokens: usize,
    /// Tokens sampled so far.
    generated: usize,
}

impl<R: Runtime<DType = DType>> RowSampler<R>
where
    R::Client: TensorOps<R>
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
        + boostr::quant::traits::DequantOps<R>,
{
    /// A sampler for `prompt` under `gen_config` that samples at most
    /// `max_tokens` tokens on `executor`'s device.
    ///
    /// # Errors
    ///
    /// A `max_tokens` of 0, a constraint that does not compile or upload,
    /// or an invalid DRY field.
    pub(crate) fn new(
        executor: Arc<Executor<R>>,
        prompt: &[u32],
        gen_config: GenerationConfig,
        max_tokens: usize,
    ) -> Result<Self> {
        if max_tokens == 0 {
            bail!("a request must sample at least one token");
        }
        let state = executor.sampler_state(prompt, &gen_config)?;
        Ok(Self {
            executor,
            state,
            gen_config,
            max_tokens,
            generated: 0,
        })
    }

    /// Sample the next token from `row`, the logits of the last position.
    ///
    /// # Errors
    ///
    /// A row of another length than the vocabulary, a call after the token
    /// that finished the request, a failed sample or logprob, or a token
    /// the grammar refuses.
    pub(crate) fn sample(&mut self, row: &[f32]) -> Result<SampledRow> {
        if self.generated >= self.max_tokens {
            bail!(
                "the request already sampled its {} tokens; a finished request takes no row",
                self.max_tokens
            );
        }
        let vocab = self.executor.vocab_size();
        if row.len() != vocab {
            bail!(
                "a logits row of {} values reached a sampler for a vocabulary of {vocab}",
                row.len()
            );
        }
        let logits = Tensor::<R>::from_slice(row, &[1, 1, vocab], self.executor.device())?;
        let exec = &self.executor;
        let pending = exec.sample_begin(&logits, &mut self.state, &self.gen_config, None)?;
        let token_id = exec.sample_finish(&pending, &mut self.state)?;
        let (logprob, top_logprobs) = exec.token_logprobs(&logits, token_id, &self.gen_config)?;
        let finish = exec.decode_finish_reason(token_id, self.generated, self.max_tokens);
        self.generated += 1;
        Ok(SampledRow {
            token_id,
            logprob,
            top_logprobs,
            finish,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::CpuRuntime;

    use crate::engine::test_support::tiny_executor;

    fn greedy() -> GenerationConfig {
        GenerationConfig {
            repeat_penalty: 1.0,
            ..GenerationConfig::greedy()
        }
    }

    fn peaked(vocab: usize, peak: usize) -> Vec<f32> {
        let mut row = vec![0.0f32; vocab];
        row[peak] = 20.0;
        row
    }

    #[test]
    fn rows_sample_with_logprobs_and_end_at_the_cap() -> Result<()> {
        let exec = Arc::new(tiny_executor()?);
        let vocab = exec.vocab_size();
        let cfg = GenerationConfig {
            logprobs: true,
            top_logprobs: 2,
            ..greedy()
        };
        let mut sampler = RowSampler::<CpuRuntime>::new(Arc::clone(&exec), &[1], cfg, 2)?;
        let first = sampler.sample(&peaked(vocab, 7))?;
        assert_eq!((first.token_id, first.finish), (7, None));
        assert!(first.logprob.is_some_and(|lp| lp <= 0.0 && lp > -1e-3));
        assert_eq!(first.top_logprobs.map(|t| t.len()), Some(2));
        let second = sampler.sample(&peaked(vocab, 9))?;
        assert_eq!(
            (second.token_id, second.finish),
            (9, Some(FinishReason::Length))
        );
        assert!(sampler.sample(&peaked(vocab, 9)).is_err());
        Ok(())
    }

    #[test]
    fn a_row_of_the_wrong_length_and_a_zero_cap_are_errors() -> Result<()> {
        let exec = Arc::new(tiny_executor()?);
        let vocab = exec.vocab_size();
        assert!(RowSampler::<CpuRuntime>::new(Arc::clone(&exec), &[1], greedy(), 0).is_err());
        let mut sampler = RowSampler::<CpuRuntime>::new(exec, &[1], greedy(), 4)?;
        let err = sampler.sample(&peaked(vocab - 1, 0)).unwrap_err();
        assert!(err.to_string().contains("vocabulary"), "{err}");
        Ok(())
    }
}
