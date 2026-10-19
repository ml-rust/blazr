//! Token selection dispatch (fused kernel or mirostat v2) and the
//! `GeneratedToken` builder with logprob computation.

use anyhow::Result;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, DeviceGrammarDfa, GrammarDfaOps, NormalizationOps,
    Runtime, SampleSeed, SamplingOps, SamplingScratch, ScalarOps, Tensor, TypeConversionOps,
    UnaryOps,
};

use crate::config::GenerationConfig;
use crate::tokenizer::{token_piece, TokenTextStream};

use crate::engine::executor::Executor;
use crate::engine::host_read::last_logits_row;
use crate::engine::types::{FinishReason, GeneratedToken, TokenLogprob};

use super::MirostatState;

impl<R: Runtime<DType = DType>> Executor<R>
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
    /// Sample a token, dispatching to either standard or mirostat sampling.
    ///
    /// Returns the `[1]` I64 token tensor and, for mirostat, the token id.
    /// The optional grammar mask applies on both paths. Without mirostat,
    /// the fused logits_to_token kernel samples. The caller reads that id
    /// later through a pipelined copy.
    ///
    /// With `scratch`, the fused kernel writes into it, so a caller that
    /// steps the same lane every step reuses one buffer. Mirostat ignores
    /// `scratch`: its state update needs the full distribution on the host,
    /// and it returns a freshly allocated tensor.
    ///
    /// `seed` names this step's Philox draw for the fused kernel. Mirostat
    /// ignores it: its host generator takes the request seed once.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_token_dispatch_with_grammar(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
        seed: Option<SampleSeed>,
        mirostat: &mut Option<MirostatState>,
        gpu_grammar: Option<&DeviceGrammarDfa<R>>,
        scratch: Option<&SamplingScratch<R>>,
    ) -> Result<(Tensor<R>, Option<u32>)> {
        if let Some(ref mut ms) = mirostat {
            // Mirostat samples on the host, so the grammar mask applies first.
            let masked = match gpu_grammar {
                Some(grammar) => R::default_client(self.device())
                    .grammar_dfa_mask_logits(logits, grammar)
                    .map_err(|e| anyhow::anyhow!("grammar_dfa_mask_logits failed: {e}"))?,
                None => logits.clone(),
            };
            let (tensor, id) = self.mirostat_sample_on_device(&masked, ms, gen_config)?;
            return Ok((tensor, Some(id)));
        }
        let tensor = match scratch {
            Some(scratch) => self.logits_to_token_on_device_with_grammar_into(
                logits,
                recent_tokens,
                gen_config,
                seed,
                gpu_grammar,
                scratch,
            )?,
            None => self.logits_to_token_on_device_with_grammar(
                logits,
                recent_tokens,
                gen_config,
                seed,
                gpu_grammar,
            )?,
        };
        Ok((tensor, None))
    }

    /// Sample a token using Mirostat v2 from logits on device.
    ///
    /// Pulls logits to CPU, runs mirostat sampling, returns token as GPU tensor.
    fn mirostat_sample_on_device(
        &self,
        logits: &Tensor<R>,
        mirostat: &mut MirostatState,
        gen_config: &GenerationConfig,
    ) -> Result<(Tensor<R>, u32)> {
        let last_logits = last_logits_row(logits)?;
        let (token_id, _logprob) = mirostat.sample(&last_logits, gen_config.temperature);

        let token_tensor = Tensor::from_slice(&[token_id as i64], &[1], logits.device())?;
        Ok((token_tensor, token_id))
    }

    /// Build a GeneratedToken with optional logprobs computation.
    ///
    /// `text` is the request's own decoder, so a SentencePiece token keeps
    /// its leading space and a split character arrives whole.
    pub(crate) fn make_token(
        &self,
        token_id: u32,
        text: &mut TokenTextStream,
        logits: &Tensor<R>,
        gen_config: &GenerationConfig,
        finish_reason: Option<FinishReason>,
    ) -> Result<GeneratedToken> {
        let text = text.next_text(
            self.tokenizer(),
            token_id,
            finish_reason == Some(FinishReason::Eos),
            finish_reason.is_some(),
        );

        let (logprob, top_logprobs) = self.token_logprobs(logits, token_id, gen_config)?;

        Ok(GeneratedToken {
            token_id,
            text,
            logprob,
            top_logprobs,
            finish_reason,
        })
    }

    /// The logprob of `token_id` and the top logprobs, when `gen_config`
    /// asks for them. Both are `None` otherwise, and nothing is computed.
    ///
    /// Every decode loop and the batching engine call this with the logits
    /// the token was sampled from.
    pub(crate) fn token_logprobs(
        &self,
        logits: &Tensor<R>,
        token_id: u32,
        gen_config: &GenerationConfig,
    ) -> Result<(Option<f32>, Option<Vec<TokenLogprob>>)> {
        if !gen_config.logprobs {
            return Ok((None, None));
        }
        let (lp, top) = self.compute_logprobs(logits, token_id, gen_config.top_logprobs)?;
        Ok((Some(lp), Some(top)))
    }

    /// Compute log probabilities for the last position in logits.
    ///
    /// Returns (logprob of chosen token, top-N logprobs) by computing log-softmax
    /// on the raw logits at the last sequence position. Only that position's
    /// row copies to the host.
    fn compute_logprobs(
        &self,
        logits: &Tensor<R>,
        chosen_token: u32,
        top_n: usize,
    ) -> Result<(f32, Vec<TokenLogprob>)> {
        let last_logits = last_logits_row(logits)?;
        let vocab_size = last_logits.len();
        let last_logits = last_logits.as_slice();

        // Log-softmax: log(softmax(x)) = x - log(sum(exp(x - max)))
        let max_logit = last_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = last_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .sum::<f32>()
            .ln()
            + max_logit;

        // Compute log probs and find top-N
        let n = top_n.min(vocab_size).min(20);
        let mut top: Vec<(usize, f32)> = Vec::with_capacity(n + 1);

        // With no top entries asked for, nothing is scanned.
        let scanned: &[f32] = if n == 0 { &[] } else { last_logits };
        for (i, &logit) in scanned.iter().enumerate() {
            let lp = logit - log_sum_exp;
            if top.len() < n {
                top.push((i, lp));
                if top.len() == n {
                    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                }
            } else if lp > top[n - 1].1 {
                top[n - 1] = (i, lp);
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let chosen_logprob = if (chosen_token as usize) < vocab_size {
            last_logits[chosen_token as usize] - log_sum_exp
        } else {
            f32::NEG_INFINITY
        };

        let top_logprobs: Vec<TokenLogprob> = top
            .into_iter()
            .map(|(id, lp)| TokenLogprob {
                token_id: id as u32,
                text: token_piece(self.tokenizer(), id as u32),
                logprob: lp,
            })
            .collect();

        Ok((chosen_logprob, top_logprobs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::CpuRuntime;

    use crate::engine::test_support::tiny_executor;

    /// `[1, 2, vocab]` logits: the last position peaks at `peak`.
    fn two_positions(exec: &Executor<CpuRuntime>, peak: usize) -> Result<Tensor<CpuRuntime>> {
        let vocab = exec.vocab_size();
        let mut rows = vec![0.0f32; 2 * vocab];
        rows[0] = 50.0;
        rows[vocab + peak] = 20.0;
        Ok(Tensor::from_slice(&rows, &[1, 2, vocab], exec.device())?)
    }

    #[test]
    fn logprobs_read_the_last_position_and_only_when_asked() -> Result<()> {
        let exec = tiny_executor()?;
        let logits = two_positions(&exec, 7)?;
        let off = GenerationConfig::default();
        assert!(matches!(
            exec.token_logprobs(&logits, 7, &off)?,
            (None, None)
        ));

        let asked = GenerationConfig {
            logprobs: true,
            top_logprobs: 2,
            ..GenerationConfig::default()
        };
        let (lp, top) = exec.token_logprobs(&logits, 7, &asked)?;
        assert!(lp.is_some_and(|lp| lp <= 0.0 && lp > -1e-3));
        let top = top.unwrap_or_default();
        assert_eq!(top.len(), 2);
        assert_eq!(top.first().map(|t| t.token_id), Some(7));
        Ok(())
    }

    #[test]
    fn zero_top_logprobs_gives_the_chosen_logprob_and_no_entries() -> Result<()> {
        let exec = tiny_executor()?;
        let asked = GenerationConfig {
            logprobs: true,
            top_logprobs: 0,
            ..GenerationConfig::default()
        };
        let (lp, top) = exec.token_logprobs(&two_positions(&exec, 3)?, 3, &asked)?;
        assert!(lp.is_some());
        assert_eq!(top.map(|t| t.len()), Some(0));
        Ok(())
    }

    /// BF16 logits, the CUDA default, give the same logprobs as F32 logits.
    ///
    /// Sabotage check: read the row with `try_to_vec::<f32>` and no cast,
    /// and this test fails with a host read type mismatch.
    #[cfg(feature = "f16")]
    #[test]
    fn bf16_logits_give_logprobs() -> Result<()> {
        let exec = tiny_executor()?;
        let vocab = exec.vocab_size();
        // BF16 bit patterns: 50.0 at position 0, 20.0 at token 7 of the last.
        let mut bits = vec![0u16; 2 * vocab];
        bits[0] = 0x4248;
        bits[vocab + 7] = 0x41a0;
        let bytes: &[u8] = bytemuck::cast_slice(&bits);
        let logits =
            Tensor::<CpuRuntime>::from_bytes(bytes, &[1, 2, vocab], DType::BF16, exec.device())?;
        let asked = GenerationConfig {
            logprobs: true,
            top_logprobs: 2,
            ..GenerationConfig::default()
        };

        let (lp, top) = exec.token_logprobs(&logits, 7, &asked)?;
        let (want_lp, want_top) = exec.token_logprobs(&two_positions(&exec, 7)?, 7, &asked)?;
        assert_eq!(lp, want_lp);
        let ids = |t: Option<Vec<TokenLogprob>>| -> Vec<u32> {
            t.unwrap_or_default().iter().map(|t| t.token_id).collect()
        };
        assert_eq!(ids(top), ids(want_top));
        Ok(())
    }
}
