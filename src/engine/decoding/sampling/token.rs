//! Token selection dispatch (fused kernel or mirostat v2) and the
//! `GeneratedToken` builder with logprob computation.

use anyhow::Result;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, DeviceGrammarDfa, GrammarDfaOps, NormalizationOps,
    Runtime, SamplingOps, SamplingScratch, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::tokenizer::{token_piece, TokenTextStream};

use crate::engine::executor::Executor;
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
    /// Without mirostat, the fused logits_to_token kernel samples with the
    /// optional grammar mask. The caller reads that id later through a
    /// pipelined copy.
    ///
    /// With `scratch`, the fused kernel writes into it, so a caller that
    /// steps the same lane every step reuses one buffer. Mirostat ignores
    /// `scratch`: its state update needs the full distribution on the host,
    /// and it returns a freshly allocated tensor.
    pub(crate) fn sample_token_dispatch_with_grammar(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
        mirostat: &mut Option<MirostatState>,
        gpu_grammar: Option<&DeviceGrammarDfa<R>>,
        scratch: Option<&SamplingScratch<R>>,
    ) -> Result<(Tensor<R>, Option<u32>)> {
        if let Some(ref mut ms) = mirostat {
            let (tensor, id) = self.mirostat_sample_on_device(logits, ms, gen_config)?;
            return Ok((tensor, Some(id)));
        }
        let tensor = match scratch {
            Some(scratch) => self.logits_to_token_on_device_with_grammar_into(
                logits,
                recent_tokens,
                gen_config,
                gpu_grammar,
                scratch,
            )?,
            None => self.logits_to_token_on_device_with_grammar(
                logits,
                recent_tokens,
                gen_config,
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
        let shape = logits.shape();
        let seq_len = shape[1];
        let vocab_size = shape[2];
        let all_logits: Vec<f32> = logits.to_vec();
        let offset = (seq_len - 1) * vocab_size;
        let last_logits = &all_logits[offset..offset + vocab_size];

        let (token_id, _logprob) = mirostat.sample(last_logits, gen_config.temperature);

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

        let (logprob, top_logprobs) = if gen_config.logprobs {
            let (lp, top) = self.compute_logprobs(logits, token_id, gen_config.top_logprobs)?;
            (Some(lp), Some(top))
        } else {
            (None, None)
        };

        Ok(GeneratedToken {
            token_id,
            text,
            logprob,
            top_logprobs,
            finish_reason,
        })
    }

    /// Compute log probabilities for the last position in logits.
    ///
    /// Returns (logprob of chosen token, top-N logprobs) by computing log-softmax
    /// on the raw logits at the last sequence position.
    fn compute_logprobs(
        &self,
        logits: &Tensor<R>,
        chosen_token: u32,
        top_n: usize,
    ) -> Result<(f32, Vec<TokenLogprob>)> {
        let shape = logits.shape();
        let seq_len = shape[1];
        let vocab_size = shape[2];
        let all_logits: Vec<f32> = logits.to_vec();
        let offset = (seq_len - 1) * vocab_size;
        let last_logits = &all_logits[offset..offset + vocab_size];

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

        for (i, &logit) in last_logits.iter().enumerate() {
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
