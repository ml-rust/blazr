//! One request's sampling state and the per-token sampling step.
//!
//! Every single-sequence decode loop and every decode lane samples through
//! [`Executor::sample_begin`] and [`Executor::sample_finish`]. A step picks
//! the sampling path, samples on the device, reads back the token id, pushes
//! it into the repetition history and advances the grammar.
//!
//! The step comes in two halves so a decode loop can enqueue the next forward
//! between them. The forward then runs while the token id copies to the host.
//! [`Executor::sample_step`] runs both halves back to back.

use anyhow::{anyhow, Result};

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, GrammarDfaOps, NormalizationOps, Runtime,
    SamplingOps, SamplingScratch, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::engine::grammar::GrammarDfa;
use crate::engine::sampling::MirostatState;
use crate::tokenizer::{token_piece_bytes, TokenTextStream};

use super::executor::Executor;
use super::types::FinishReason;

/// Sampling state one request carries through its decode loop.
pub(in crate::engine) struct SamplerState<R: Runtime<DType = DType>> {
    /// Prompt ids, extended with each sampled id for repetition penalty.
    pub token_history: Vec<u32>,
    pub mirostat: Option<MirostatState>,
    pub grammar_dfa: Option<GrammarDfa>,
    pub device_grammar: Option<boostr::DeviceGrammarDfa<R>>,
    /// The request's text decoder, primed with the prompt.
    pub text: TokenTextStream,
}

/// Where a sampled token id comes from.
enum TokenSource {
    /// Mirostat picks the id on the host, so it is already known.
    Mirostat(u32),
    /// The id is on the device; the event marks when the sample finishes.
    Device(u64),
}

/// A sampled token whose id the host has not read yet.
pub(in crate::engine) struct PendingToken<R: Runtime> {
    /// The `[1]` I64 token on the device, ready to feed the next forward.
    pub tensor: Tensor<R>,
    source: TokenSource,
}

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
    /// Mirostat v2 state for one request, when `gen_config` enables it.
    pub(in crate::engine) fn mirostat_state(
        gen_config: &GenerationConfig,
    ) -> Option<MirostatState> {
        if gen_config.mirostat_mode >= 2 {
            Some(MirostatState::new(
                gen_config.mirostat_tau,
                gen_config.mirostat_eta,
                gen_config.seed,
            ))
        } else {
            None
        }
    }

    /// Repetition history, mirostat state and grammar DFAs for one request.
    ///
    /// A grammar that does not compile logs a warning and decodes
    /// unconstrained.
    pub(in crate::engine) fn sampler_state(
        &self,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<SamplerState<R>> {
        // The CPU DFA tracks the grammar state. The device DFA masks logits.
        let mut grammar_dfa: Option<GrammarDfa> = None;
        let mut device_grammar: Option<boostr::DeviceGrammarDfa<R>> = None;
        if let Some(ref grammar) = gen_config.grammar {
            match crate::engine::grammar::compile_grammar_to_dfa(grammar) {
                Ok(dfa) => {
                    tracing::info!("Grammar DFA compiled: {} states", dfa.num_states());
                    let vocab_bytes: Vec<Vec<u8>> = (0..self.vocab_size())
                        .map(|i| token_piece_bytes(self.tokenizer(), i as u32))
                        .collect();
                    device_grammar = Some(dfa.to_device::<R>(&vocab_bytes, self.device())?);
                    grammar_dfa = Some(dfa);
                }
                Err(e) => {
                    tracing::warn!("Failed to compile grammar DFA: {}", e);
                }
            }
        }
        Ok(SamplerState {
            token_history: prompt_tokens.to_vec(),
            mirostat: Self::mirostat_state(gen_config),
            grammar_dfa,
            device_grammar,
            text: TokenTextStream::new(self.tokenizer(), prompt_tokens),
        })
    }

    /// Sample the next token from `logits` and leave its id on the device.
    ///
    /// Mirostat, the device grammar mask and the device filters apply as
    /// `state` and `gen_config` select. With `scratch`, the device sample
    /// writes into it instead of allocating. Mirostat ignores `scratch`.
    ///
    /// The event is recorded before this returns. Enqueue the next forward
    /// after this call, then call [`Self::sample_finish`].
    pub(in crate::engine) fn sample_begin(
        &self,
        logits: &Tensor<R>,
        state: &mut SamplerState<R>,
        gen_config: &GenerationConfig,
        scratch: Option<&SamplingScratch<R>>,
    ) -> Result<PendingToken<R>> {
        let (tensor, mirostat_id) = self.sample_token_dispatch_with_grammar(
            logits,
            &state.token_history,
            gen_config,
            &mut state.mirostat,
            state.device_grammar.as_ref(),
            scratch,
        )?;
        let source = match mirostat_id {
            Some(id) => TokenSource::Mirostat(id),
            None => TokenSource::Device(
                tensor
                    .record_event()
                    .map_err(|e| anyhow!("Record event failed: {}", e))?,
            ),
        };
        Ok(PendingToken { tensor, source })
    }

    /// Read `pending`'s id, push it into the history and advance the grammar.
    ///
    /// The device read is the one scalar copy a step makes.
    pub(in crate::engine) fn sample_finish(
        &self,
        pending: &PendingToken<R>,
        state: &mut SamplerState<R>,
    ) -> Result<u32> {
        let token_id = match pending.source {
            TokenSource::Mirostat(id) => id,
            TokenSource::Device(event) => Self::read_token_id(&pending.tensor, event)?,
        };
        state.token_history.push(token_id);
        self.advance_grammar_state(&mut state.grammar_dfa, &mut state.device_grammar, token_id);
        Ok(token_id)
    }

    /// Sample one token from `logits` and record it in `state`.
    ///
    /// Runs [`Self::sample_begin`] then [`Self::sample_finish`] with no work
    /// between them. The decode lanes call it, and they build with `cuda` only.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(in crate::engine) fn sample_step(
        &self,
        logits: &Tensor<R>,
        state: &mut SamplerState<R>,
        gen_config: &GenerationConfig,
        scratch: Option<&SamplingScratch<R>>,
    ) -> Result<u32> {
        let pending = self.sample_begin(logits, state, gen_config, scratch)?;
        self.sample_finish(&pending, state)
    }

    /// Whether `next_token` ends generation, and why.
    ///
    /// Every decode loop and every lane uses it, so the eos and length
    /// policy cannot drift between them.
    pub(in crate::engine) fn decode_finish_reason(
        &self,
        next_token: u32,
        step: usize,
        max_tokens: usize,
    ) -> Option<FinishReason> {
        if self.tokenizer().is_eos(next_token) {
            Some(FinishReason::Eos)
        } else if step + 1 == max_tokens {
            Some(FinishReason::Length)
        } else {
            None
        }
    }

    /// Advance the CPU grammar DFA by `next_token`'s decoded bytes, then sync
    /// the device-side DFA state used for on-device logit masking.
    fn advance_grammar_state(
        &self,
        grammar_dfa: &mut Option<GrammarDfa>,
        device_grammar: &mut Option<boostr::DeviceGrammarDfa<R>>,
        next_token: u32,
    ) {
        if let Some(ref mut dfa) = grammar_dfa {
            for byte in token_piece_bytes(self.tokenizer(), next_token) {
                dfa.advance(byte);
            }
            if let Some(ref mut dg) = device_grammar {
                dg.current_state = dfa.current_state() as u32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::CpuRuntime;

    use crate::engine::test_support::tiny_executor;

    /// The logit the peak token gets. Every other token gets 0.
    const PEAK: f32 = 20.0;

    /// `[1, 1, vocab]` logits that peak at `peak`.
    fn peaked_logits(exec: &Executor<CpuRuntime>, peak: usize) -> Result<Tensor<CpuRuntime>> {
        let vocab = exec.vocab_size();
        let mut row = vec![0.0f32; vocab];
        row[peak] = PEAK;
        Ok(Tensor::from_slice(&row, &[1, 1, vocab], exec.device())?)
    }

    /// Greedy decoding with no repetition penalty.
    fn greedy() -> GenerationConfig {
        GenerationConfig {
            repeat_penalty: 1.0,
            ..GenerationConfig::greedy()
        }
    }

    #[test]
    fn greedy_step_returns_the_peak_and_extends_the_history() -> Result<()> {
        let exec = tiny_executor()?;
        let cfg = greedy();
        let mut state = exec.sampler_state(&[1, 2], &cfg)?;
        let id = exec.sample_step(&peaked_logits(&exec, 7)?, &mut state, &cfg, None)?;
        assert_eq!(id, 7);
        assert_eq!(state.token_history, vec![1, 2, 7]);
        Ok(())
    }

    #[test]
    fn scratch_and_split_halves_match_the_plain_step() -> Result<()> {
        let exec = tiny_executor()?;
        let cfg = greedy();
        let logits = peaked_logits(&exec, 11)?;
        let scratch = SamplingScratch::<CpuRuntime>::new(exec.vocab_size(), exec.device())?;
        let mut plain = exec.sampler_state(&[3], &cfg)?;
        let mut into = exec.sampler_state(&[3], &cfg)?;
        let mut split = exec.sampler_state(&[3], &cfg)?;

        let a = exec.sample_step(&logits, &mut plain, &cfg, None)?;
        let b = exec.sample_step(&logits, &mut into, &cfg, Some(&scratch))?;
        let pending = exec.sample_begin(&logits, &mut split, &cfg, None)?;
        assert!(matches!(pending.source, TokenSource::Device(_)));
        let c = exec.sample_finish(&pending, &mut split)?;

        assert_eq!((a, b, c), (11, 11, 11));
        assert_eq!(plain.token_history, into.token_history);
        assert_eq!(plain.token_history, split.token_history);
        Ok(())
    }

    #[test]
    fn mirostat_step_takes_the_host_id_and_extends_the_history() -> Result<()> {
        let exec = tiny_executor()?;
        let cfg = GenerationConfig {
            mirostat_mode: 2,
            seed: Some(7),
            repeat_penalty: 1.0,
            ..GenerationConfig::default()
        };
        let mut state = exec.sampler_state(&[4], &cfg)?;
        assert!(state.mirostat.is_some());

        let pending = exec.sample_begin(&peaked_logits(&exec, 9)?, &mut state, &cfg, None)?;
        assert!(matches!(pending.source, TokenSource::Mirostat(9)));
        let id = exec.sample_finish(&pending, &mut state)?;
        assert_eq!(id, 9);
        assert_eq!(state.token_history, vec![4, 9]);
        Ok(())
    }

    #[test]
    fn grammar_step_masks_to_the_grammar_and_advances_both_dfas() -> Result<()> {
        let exec = tiny_executor()?;
        let five = (0..exec.vocab_size() as u32)
            .find(|&id| token_piece_bytes(exec.tokenizer(), id) == b"5")
            .ok_or_else(|| anyhow!("no token in the tiny vocab decodes to \"5\""))?;
        let peak = if five == 0 { 1 } else { 0 };
        let cfg = GenerationConfig {
            grammar: Some(r#"root ::= "5""#.to_string()),
            ..greedy()
        };
        let mut state = exec.sampler_state(&[1], &cfg)?;
        let start = state.grammar_dfa.as_ref().map(GrammarDfa::current_state);
        assert!(start.is_some(), "the grammar compiles");

        let id = exec.sample_step(&peaked_logits(&exec, peak)?, &mut state, &cfg, None)?;
        assert_eq!(id, five);

        let dfa = state
            .grammar_dfa
            .as_ref()
            .ok_or_else(|| anyhow!("the grammar DFA is gone"))?;
        assert_ne!(Some(dfa.current_state()), start);
        assert!(dfa.is_accepting());
        let device_state = state.device_grammar.as_ref().map(|d| d.current_state);
        assert_eq!(device_state, Some(dfa.current_state() as u32));
        Ok(())
    }

    #[test]
    fn finish_reason_is_eos_then_length_then_none() -> Result<()> {
        let exec = tiny_executor()?;
        let eos = exec
            .tokenizer()
            .eos_token_id()
            .ok_or_else(|| anyhow!("llama3 states no end-of-sequence token"))?;
        assert_eq!(
            exec.decode_finish_reason(eos, 0, 4),
            Some(FinishReason::Eos)
        );
        assert_eq!(
            exec.decode_finish_reason(5, 3, 4),
            Some(FinishReason::Length)
        );
        assert_eq!(exec.decode_finish_reason(5, 1, 4), None);
        Ok(())
    }
}
