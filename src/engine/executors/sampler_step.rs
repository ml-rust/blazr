//! One request's sampling state and the per-token sampling step.
//!
//! Every single-sequence decode loop, every decode lane and every
//! continuous-batching row samples through [`Executor::sample_begin`] and
//! [`Executor::sample_finish`]. A step picks the sampling path, samples on
//! the device, reads back the token id, pushes it into the repetition history
//! and advances the grammar.
//!
//! The step comes in two halves so a decode loop can enqueue the next forward
//! between them. The forward then runs while the token id copies to the host.
//! [`Executor::sample_step`] runs both halves back to back.

use anyhow::{anyhow, Result};

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, GrammarDfaOps, NormalizationOps, Runtime, SampleSeed,
    SamplingOps, SamplingScratch, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::engine::sampling::MirostatState;
use crate::tokenizer::TokenTextStream;

use super::executor::Executor;
use super::sampler_grammar::GrammarState;
use super::types::FinishReason;

/// Sampling state one request carries through its decode loop.
pub(in crate::engine) struct SamplerState<R: Runtime<DType = DType>> {
    /// Prompt ids, extended with each sampled id for repetition penalty.
    pub token_history: Vec<u32>,
    pub mirostat: Option<MirostatState>,
    /// The request's constraint: its DFA, device mask and current state.
    pub grammar: Option<GrammarState<R>>,
    /// The request's text decoder, primed with the prompt.
    pub text: TokenTextStream,
    /// Index of the next token this request samples, from 0. A seeded
    /// request uses it as the Philox counter, so each step draws anew.
    pub step: u64,
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

    /// Repetition history, mirostat state and grammar state for one request.
    ///
    /// # Errors
    ///
    /// An error when the request's constraint does not compile or does not
    /// upload. A constrained request never decodes unconstrained.
    pub(in crate::engine) fn sampler_state(
        &self,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<SamplerState<R>> {
        Ok(SamplerState {
            token_history: prompt_tokens.to_vec(),
            mirostat: Self::mirostat_state(gen_config),
            grammar: self.grammar_state(gen_config)?,
            text: TokenTextStream::new(self.tokenizer(), prompt_tokens),
            step: 0,
        })
    }

    /// Sample the next token from `logits` and leave its id on the device.
    ///
    /// Mirostat, the device grammar mask and the device filters apply as
    /// `state` and `gen_config` select. With `scratch`, the device sample
    /// writes into it instead of allocating. Mirostat ignores `scratch`.
    /// A seeded request draws the Philox uniform at `(seed, state.step)`.
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
        let seed = Self::step_seed(gen_config, state.step);
        let (tensor, mirostat_id) = self.sample_token_dispatch_with_grammar(
            logits,
            &state.token_history,
            gen_config,
            seed,
            &mut state.mirostat,
            state.grammar.as_ref().map(GrammarState::device),
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
    ///
    /// # Errors
    ///
    /// An error when the read fails, or when the token breaks the grammar.
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
        state.step = state.step.wrapping_add(1);
        Self::advance_grammar_state(&mut state.grammar, token_id)?;
        Ok(token_id)
    }

    /// The Philox draw step `step` of a request uses: the request seed as
    /// the key and `step` as the counter. `None` for an unseeded request.
    pub(in crate::engine) fn step_seed(
        gen_config: &GenerationConfig,
        step: u64,
    ) -> Option<SampleSeed> {
        gen_config.seed.map(|key| SampleSeed::new(key, step))
    }

    /// Sample one token from `logits` and record it in `state`.
    ///
    /// Runs [`Self::sample_begin`] then [`Self::sample_finish`] with no work
    /// between them. The decode lanes and the continuous batching engine call it.
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

    /// Advance the request's grammar past `next_token` on the host and the
    /// device. Without a grammar this does nothing.
    ///
    /// # Errors
    ///
    /// An error when `next_token` has no transition from the current state.
    fn advance_grammar_state(grammar: &mut Option<GrammarState<R>>, next_token: u32) -> Result<()> {
        match grammar {
            Some(grammar) => grammar.advance(next_token),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use boostr::inference::grammar::json_schema::SchemaOptions;
    use boostr::inference::grammar::CompileLimits;
    use boostr::CpuRuntime;

    use crate::config::{CompiledConstraint, Constraint};
    use crate::engine::compile_constraint;
    use crate::engine::test_support::{full_vocab_executor, piece_id, tiny_executor};

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

    /// Seeded sampling at temperature 1 with no filter and no penalty.
    fn seeded(seed: u64) -> GenerationConfig {
        GenerationConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: Some(seed),
            ..GenerationConfig::default()
        }
    }

    /// `[1, 1, vocab]` logits every token ties on.
    fn flat_logits(exec: &Executor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let vocab = exec.vocab_size();
        Ok(Tensor::from_slice(
            &vec![0.0f32; vocab],
            &[1, 1, vocab],
            exec.device(),
        )?)
    }

    #[test]
    fn each_seeded_step_takes_the_next_counter() -> Result<()> {
        let exec = tiny_executor()?;
        let cfg = seeded(7);
        let mut state = exec.sampler_state(&[1], &cfg)?;
        assert_eq!(state.step, 0);
        let logits = flat_logits(&exec)?;
        exec.sample_step(&logits, &mut state, &cfg, None)?;
        exec.sample_step(&logits, &mut state, &cfg, None)?;
        assert_eq!(state.step, 2);

        let step = |n| Executor::<CpuRuntime>::step_seed(&cfg, n);
        assert_eq!(step(0), Some(SampleSeed::new(7, 0)));
        assert_eq!(step(1), Some(SampleSeed::new(7, 1)));
        assert_eq!(Executor::<CpuRuntime>::step_seed(&greedy(), 3), None);
        Ok(())
    }

    /// Flat logits map one uniform to one token. A draw reseeded from the
    /// request seed alone gives the same uniform, so the same token, at
    /// every step. Sabotage check: pass `state.step = 0` to every step, and
    /// the uniform and token assertions fail.
    #[test]
    fn seeded_steps_reproduce_and_draw_a_new_uniform_each_step() -> Result<()> {
        const STEPS: usize = 16;
        let exec = tiny_executor()?;
        let cfg = seeded(7);
        let logits = flat_logits(&exec)?;
        let scratch = SamplingScratch::<CpuRuntime>::new(exec.vocab_size(), exec.device())?;

        let mut plain = exec.sampler_state(&[1], &cfg)?;
        let mut into = exec.sampler_state(&[1], &cfg)?;
        let mut plain_ids = Vec::with_capacity(STEPS);
        let mut into_ids = Vec::with_capacity(STEPS);
        let mut uniforms = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            plain_ids.push(exec.sample_step(&logits, &mut plain, &cfg, None)?);
            into_ids.push(exec.sample_step(&logits, &mut into, &cfg, Some(&scratch))?);
            uniforms.push(scratch.random.item::<f32>()?.to_bits());
        }

        assert_eq!(plain_ids, into_ids, "one seed must give one token stream");
        for (n, pair) in uniforms.windows(2).enumerate() {
            assert_ne!(pair[0], pair[1], "steps {n} and {} drew one uniform", n + 1);
        }
        assert!(
            plain_ids.iter().any(|&id| id != plain_ids[0]),
            "every step sampled token {}",
            plain_ids[0]
        );

        let mut again = exec.sampler_state(&[1], &cfg)?;
        let mut again_ids = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            again_ids.push(exec.sample_step(&logits, &mut again, &cfg, None)?);
        }
        assert_eq!(
            plain_ids, again_ids,
            "a second run with the seed must match"
        );
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

    fn constrained(constraint: Constraint) -> GenerationConfig {
        GenerationConfig {
            constraint: Some(constraint),
            ..greedy()
        }
    }

    fn gbnf(text: &str) -> GenerationConfig {
        constrained(Constraint::Gbnf(text.to_string()))
    }

    /// The grammar state of `state`, which must exist.
    fn grammar_of(state: &SamplerState<CpuRuntime>) -> Result<&GrammarState<CpuRuntime>> {
        state
            .grammar
            .as_ref()
            .ok_or_else(|| anyhow!("the request carries no grammar state"))
    }

    #[test]
    fn grammar_step_masks_to_the_grammar_and_advances_both_states() -> Result<()> {
        let exec = tiny_executor()?;
        let five = piece_id(&exec, b"5")?;
        let peak = if five == 0 { 1 } else { 0 };
        let cfg = gbnf(r#"root ::= "5""#);
        let mut state = exec.sampler_state(&[1], &cfg)?;
        let start = grammar_of(&state)?.state();

        let id = exec.sample_step(&peaked_logits(&exec, peak)?, &mut state, &cfg, None)?;
        assert_eq!(id, five);

        let grammar = grammar_of(&state)?;
        assert_ne!(grammar.state(), start);
        assert!(grammar.is_accepting());
        assert_eq!(grammar.device().current_state(), grammar.state());
        Ok(())
    }

    /// Logits that prefer EOS at every step: the grammar holds EOS back
    /// until it completes, then allows nothing else.
    #[test]
    fn eos_comes_exactly_when_the_grammar_completes() -> Result<()> {
        let exec = full_vocab_executor(None)?;
        let eos = exec
            .tokenizer()
            .eos_token_id()
            .ok_or_else(|| anyhow!("llama3 states no end-of-sequence token"))?;
        let five = piece_id(&exec, b"5")?;
        let cfg = gbnf(r#"root ::= "5""#);
        let mut state = exec.sampler_state(&[1], &cfg)?;
        let logits = peaked_logits(&exec, eos as usize)?;

        let first = exec.sample_step(&logits, &mut state, &cfg, None)?;
        assert_eq!(first, five, "EOS must wait for a complete match");
        assert_eq!(exec.decode_finish_reason(first, 0, 8), None);
        let second = exec.sample_step(&logits, &mut state, &cfg, None)?;
        assert_eq!(second, eos, "a complete match allows EOS alone");
        assert_eq!(
            exec.decode_finish_reason(second, 1, 8),
            Some(FinishReason::Eos)
        );
        Ok(())
    }

    #[test]
    fn eos_ids_are_the_finish_ids_inside_the_model_vocab() -> Result<()> {
        let tiny = tiny_executor()?;
        assert!(tiny.eos_ids().is_empty(), "the EOS id is past 32 ids");
        let full = full_vocab_executor(None)?;
        let eos = full
            .tokenizer()
            .eos_token_id()
            .ok_or_else(|| anyhow!("llama3 states no end-of-sequence token"))?;
        assert_eq!(full.eos_ids(), vec![eos]);
        for id in full.eos_ids() {
            assert_eq!(full.decode_finish_reason(id, 0, 8), Some(FinishReason::Eos));
        }
        Ok(())
    }

    #[test]
    fn an_eos_id_before_a_complete_match_is_an_error() -> Result<()> {
        let exec = full_vocab_executor(None)?;
        let eos = exec
            .tokenizer()
            .eos_token_id()
            .ok_or_else(|| anyhow!("llama3 states no end-of-sequence token"))?;
        let mut state = exec.sampler_state(&[1], &gbnf(r#"root ::= "5""#))?;
        let err = Executor::<CpuRuntime>::advance_grammar_state(&mut state.grammar, eos);
        assert!(err.is_err_and(|e| e.to_string().contains("does not accept")));
        let off = piece_id(&exec, b"6")?;
        let err = Executor::<CpuRuntime>::advance_grammar_state(&mut state.grammar, off);
        assert!(err.is_err_and(|e| e.to_string().contains("no transition")));
        assert_eq!(
            grammar_of(&state)?.state(),
            0,
            "a refused token leaves the state"
        );
        Ok(())
    }

    #[tokio::test]
    async fn max_tokens_cuts_a_constrained_run_with_length() -> Result<()> {
        let exec = full_vocab_executor(None)?;
        let seven = piece_id(&exec, b"7")?;
        let cfg = GenerationConfig {
            max_tokens: 2,
            logit_bias: [(seven, 10.0)].into_iter().collect(),
            ..gbnf(r#"root ::= "7" "7" "7""#)
        };
        let result = exec.generate_text(&[1], &cfg).await?;
        assert_eq!(result.text, "77");
        assert_eq!(result.finish_reason, FinishReason::Length);
        Ok(())
    }

    #[test]
    fn a_bad_constraint_fails_the_request_before_it_decodes() -> Result<()> {
        let exec = tiny_executor()?;
        let invalid = exec.sampler_state(&[1], &gbnf("root ::= ("));
        assert!(invalid.is_err_and(|e| e.to_string().contains("does not compile")));
        let schema = serde_json::json!({"type": "integer", "minimum": 0});
        let unsupported = exec.sampler_state(&[1], &constrained(Constraint::JsonSchema(schema)));
        assert!(unsupported.is_err_and(|e| e.to_string().contains("minimum")));
        Ok(())
    }

    #[test]
    fn an_admitted_dfa_is_used_and_a_stale_one_is_ignored() -> Result<()> {
        let exec = tiny_executor()?;
        let five = Constraint::Gbnf(r#"root ::= "5""#.to_string());
        let dfa = Arc::new(compile_constraint(
            &five,
            &SchemaOptions::default(),
            &CompileLimits::default(),
        )?);
        let admitted = GenerationConfig {
            compiled: Some(CompiledConstraint::new(five.clone(), Arc::clone(&dfa))),
            ..constrained(five)
        };
        assert!(exec.sampler_state(&[1], &admitted)?.grammar.is_some());

        // The compiled DFA names another constraint, so the bad text compiles here.
        let stale = GenerationConfig {
            constraint: Some(Constraint::Gbnf("root ::= (".to_string())),
            ..admitted
        };
        assert!(exec.sampler_state(&[1], &stale).is_err());
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
