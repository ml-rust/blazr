//! One request's grammar state: the compiled DFA, its device copy, and the
//! host-side state the sampler advances after each token.
//!
//! The device mask allows a token by the rules of boostr's
//! `DeviceGrammarDfa`. An EOS id is allowed exactly in an accepting state.
//! A token with no bytes is never allowed. A token outside the mask is an
//! internal error here, never a silent step.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::inference::grammar::json_schema::SchemaOptions;
use boostr::inference::grammar::{ByteDfa, CompileLimits, VocabBytes};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, DeviceGrammarDfa, GrammarDfaOps, NormalizationOps,
    Runtime, SamplingOps, ScalarOps, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::engine::compile_constraint;
use crate::tokenizer::token_piece_bytes;

use super::executor::Executor;

/// The grammar a request decodes under, and where it stands.
pub(in crate::engine) struct GrammarState<R: Runtime<DType = DType>> {
    dfa: Arc<ByteDfa>,
    device: DeviceGrammarDfa<R>,
    vocab: VocabBytes,
    eos: Vec<u32>,
    /// The host-side DFA state. The device copy holds the same value.
    state: u32,
}

impl<R: Runtime<DType = DType>> GrammarState<R> {
    /// The device DFA the sampler masks logits with.
    pub(in crate::engine) fn device(&self) -> &DeviceGrammarDfa<R> {
        &self.device
    }

    /// The current DFA state.
    #[cfg(test)]
    pub(in crate::engine) fn state(&self) -> u32 {
        self.state
    }

    /// True when the output so far is a complete match of the grammar.
    pub(in crate::engine) fn is_accepting(&self) -> bool {
        self.dfa.is_accepting(self.state)
    }

    /// Advance past `token`, on the host and on the device.
    ///
    /// An EOS id walks no bytes. A token the mask forbids is an error that
    /// names the token, its bytes and the state.
    pub(in crate::engine) fn advance(&mut self, token: u32) -> Result<()> {
        if self.eos.contains(&token) {
            if !self.is_accepting() {
                return Err(anyhow!(
                    "grammar error: EOS token {token} arrived in DFA state {}, which does not \
                     accept. The grammar mask allows EOS only in an accepting state",
                    self.state
                ));
            }
            return Ok(());
        }
        let bytes = self.vocab.token_bytes(token).ok_or_else(|| {
            anyhow!(
                "grammar error: token {token} is outside the grammar vocabulary of {} ids",
                self.vocab.len()
            )
        })?;
        let next = self
            .dfa
            .walk(self.state, bytes)
            .filter(|_| !bytes.is_empty())
            .ok_or_else(|| {
                anyhow!(
                    "grammar error: token {token} (bytes {bytes:?}) has no transition from DFA \
                     state {}. The grammar mask must forbid it",
                    self.state
                )
            })?;
        self.device
            .set_current_state(next)
            .map_err(|e| anyhow!("grammar error: the device DFA refused state {next}: {e}"))?;
        self.state = next;
        Ok(())
    }
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
        + ModelClient<R>,
{
    /// The bytes every model token id adds to the output text, in id order.
    ///
    /// The first call builds them from the tokenizer. An id the tokenizer
    /// skips on decode, or does not know, gets no bytes, so no grammar
    /// allows it. EOS ids reach the mask through the EOS list instead.
    pub(in crate::engine) fn grammar_vocab(&self) -> Result<&VocabBytes> {
        if let Some(vocab) = self.grammar_vocab.get() {
            return Ok(vocab);
        }
        let tokenizer = self.tokenizer();
        let pieces = (0..self.vocab_size()).map(|id| match u32::try_from(id) {
            Ok(id) => token_piece_bytes(tokenizer, id),
            Err(_) => Vec::new(),
        });
        let vocab = VocabBytes::new(pieces)
            .map_err(|e| anyhow!("building the grammar vocabulary failed: {e}"))?;
        // A concurrent first call can win the set. Both built the same bytes.
        let _ = self.grammar_vocab.set(vocab);
        self.grammar_vocab
            .get()
            .ok_or_else(|| anyhow!("the grammar vocabulary is missing after it was built"))
    }

    /// The EOS ids the grammar mask allows in an accepting state.
    ///
    /// They are the ids [`Self::decode_finish_reason`] ends generation on,
    /// inside the model vocabulary. The model can never emit an id past it.
    pub(in crate::engine) fn grammar_eos_ids(&self) -> Vec<u32> {
        self.tokenizer()
            .eos_token_id()
            .filter(|&id| usize::try_from(id).is_ok_and(|at| at < self.vocab_size()))
            .into_iter()
            .collect()
    }

    /// The grammar state for `gen_config`, or `None` when it sets no constraint.
    ///
    /// Admission puts the compiled DFA in `gen_config.compiled`. Without it,
    /// or when it names another constraint, the constraint compiles here.
    ///
    /// # Errors
    ///
    /// An error when the constraint does not compile or the device upload
    /// fails. The request then never starts unconstrained.
    pub(in crate::engine) fn grammar_state(
        &self,
        gen_config: &GenerationConfig,
    ) -> Result<Option<GrammarState<R>>> {
        let Some(constraint) = gen_config.constraint.as_ref() else {
            return Ok(None);
        };
        let compiled = gen_config
            .compiled
            .as_ref()
            .filter(|compiled| compiled.source() == constraint);
        let dfa = match compiled {
            Some(compiled) => Arc::clone(compiled.dfa()),
            None => {
                let limits = CompileLimits::default();
                let dfa = compile_constraint(constraint, &SchemaOptions::default(), &limits)
                    .map_err(|e| anyhow!("the request's constraint does not compile: {e}"))?;
                Arc::new(dfa)
            }
        };
        let vocab = self.grammar_vocab()?.clone();
        let eos = self.grammar_eos_ids();
        let device = DeviceGrammarDfa::<R>::from_byte_dfa(&dfa, &vocab, &eos, self.device())
            .map_err(|e| anyhow!("uploading the grammar DFA failed: {e}"))?;
        tracing::debug!(states = dfa.num_states(), "grammar DFA ready");
        Ok(Some(GrammarState {
            state: dfa.start(),
            dfa,
            device,
            vocab,
            eos,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Constraint;
    use crate::engine::test_support::{
        check_two_field_instance, full_vocab_executor, two_field_bias, two_field_schema,
    };
    use crate::engine::FinishReason;

    #[tokio::test]
    async fn a_json_schema_run_ends_by_eos_on_a_valid_instance() -> Result<()> {
        let exec = full_vocab_executor(None)?;
        for (temperature, seed) in [(0.0, None), (0.7, Some(11))] {
            let cfg = GenerationConfig {
                max_tokens: 40,
                temperature,
                seed,
                repeat_penalty: 1.0,
                logit_bias: two_field_bias(&exec)?,
                constraint: Some(Constraint::JsonSchema(two_field_schema())),
                ..GenerationConfig::default()
            };
            let result = exec.generate_text(&[1], &cfg).await?;
            check_two_field_instance(&result.text)?;
            assert_eq!(
                result.finish_reason,
                FinishReason::Eos,
                "temperature {temperature}: {:?}",
                result.text
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn json_object_mode_yields_an_object() -> Result<()> {
        let exec = full_vocab_executor(None)?;
        let cfg = GenerationConfig {
            max_tokens: 40,
            repeat_penalty: 1.0,
            logit_bias: two_field_bias(&exec)?,
            constraint: Some(Constraint::JsonObject),
            ..GenerationConfig::greedy()
        };
        let result = exec.generate_text(&[1], &cfg).await?;
        let value: serde_json::Value = serde_json::from_str(&result.text)?;
        assert!(value.is_object(), "{:?}", result.text);
        assert_eq!(result.finish_reason, FinishReason::Eos);
        Ok(())
    }

    #[test]
    fn the_vocabulary_builds_once_per_executor() -> Result<()> {
        let exec = full_vocab_executor(None)?;
        let first = exec.grammar_vocab()?.clone();
        let second = exec.grammar_vocab()?;
        assert!(first.ptr_eq(second));
        assert_eq!(first.len(), exec.vocab_size());
        Ok(())
    }
}
