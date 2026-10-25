//! One request's sampling state: repetition history, mirostat, DRY and grammar.

use anyhow::{anyhow, Result};

use boostr::inference::sampling::{DryParams, MirostatV2};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, GrammarDfaOps, NormalizationOps, Runtime,
    SamplingOps, ScalarOps, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::tokenizer::TokenTextStream;

use super::executor::Executor;
use super::sampler_grammar::GrammarState;
use crate::engine::decoding::sampling::dry_params;

/// Sampling state one request carries through its decode loop.
pub(in crate::engine) struct SamplerState<R: Runtime<DType = DType>> {
    /// Prompt ids, extended with each sampled id for repetition penalty.
    pub token_history: Vec<u32>,
    pub mirostat: Option<MirostatV2>,
    /// The request's DRY penalty with its breakers resolved, or `None`
    /// when DRY is off.
    pub dry: Option<DryParams>,
    /// The request's constraint: its DFA, device mask and current state.
    pub grammar: Option<GrammarState<R>>,
    /// The request's text decoder, primed with the prompt.
    pub text: TokenTextStream,
    /// Index of the next token this request samples, from 0. A seeded
    /// request uses it as the Philox counter, so each step draws anew.
    pub step: u64,
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
    pub(in crate::engine) fn mirostat_state(gen_config: &GenerationConfig) -> Option<MirostatV2> {
        if gen_config.mirostat_mode >= 2 {
            Some(MirostatV2::new(
                gen_config.mirostat_tau,
                gen_config.mirostat_eta,
                gen_config.seed,
            ))
        } else {
            None
        }
    }

    /// Repetition history, mirostat, DRY and grammar state for one request.
    ///
    /// # Errors
    ///
    /// An error when the request's constraint does not compile or does not
    /// upload, or when a DRY breaker or field is invalid. A constrained
    /// request never decodes unconstrained.
    pub(in crate::engine) fn sampler_state(
        &self,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<SamplerState<R>> {
        Ok(SamplerState {
            token_history: prompt_tokens.to_vec(),
            mirostat: Self::mirostat_state(gen_config),
            dry: dry_params(self.tokenizer(), gen_config).map_err(|e| anyhow!(e))?,
            grammar: self.grammar_state(gen_config)?,
            text: TokenTextStream::new(self.tokenizer(), prompt_tokens),
            step: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use boostr::inference::grammar::json_schema::SchemaOptions;
    use boostr::inference::grammar::CompileLimits;

    use crate::config::{CompiledConstraint, Constraint};
    use crate::engine::compile_constraint;
    use crate::engine::test_support::tiny_executor;

    fn constrained(constraint: Constraint) -> GenerationConfig {
        GenerationConfig {
            constraint: Some(constraint),
            repeat_penalty: 1.0,
            ..GenerationConfig::greedy()
        }
    }

    fn gbnf(text: &str) -> GenerationConfig {
        constrained(Constraint::Gbnf(text.to_string()))
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
}
