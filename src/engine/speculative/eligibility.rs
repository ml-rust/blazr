//! Which requests can decode speculatively, and with what plan.

use boostr::inference::speculative::{SpeculativeConfig, MAX_NUM_DRAFT_TOKENS};
use boostr::model::{LoadedModel, ModelClient};
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, GrammarDfaOps, NormalizationOps, Runtime,
    SamplingOps, ScalarOps, TypeConversionOps, UnaryOps,
};

use super::sample_rule;
use crate::config::{DrafterSpec, GenerationConfig};
use crate::engine::executor::Executor;

/// The plan of one speculative generation.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::engine) struct SpeculativePlan {
    /// The draft count, token budget, EOS ids and acceptance rule.
    pub config: SpeculativeConfig,
    /// Prompt lookup, or the executor's draft model.
    pub drafter: DrafterSpec,
}

/// Why a request decodes through the plain sampler instead.
///
/// A fallback is never an error: the request runs unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::engine) enum Fallback {
    /// `inference.speculative.enabled` is off, or the section is unset.
    Disabled,
    /// The token budget is 0.
    ZeroBudget,
    /// The prompt has no tokens.
    EmptyPrompt,
    /// The model keeps SSM or Gated DeltaNet state, which has no rollback.
    Architecture,
    /// The model runs tensor-parallel. Follower caches have no rollback.
    TensorParallel,
    /// The request sets a constraint: JSON mode, a schema, GBNF or constrained tools.
    Grammar,
    /// Mirostat v2 samples on the host.
    Mirostat,
    /// DRY penalties run on the host.
    Dry,
    /// Typical-p filtering runs on the host.
    Typical,
    /// Dynamic temperature changes a sampled row's temperature.
    Dynatemp,
    /// The configured draft count is outside `1..=MAX_NUM_DRAFT_TOKENS`.
    DraftCount,
    /// boostr refuses the configured drafter keys.
    Drafter,
    /// `drafter: model` is set, but the executor holds no draft model.
    DraftModel,
    /// The prompt, the budget and one step of drafts pass the draft model's context.
    DraftContext,
    /// boostr's check refuses the mapped sampling rule.
    SampleRule,
    /// A prompt id is at or past the model vocabulary.
    PromptVocab,
}

impl Fallback {
    /// A fixed label for metrics and logs.
    pub(in crate::engine) fn reason(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::ZeroBudget => "zero_budget",
            Self::EmptyPrompt => "empty_prompt",
            Self::Architecture => "architecture",
            Self::TensorParallel => "tensor_parallel",
            Self::Grammar => "grammar",
            Self::Mirostat => "mirostat",
            Self::Dry => "dry",
            Self::Typical => "typical",
            Self::Dynatemp => "dynatemp",
            Self::DraftCount => "draft_count",
            Self::Drafter => "drafter",
            Self::DraftModel => "draft_model",
            Self::DraftContext => "draft_context",
            Self::SampleRule => "sample_rule",
            Self::PromptVocab => "prompt_vocab",
        }
    }
}

/// Why no request on `model` can decode speculatively.
///
/// It checks what a load fixes: the architecture and a tensor-parallel
/// communicator.
pub(in crate::engine) fn load_fallback<R>(model: &LoadedModel<R>) -> Result<(), Fallback>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    if model.needs_ssm_state() || model.needs_gdn_state() || !model.needs_kv_cache() {
        return Err(Fallback::Architecture);
    }
    if model.tp_communicator().is_some() {
        return Err(Fallback::TensorParallel);
    }
    Ok(())
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
    /// The speculative plan of one request, or why it falls back.
    ///
    /// `max_tokens` is the request's token budget after the context clamp.
    /// The checks run in the order of [`Fallback`]'s variants. The first one
    /// that refuses names the fallback.
    pub(in crate::engine) fn speculative_plan(
        &self,
        gen: &GenerationConfig,
        prompt: &[u32],
        max_tokens: usize,
    ) -> Result<SpeculativePlan, Fallback> {
        let spec = self
            .config
            .inference
            .speculative
            .as_ref()
            .filter(|spec| spec.enabled)
            .ok_or(Fallback::Disabled)?;
        if max_tokens == 0 {
            return Err(Fallback::ZeroBudget);
        }
        if prompt.is_empty() {
            return Err(Fallback::EmptyPrompt);
        }
        self.speculative_backend()?;
        Self::host_sampling(gen)?;
        let num_draft_tokens = spec.draft_count();
        if !(1..=MAX_NUM_DRAFT_TOKENS).contains(&num_draft_tokens) {
            return Err(Fallback::DraftCount);
        }
        let drafter = spec.drafter_spec().map_err(|_| Fallback::Drafter)?;
        if drafter == DrafterSpec::Model {
            let draft = self.draft.as_deref().ok_or(Fallback::DraftModel)?;
            if prompt.len() + max_tokens + num_draft_tokens > draft.config.max_seq_len() {
                return Err(Fallback::DraftContext);
            }
        }
        let vocab = self.vocab_size();
        let config = SpeculativeConfig {
            num_draft_tokens,
            max_tokens,
            eos_token_ids: self.eos_ids(),
            sampling: sample_rule(gen),
        };
        // The draft count, budget and EOS ids passed above, so only the rule can fail here.
        config.check(vocab).map_err(|_| Fallback::SampleRule)?;
        if prompt.iter().any(|&id| id as usize >= vocab) {
            return Err(Fallback::PromptVocab);
        }
        Ok(SpeculativePlan { config, drafter })
    }

    /// The model and the device group fit a speculative target.
    fn speculative_backend(&self) -> Result<(), Fallback> {
        load_fallback(&self.model)?;
        #[cfg(feature = "cuda")]
        if self.tp_state.is_some() {
            return Err(Fallback::TensorParallel);
        }
        Ok(())
    }

    /// `gen` sets no sampler feature that runs outside the boostr rule.
    fn host_sampling(gen: &GenerationConfig) -> Result<(), Fallback> {
        if gen.constraint.is_some() {
            return Err(Fallback::Grammar);
        }
        if Self::mirostat_state(gen).is_some() {
            return Err(Fallback::Mirostat);
        }
        if gen.dry_multiplier > 0.0 {
            return Err(Fallback::Dry);
        }
        if gen.typical_p > 0.0 {
            return Err(Fallback::Typical);
        }
        // The plain sampler reads the dynamic temperature only off the greedy path.
        if gen.dynatemp_range > 0.0 && !gen.is_greedy() {
            return Err(Fallback::Dynatemp);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use anyhow::Result;
    use boostr::inference::speculative::PromptLookup;
    use boostr::CpuRuntime;

    use crate::config::{Constraint, DrafterKind, PromptLookupConfig, SpeculativeDecodingConfig};
    use crate::engine::speculative::{Fallback, SpeculativePlan};
    use crate::engine::test_support::{
        full_vocab_executor, seeded_executor, seeded_executor_pair, PagedPool,
    };

    const SEED: u64 = 0x5bec;
    const PROMPT: [u32; 3] = [1, 2, 3];

    fn enabled(mut exec: Executor<CpuRuntime>, k: usize) -> Executor<CpuRuntime> {
        exec.config.inference.speculative = Some(SpeculativeDecodingConfig {
            enabled: true,
            num_draft_tokens: Some(k),
            ..SpeculativeDecodingConfig::default()
        });
        exec
    }

    fn greedy() -> GenerationConfig {
        GenerationConfig {
            repeat_penalty: 1.0,
            ..GenerationConfig::greedy()
        }
    }

    #[test]
    fn an_eligible_request_gets_the_full_plan() -> Result<()> {
        let exec = enabled(seeded_executor(SEED, None)?, 4);
        let plan = exec.speculative_plan(&greedy(), &PROMPT, 8);
        let want = SpeculativePlan {
            config: SpeculativeConfig {
                num_draft_tokens: 4,
                max_tokens: 8,
                eos_token_ids: exec.eos_ids(),
                sampling: None,
            },
            drafter: DrafterSpec::PromptLookup(PromptLookup::default()),
        };
        assert_eq!(plan.as_ref().map(|p| p.drafter), Ok(want.drafter));
        assert_eq!(plan, Ok(want));
        let sampled = GenerationConfig {
            temperature: 0.7,
            seed: Some(9),
            ..GenerationConfig::default()
        };
        let plan = exec.speculative_plan(&sampled, &PROMPT, 8);
        assert_eq!(plan.map(|p| p.config.sampling), Ok(sample_rule(&sampled)));
        Ok(())
    }

    #[test]
    fn the_plan_ends_on_the_tokenizer_eos() -> Result<()> {
        let exec = enabled(full_vocab_executor(None)?, 5);
        let eos = exec
            .tokenizer()
            .eos_token_id()
            .ok_or_else(|| anyhow::anyhow!("llama3 states no end-of-sequence token"))?;
        let plan = exec.speculative_plan(&greedy(), &PROMPT, 8);
        assert_eq!(plan.map(|p| p.config.eos_token_ids), Ok(vec![eos]));
        Ok(())
    }

    #[test]
    fn an_unset_or_disabled_section_falls_back() -> Result<()> {
        let mut exec = seeded_executor(SEED, None)?;
        assert_eq!(
            exec.speculative_plan(&greedy(), &PROMPT, 8),
            Err(Fallback::Disabled)
        );
        exec.config.inference.speculative = Some(SpeculativeDecodingConfig::default());
        assert_eq!(
            exec.speculative_plan(&greedy(), &PROMPT, 8),
            Err(Fallback::Disabled)
        );
        Ok(())
    }

    #[test]
    fn each_sampler_feature_names_its_fallback() -> Result<()> {
        let exec = enabled(seeded_executor(SEED, None)?, 5);
        let cases = [
            (
                GenerationConfig {
                    constraint: Some(Constraint::JsonObject),
                    ..greedy()
                },
                Fallback::Grammar,
            ),
            (
                GenerationConfig {
                    mirostat_mode: 2,
                    ..greedy()
                },
                Fallback::Mirostat,
            ),
            (
                GenerationConfig {
                    dry_multiplier: 0.8,
                    ..greedy()
                },
                Fallback::Dry,
            ),
            (
                GenerationConfig {
                    typical_p: 0.95,
                    ..greedy()
                },
                Fallback::Typical,
            ),
            (
                GenerationConfig {
                    temperature: 0.7,
                    dynatemp_range: 0.5,
                    ..GenerationConfig::default()
                },
                Fallback::Dynatemp,
            ),
            (
                GenerationConfig {
                    temperature: 0.7,
                    top_p: 0.0,
                    ..GenerationConfig::default()
                },
                Fallback::SampleRule,
            ),
        ];
        for (gen, want) in cases {
            assert_eq!(
                exec.speculative_plan(&gen, &PROMPT, 8),
                Err(want),
                "{gen:?}"
            );
        }
        // Features the plain sampler ignores keep a request eligible.
        for gen in [
            GenerationConfig {
                mirostat_mode: 1,
                ..greedy()
            },
            GenerationConfig {
                dynatemp_range: 0.5,
                ..greedy()
            },
        ] {
            assert!(exec.speculative_plan(&gen, &PROMPT, 8).is_ok(), "{gen:?}");
        }
        Ok(())
    }

    #[test]
    fn budget_prompt_and_draft_count_fall_back() -> Result<()> {
        let exec = enabled(seeded_executor(SEED, None)?, 5);
        let gen = greedy();
        assert_eq!(
            exec.speculative_plan(&gen, &PROMPT, 0),
            Err(Fallback::ZeroBudget)
        );
        assert_eq!(
            exec.speculative_plan(&gen, &[], 8),
            Err(Fallback::EmptyPrompt)
        );
        let outside = exec.vocab_size() as u32;
        assert_eq!(
            exec.speculative_plan(&gen, &[1, outside], 8),
            Err(Fallback::PromptVocab)
        );
        for k in [0, MAX_NUM_DRAFT_TOKENS + 1] {
            let exec = enabled(seeded_executor(SEED, None)?, k);
            assert_eq!(
                exec.speculative_plan(&gen, &PROMPT, 8),
                Err(Fallback::DraftCount)
            );
        }
        Ok(())
    }

    #[test]
    fn the_plan_takes_the_configured_drafter() -> Result<()> {
        let mut exec = enabled(seeded_executor(SEED, None)?, 3);
        let lookup = PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 4,
            window: 32,
        };
        if let Some(spec) = exec.config.inference.speculative.as_mut() {
            spec.prompt_lookup = lookup;
        }
        let plan = exec.speculative_plan(&greedy(), &PROMPT, 8);
        assert_eq!(
            plan.map(|p| p.drafter),
            Ok(DrafterSpec::PromptLookup(PromptLookup::new(2, 4, 32)?))
        );
        if let Some(spec) = exec.config.inference.speculative.as_mut() {
            spec.prompt_lookup.window = 4;
        }
        assert_eq!(
            exec.speculative_plan(&greedy(), &PROMPT, 8),
            Err(Fallback::Drafter)
        );
        Ok(())
    }

    #[test]
    fn the_model_drafter_needs_a_draft_with_room() -> Result<()> {
        let drafting = |mut exec: Executor<CpuRuntime>| {
            if let Some(spec) = exec.config.inference.speculative.as_mut() {
                spec.drafter = DrafterKind::Model;
                spec.draft_model = Some("draft".into());
            }
            exec
        };
        let exec = drafting(enabled(seeded_executor(SEED, None)?, 4));
        assert_eq!(
            exec.speculative_plan(&greedy(), &PROMPT, 8),
            Err(Fallback::DraftModel)
        );
        // The draft context is 16: 3 prompt + 8 budget + 4 drafts fit, 3 + 10 + 4 do not.
        let exec = drafting(enabled(seeded_executor_pair(SEED, SEED, 16, None)?, 4));
        let plan = exec.speculative_plan(&greedy(), &PROMPT, 8);
        assert_eq!(plan.map(|p| p.drafter), Ok(DrafterSpec::Model));
        assert_eq!(
            exec.speculative_plan(&greedy(), &PROMPT, 10),
            Err(Fallback::DraftContext)
        );
        Ok(())
    }

    #[test]
    fn paged_attention_gets_a_plan() -> Result<()> {
        let pool = PagedPool {
            block_size: 4,
            blocks: 16,
        };
        let exec = enabled(seeded_executor(SEED, Some(pool))?, 5);
        assert!(exec.speculative_plan(&greedy(), &PROMPT, 8).is_ok());
        Ok(())
    }

    #[test]
    fn every_reason_is_a_distinct_label() {
        let all = [
            Fallback::Disabled,
            Fallback::ZeroBudget,
            Fallback::EmptyPrompt,
            Fallback::Architecture,
            Fallback::TensorParallel,
            Fallback::Grammar,
            Fallback::Mirostat,
            Fallback::Dry,
            Fallback::Typical,
            Fallback::Dynatemp,
            Fallback::DraftCount,
            Fallback::Drafter,
            Fallback::DraftModel,
            Fallback::DraftContext,
            Fallback::SampleRule,
            Fallback::PromptVocab,
        ];
        let labels: HashSet<&str> = all.iter().map(|f| f.reason()).collect();
        assert_eq!(labels.len(), all.len());
    }
}
