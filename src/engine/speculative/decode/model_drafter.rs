//! The drafter over an executor's draft model.

use anyhow::{anyhow, Result};

use boostr::inference::speculative::{ModelDrafter, SpeculativeConfig};
use boostr::{DType, RoPEPackedOps, Runtime};

use crate::config::GenerationConfig;
use crate::engine::executor::Executor;
use crate::engine::speculative::target_glue::TargetClient;
use crate::engine::speculative::ContiguousTarget;
use crate::engine::types::PrefillCancel;

impl<R: Runtime<DType = DType>> Executor<R>
where
    R::Client: TargetClient<R> + RoPEPackedOps<R>,
{
    /// A drafter over the draft model, with its own contiguous KV cache.
    ///
    /// The cache holds the prompt, the budget and one step of drafts. It
    /// drops with the request.
    ///
    /// # Errors
    ///
    /// An error when the executor holds no draft model, or the draft cache
    /// cannot be built.
    pub(super) fn model_drafter<'a>(
        &'a self,
        gen: &GenerationConfig,
        prompt_len: usize,
        config: &SpeculativeConfig,
        cancel: PrefillCancel<'a>,
    ) -> Result<ModelDrafter<R, ContiguousTarget<'a, R>, R::Client>> {
        let draft = self.draft.as_deref().ok_or_else(|| {
            anyhow!(
                "drafter model has no loaded draft model; set inference.speculative.draft_model \
                 and reload the model"
            )
        })?;
        let budget = config.max_tokens + config.num_draft_tokens;
        let model = ContiguousTarget::new(draft, gen, prompt_len, budget)?.with_cancel(cancel);
        Ok(ModelDrafter::new(model, R::default_client(draft.device())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use boostr::inference::BlockAllocator;

    use super::super::test_rig::{
        assert_logprobs_close, count, margin, shape, tie_free, with_logprobs, Exec, CTX,
        MAX_TOKENS, NEAR_TIE, RANDOM, REPETITIVE, SEED,
    };
    use crate::config::{DrafterKind, SpeculativeDecodingConfig};
    use crate::engine::requests::batch_test_rig::alone_tokens;
    use crate::engine::test_support::{
        seeded_executor_ctx, seeded_executor_pair, seeded_paged, PagedPool,
    };
    use crate::engine::types::GeneratedToken;
    use crate::telemetry::names;
    use crate::telemetry::test_recorder::TestRecorder;

    /// `exec` with `drafter: model` on and `k` drafts. Its draft is attached.
    fn model_speculating(mut exec: Exec, k: usize) -> Exec {
        exec.config.inference.speculative = Some(SpeculativeDecodingConfig {
            enabled: true,
            drafter: DrafterKind::Model,
            draft_model: Some("seeded-draft".to_string()),
            num_draft_tokens: Some(k),
            ..SpeculativeDecodingConfig::default()
        });
        exec
    }

    /// The drafted and accepted counts `run` adds.
    async fn counted<F, T>(recorder: &TestRecorder, run: F) -> Result<(T, f64, f64)>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let drafted = count(recorder, names::SPECULATIVE_DRAFTED_TOTAL);
        let accepted = count(recorder, names::SPECULATIVE_ACCEPTED_TOTAL);
        let out = run.await?;
        Ok((
            out,
            count(recorder, names::SPECULATIVE_DRAFTED_TOTAL) - drafted,
            count(recorder, names::SPECULATIVE_ACCEPTED_TOTAL) - accepted,
        ))
    }

    /// Greedy output of `spec` equals `plain` on every tie-free prompt.
    ///
    /// With `full`, every draft is accepted.
    async fn assert_greedy_as_plain(plain: &Exec, spec: &Exec, full: bool) -> Result<()> {
        let recorder = TestRecorder::install();
        let gen = with_logprobs(MAX_TOKENS);
        let mut compared = 0;
        for &prompt in RANDOM.iter().chain(REPETITIVE.iter()) {
            let want = alone_tokens(plain, prompt, &gen).await?;
            if margin(&want) < NEAR_TIE {
                continue;
            }
            let (got, drafted, accepted) =
                counted(&recorder, alone_tokens(spec, prompt, &gen)).await?;
            let what = format!("prompt {prompt:?}");
            assert_eq!(shape(&got), shape(&want), "{what}");
            assert_logprobs_close(&got, &want, &what);
            assert!(drafted > 0.0, "{what}: the draft model drafted nothing");
            if full {
                assert_eq!(
                    accepted, drafted,
                    "{what}: a same-weight draft lost a draft"
                );
            }
            compared += 1;
        }
        assert!(compared >= 3, "only {compared} prompts have no near-tie");
        assert!(!recorder
            .render()
            .contains(names::SPECULATIVE_FALLBACKS_TOTAL));
        Ok(())
    }

    #[tokio::test]
    async fn a_same_weight_draft_matches_plain_greedy_and_accepts_every_draft() -> Result<()> {
        let plain = seeded_executor_ctx(SEED, CTX)?;
        for k in [1, 4] {
            let spec = model_speculating(seeded_executor_pair(SEED, SEED, CTX, None)?, k);
            assert_greedy_as_plain(&plain, &spec, true).await?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_different_draft_keeps_the_plain_greedy_output() -> Result<()> {
        let plain = seeded_executor_ctx(SEED, CTX)?;
        let spec = model_speculating(seeded_executor_pair(SEED, SEED ^ 0xd2af, CTX, None)?, 5);
        assert_greedy_as_plain(&plain, &spec, false).await
    }

    #[tokio::test]
    async fn a_same_weight_draft_samples_seeded_output_and_accepts_every_draft() -> Result<()> {
        let recorder = TestRecorder::install();
        let spec = model_speculating(seeded_executor_pair(SEED, SEED, CTX, None)?, 4);
        let sampled = GenerationConfig {
            temperature: 0.9,
            seed: Some(7),
            max_tokens: 24,
            ..GenerationConfig::default()
        };
        let ids = |tokens: Vec<GeneratedToken>| -> Vec<u32> {
            tokens.into_iter().map(|t| t.token_id).collect()
        };
        let prompt = REPETITIVE[0];
        let (first, drafted, accepted) =
            counted(&recorder, alone_tokens(&spec, prompt, &sampled)).await?;
        let first = ids(first);
        assert_eq!(first.len(), 24);
        assert!(first.iter().all(|&id| (id as usize) < spec.vocab_size()));
        assert!(drafted > 0.0, "the draft model drafted nothing");
        assert_eq!(accepted, drafted, "a same-weight draft lost a draft");
        assert_eq!(
            ids(alone_tokens(&spec, prompt, &sampled).await?),
            first,
            "a seed repeats its output"
        );
        assert!(!recorder
            .render()
            .contains(names::SPECULATIVE_FALLBACKS_TOTAL));
        Ok(())
    }

    #[tokio::test]
    async fn a_paged_target_with_a_contiguous_draft_matches_plain_paged_greedy() -> Result<()> {
        let recorder = TestRecorder::install();
        let pool = PagedPool {
            block_size: 4,
            blocks: 16,
        };
        let plain = seeded_paged(SEED, pool, CTX, false)?;
        let spec = model_speculating(seeded_executor_pair(SEED, SEED, CTX, Some(pool))?, 5);
        assert!(spec
            .draft
            .as_ref()
            .is_some_and(|d| !d.config.inference.paged_attention));
        let prompt = tie_free(&plain, &REPETITIVE, 12).await?;
        let gen = with_logprobs(12);
        let want = alone_tokens(&plain, prompt, &gen).await?;
        let (got, drafted, accepted) =
            counted(&recorder, alone_tokens(&spec, prompt, &gen)).await?;
        assert_eq!(shape(&got), shape(&want));
        assert_logprobs_close(&got, &want, "paged");
        assert!(drafted > 0.0, "the draft model drafted nothing");
        assert_eq!(accepted, drafted, "a same-weight draft lost a draft");
        let free = spec
            .shared_allocator()
            .ok_or_else(|| anyhow!("the executor has no shared allocator"))?
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?
            .free_block_count();
        assert_eq!(free, pool.blocks, "the request returned its blocks");
        Ok(())
    }
}
