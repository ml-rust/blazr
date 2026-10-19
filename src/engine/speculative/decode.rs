//! The speculative token stream of one single-sequence request.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::inference::speculative::{
    DraftSource, FinishReason as SpeculativeFinish, ModelDrafter, SpeculativeConfig,
    SpeculativeGenerator, SpeculativeModel, SpeculativeStats, StepOutput,
};
use boostr::{DType, RoPEPackedOps, Runtime};

use super::target_glue::{LastLogits, TargetClient};
use super::{ContiguousTarget, Fallback, PagedTarget, SpeculativePlan};
use crate::config::{DrafterSpec, GenerationConfig};
use crate::engine::executor::Executor;
use crate::engine::types::{FinishReason, GeneratedToken};
use crate::telemetry::record::{
    record_speculative, record_speculative_fallback, SpeculativeCounts,
};
use crate::tokenizer::TokenTextStream;

/// The latest stats of one speculative generation.
///
/// It records them when it drops, so a stopped or dropped stream reports too.
#[derive(Default)]
struct StatsGuard(SpeculativeStats);

impl Drop for StatsGuard {
    fn drop(&mut self) {
        let stats = self.0;
        record_speculative(SpeculativeCounts {
            drafted: stats.drafted,
            accepted: stats.accepted,
            emitted: stats.emitted,
            target_forwards: stats.target_forwards,
        });
        tracing::debug!(
            acceptance_rate = stats.acceptance_rate(),
            tokens_per_forward = stats.tokens_per_forward(),
            steps = stats.steps,
            "speculative decoding finished"
        );
    }
}

/// The engine finish reason of a boostr one.
fn finish_reason(reason: SpeculativeFinish) -> FinishReason {
    match reason {
        SpeculativeFinish::Eos => FinishReason::Eos,
        SpeculativeFinish::Length => FinishReason::Length,
    }
}

impl<R: Runtime<DType = DType>> Executor<R>
where
    R::Client: TargetClient<R> + RoPEPackedOps<R>,
{
    /// The speculative plan of one request, or `None` for the plain path.
    ///
    /// With speculation on, a fallback logs its reason at debug level and
    /// counts in `blazr_speculative_fallbacks_total`. With it off, nothing
    /// logs or counts.
    pub(in crate::engine) fn speculative_route(
        &self,
        gen: &GenerationConfig,
        prompt: &[u32],
        max_tokens: usize,
    ) -> Option<SpeculativePlan> {
        match self.speculative_plan(gen, prompt, max_tokens) {
            Ok(plan) => Some(plan),
            Err(Fallback::Disabled) => None,
            Err(fallback) => {
                let reason = fallback.reason();
                tracing::debug!(reason, "speculative decoding skipped");
                record_speculative_fallback(reason);
                None
            }
        }
    }

    /// Decode `prompt` speculatively under `plan`.
    ///
    /// The target is a [`PagedTarget`] when the executor runs paged attention
    /// and a [`ContiguousTarget`] otherwise. A paged pool that cannot cover a
    /// step, or a model the paged kernel cannot serve, ends the stream with
    /// the error the plain paged path returns. Nothing falls back to plain
    /// decoding mid-request. A model drafter runs the draft model over its
    /// own contiguous KV cache, whatever cache the target uses.
    ///
    /// Each step runs one target forward and yields one or more tokens.
    /// Emitted token `j` of a step takes row `j` of that forward's logits
    /// for its logprobs. The last token of the final step carries the
    /// finish reason. Dropping the stream drops the request's KV cache.
    pub(in crate::engine) fn generate_speculative<'a>(
        &'a self,
        prompt: &'a [u32],
        gen: &'a GenerationConfig,
        plan: SpeculativePlan,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            let SpeculativePlan { config, drafter } = plan;
            let paged = self.config.inference.paged_attention;
            match drafter {
                DrafterSpec::PromptLookup(lookup) if paged => {
                    let target = PagedTarget::new(self, gen, prompt.len())?;
                    for await item in self.drive_speculative(target, prompt, gen, config, lookup) {
                        yield item;
                    }
                }
                DrafterSpec::PromptLookup(lookup) => {
                    let target = ContiguousTarget::new(self, gen, prompt.len(), config.max_tokens)?;
                    for await item in self.drive_speculative(target, prompt, gen, config, lookup) {
                        yield item;
                    }
                }
                DrafterSpec::Model if paged => {
                    let target = PagedTarget::new(self, gen, prompt.len())?;
                    let drafter = self.model_drafter(gen, prompt.len(), &config)?;
                    for await item in self.drive_speculative(target, prompt, gen, config, drafter) {
                        yield item;
                    }
                }
                DrafterSpec::Model => {
                    let target = ContiguousTarget::new(self, gen, prompt.len(), config.max_tokens)?;
                    let drafter = self.model_drafter(gen, prompt.len(), &config)?;
                    for await item in self.drive_speculative(target, prompt, gen, config, drafter) {
                        yield item;
                    }
                }
            }
        }
    }

    /// A drafter over the draft model, with its own contiguous KV cache.
    ///
    /// The cache holds the prompt, the budget and one step of drafts. It
    /// drops with the request.
    ///
    /// # Errors
    ///
    /// An error when the executor holds no draft model, or the draft cache
    /// cannot be built.
    fn model_drafter<'a>(
        &'a self,
        gen: &GenerationConfig,
        prompt_len: usize,
        config: &SpeculativeConfig,
    ) -> Result<ModelDrafter<R, ContiguousTarget<'a, R>, R::Client>> {
        let draft = self.draft.as_deref().ok_or_else(|| {
            anyhow!(
                "drafter model has no loaded draft model; set inference.speculative.draft_model \
                 and reload the model"
            )
        })?;
        let budget = config.max_tokens + config.num_draft_tokens;
        let model = ContiguousTarget::new(draft, gen, prompt_len, budget)?;
        Ok(ModelDrafter::new(model, R::default_client(draft.device())))
    }

    /// The speculative loop over `target`, drafting with `drafter`.
    fn drive_speculative<'a, M, D>(
        &'a self,
        target: M,
        prompt: &'a [u32],
        gen: &'a GenerationConfig,
        config: SpeculativeConfig,
        drafter: D,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a
    where
        M: SpeculativeModel + LastLogits<R> + 'a,
        D: DraftSource<M> + 'a,
    {
        stream! {
            let max_tokens = config.max_tokens;
            let num_draft_tokens = config.num_draft_tokens;
            let mut generator = SpeculativeGenerator::new(target, drafter, config)?;
            let mut stats = StatsGuard::default();
            let mut text = TokenTextStream::new(self.tokenizer(), prompt);

            tracing::info!(phase = "prefill_start", backend = "speculative", prompt_tokens = prompt.len());
            generator.start(prompt)?;
            tracing::info!(phase = "prefill_end", backend = "speculative");
            tracing::info!(phase = "decode_start", backend = "speculative", max_tokens, num_draft_tokens);
            loop {
                let StepOutput { tokens, finish } = generator.step()?;
                stats.0 = generator.stats();
                let logits = generator
                    .model()
                    .last_logits()
                    .cloned()
                    .ok_or_else(|| anyhow!("a speculative step left no logits"))?;
                let last = tokens.len().saturating_sub(1);
                for (j, &id) in tokens.iter().enumerate() {
                    let row = logits.narrow(1, j, 1)?;
                    let fin = if j == last { finish.map(finish_reason) } else { None };
                    yield Ok(self.make_token(id, &mut text, &row, gen, fin)?);
                }
                if finish.is_some() {
                    break;
                }
            }
            tracing::info!(phase = "decode_end", backend = "speculative");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use boostr::inference::BlockAllocator;
    use boostr::CpuRuntime;
    use futures::StreamExt;

    use crate::config::{Constraint, DrafterKind, SpeculativeDecodingConfig};
    use crate::engine::requests::batch_test_rig::{
        alone_tokens, drain, greedy, ids, rig, rig_on, POOL, SEED as RIG_SEED, TINY_CTX,
    };
    use crate::engine::test_support::{
        full_vocab_executor, seeded_executor, seeded_executor_ctx, seeded_executor_pair,
        seeded_paged, PagedPool,
    };
    use crate::engine::types::GenerationResult;
    use crate::telemetry::names;
    use crate::telemetry::test_recorder::TestRecorder;

    type Exec = Executor<CpuRuntime>;

    const SEED: u64 = 0x5bec;
    const CTX: usize = 64;
    const MAX_TOKENS: usize = 40;
    /// Smallest top-2 logprob gap of a compared greedy run.
    const NEAR_TIE: f32 = 1e-4;
    /// Logprobs of the two paths agree to this much.
    const LOGPROB_TOL: f32 = 1e-5;
    const RANDOM: [&[u32]; 3] = [&[1], &[5, 9, 2], &[17, 3, 28, 11]];
    /// Prompts where prompt lookup finds matches.
    const REPETITIVE: [&[u32]; 3] = [
        &[3, 7, 11, 3, 7, 11, 3, 7, 11],
        &[4, 4, 4, 4, 4, 4],
        &[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
    ];

    fn speculating(mut exec: Exec, k: usize) -> Exec {
        exec.config.inference.speculative = Some(SpeculativeDecodingConfig {
            enabled: true,
            num_draft_tokens: Some(k),
            ..SpeculativeDecodingConfig::default()
        });
        exec
    }

    /// The seeded llama with speculation off, and its twin with `k` drafts.
    fn twins(ctx: usize, k: usize) -> Result<(Exec, Exec)> {
        let plain = seeded_executor_ctx(SEED, ctx)?;
        Ok((plain, speculating(seeded_executor_ctx(SEED, ctx)?, k)))
    }

    fn shape(tokens: &[GeneratedToken]) -> Vec<(u32, String, Option<FinishReason>)> {
        tokens
            .iter()
            .map(|t| (t.token_id, t.text.clone(), t.finish_reason))
            .collect()
    }

    fn outcome(result: &GenerationResult) -> (String, FinishReason, usize) {
        (
            result.text.clone(),
            result.finish_reason,
            result.completion_tokens,
        )
    }

    fn count(recorder: &TestRecorder, name: &str) -> f64 {
        recorder.value(name, &[]).unwrap_or(0.0)
    }

    fn fallbacks(recorder: &TestRecorder, reason: &str) -> f64 {
        recorder
            .value(names::SPECULATIVE_FALLBACKS_TOTAL, &[("reason", reason)])
            .unwrap_or(0.0)
    }

    fn with_logprobs(max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            logprobs: true,
            top_logprobs: 2,
            ..greedy(max_tokens)
        }
    }

    /// The smallest top-2 logprob gap over `tokens`.
    fn margin(tokens: &[GeneratedToken]) -> f32 {
        tokens
            .iter()
            .filter_map(|t| match t.top_logprobs.as_deref() {
                Some([a, b, ..]) => Some((a.logprob - b.logprob).abs()),
                _ => None,
            })
            .fold(f32::INFINITY, f32::min)
    }

    /// The first of `prompts` whose plain greedy run has no near-tie.
    async fn tie_free(
        plain: &Exec,
        prompts: &[&'static [u32]],
        max: usize,
    ) -> Result<&'static [u32]> {
        for &prompt in prompts {
            if margin(&alone_tokens(plain, prompt, &with_logprobs(max)).await?) >= NEAR_TIE {
                return Ok(prompt);
            }
        }
        Err(anyhow!("every prompt of {prompts:?} has a near-tie"))
    }

    fn assert_logprobs_close(got: &[GeneratedToken], want: &[GeneratedToken], what: &str) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let (Some(gl), Some(wl)) = (g.logprob, w.logprob) else {
                panic!("{what}: token {i} has no logprob");
            };
            assert!(
                (gl - wl).abs() < LOGPROB_TOL,
                "{what}: token {i}: {gl} vs {wl}"
            );
            let top = |t: &GeneratedToken| -> Vec<f32> {
                t.top_logprobs.iter().flatten().map(|l| l.logprob).collect()
            };
            let (gt, wt) = (top(g), top(w));
            assert_eq!(gt.len(), wt.len(), "{what}: token {i} top count");
            for (a, b) in gt.iter().zip(&wt) {
                assert!(
                    (a - b).abs() < LOGPROB_TOL,
                    "{what}: token {i} top: {a} vs {b}"
                );
            }
        }
    }

    #[tokio::test]
    async fn greedy_speculation_matches_plain_tokens_text_and_logprobs() -> Result<()> {
        let recorder = TestRecorder::install();
        let gen = with_logprobs(MAX_TOKENS);
        for k in [1, 2, 5, 8] {
            let (plain, spec) = twins(CTX, k)?;
            let (mut compared, mut accepted) = (0, 0.0);
            let prompts = RANDOM.iter().map(|&p| (p, false));
            for (prompt, repetitive) in prompts.chain(REPETITIVE.iter().map(|&p| (p, true))) {
                let want = alone_tokens(&plain, prompt, &gen).await?;
                if margin(&want) < NEAR_TIE {
                    continue;
                }
                let before = count(&recorder, names::SPECULATIVE_ACCEPTED_TOTAL);
                let got = alone_tokens(&spec, prompt, &gen).await?;
                if repetitive {
                    accepted += count(&recorder, names::SPECULATIVE_ACCEPTED_TOTAL) - before;
                }
                let what = format!("k {k}, prompt {prompt:?}");
                assert_eq!(shape(&got), shape(&want), "{what}");
                assert_logprobs_close(&got, &want, &what);
                let cfg = greedy(MAX_TOKENS);
                let want = plain.generate_text(prompt, &cfg).await?;
                let got = spec.generate_text(prompt, &cfg).await?;
                assert_eq!(outcome(&got), outcome(&want), "{what}");
                compared += 1;
            }
            assert!(
                compared >= 3,
                "k {k}: only {compared} prompts have no near-tie"
            );
            assert!(accepted > 0.0, "k {k}: prompt lookup accepted no draft");
        }
        assert!(count(&recorder, names::SPECULATIVE_TARGET_FORWARDS_TOTAL) > 0.0);
        assert!(!recorder
            .render()
            .contains(names::SPECULATIVE_FALLBACKS_TOTAL));
        Ok(())
    }

    #[tokio::test]
    async fn a_stop_inside_a_step_cuts_the_text_as_plain() -> Result<()> {
        let (plain, spec) = twins(CTX, 5)?;
        let prompt = tie_free(&plain, &REPETITIVE, 12).await?;
        let tokens = alone_tokens(&plain, prompt, &greedy(12)).await?;
        // Every id under 32 is one ASCII character, so a byte range is a token range.
        let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
        if text.len() != tokens.len() || !text.is_ascii() {
            return Err(anyhow!("{text:?} is not one ASCII character per token"));
        }
        let cfg = GenerationConfig {
            stop_sequences: vec![text[3..5].to_string()],
            ..greedy(12)
        };
        let want = plain.generate_text(prompt, &cfg).await?;
        assert_eq!(want.finish_reason, FinishReason::Stop);
        assert_eq!(
            outcome(&spec.generate_text(prompt, &cfg).await?),
            outcome(&want)
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_budget_ends_on_the_exact_count_with_length() -> Result<()> {
        let k = 5;
        let (plain, spec) = twins(CTX, k)?;
        let prompt = tie_free(&plain, &REPETITIVE, 2 * k + 1).await?;
        for max in [1, k, k + 1, 2 * k + 1] {
            let got = alone_tokens(&spec, prompt, &greedy(max)).await?;
            assert_eq!(
                shape(&got),
                shape(&alone_tokens(&plain, prompt, &greedy(max)).await?)
            );
            assert_eq!(got.len(), max, "max_tokens {max}");
            let finishes: Vec<_> = got.iter().map(|t| t.finish_reason).collect();
            let mut want = vec![None; max - 1];
            want.push(Some(FinishReason::Length));
            assert_eq!(finishes, want, "max_tokens {max}");
        }
        // The context leaves room for 5 of the 100 asked tokens.
        let (plain, spec) = twins(16, k)?;
        let prompt = REPETITIVE[2];
        let want = plain.generate_text(prompt, &greedy(100)).await?;
        let got = spec.generate_text(prompt, &greedy(100)).await?;
        assert_eq!(
            (got.completion_tokens, got.finish_reason),
            (5, FinishReason::Length)
        );
        assert_eq!(outcome(&got), outcome(&want));
        Ok(())
    }

    #[tokio::test]
    async fn a_biased_eos_ends_after_one_token_as_plain() -> Result<()> {
        let recorder = TestRecorder::install();
        let plain = full_vocab_executor(None)?;
        let spec = speculating(full_vocab_executor(None)?, 5);
        let eos = *plain
            .eos_ids()
            .first()
            .ok_or_else(|| anyhow!("llama3 states no end-of-sequence token"))?;
        let cfg = GenerationConfig {
            logit_bias: HashMap::from([(eos, 100.0)]),
            ..greedy(8)
        };
        let got = alone_tokens(&spec, &[1, 2, 3], &cfg).await?;
        assert_eq!(
            shape(&got),
            shape(&alone_tokens(&plain, &[1, 2, 3], &cfg).await?)
        );
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].finish_reason, Some(FinishReason::Eos));
        assert_eq!(count(&recorder, names::SPECULATIVE_EMITTED_TOTAL), 1.0);
        Ok(())
    }

    #[tokio::test]
    async fn biased_greedy_matches_plain_with_unbiased_logprobs() -> Result<()> {
        let (plain, spec) = twins(CTX, 5)?;
        let cfg = GenerationConfig {
            logit_bias: HashMap::from([(7, 100.0)]),
            ..with_logprobs(12)
        };
        for prompt in RANDOM {
            let want = alone_tokens(&plain, prompt, &cfg).await?;
            let got = alone_tokens(&spec, prompt, &cfg).await?;
            assert!(got.iter().all(|t| t.token_id == 7));
            assert_eq!(shape(&got), shape(&want), "prompt {prompt:?}");
            assert_logprobs_close(&got, &want, &format!("prompt {prompt:?}"));
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_seed_repeats_its_sampled_output() -> Result<()> {
        let recorder = TestRecorder::install();
        let (_, spec) = twins(CTX, 5)?;
        let sampled = |seed| GenerationConfig {
            temperature: 0.9,
            seed: Some(seed),
            max_tokens: 24,
            ..GenerationConfig::default()
        };
        let prompt = REPETITIVE[0];
        let ids = |tokens: Vec<GeneratedToken>| -> Vec<u32> {
            tokens.into_iter().map(|t| t.token_id).collect()
        };
        let first = ids(alone_tokens(&spec, prompt, &sampled(7)).await?);
        assert_eq!(first.len(), 24);
        assert_eq!(ids(alone_tokens(&spec, prompt, &sampled(7)).await?), first);
        let mut differs = false;
        for seed in 8..12 {
            differs |= ids(alone_tokens(&spec, prompt, &sampled(seed)).await?) != first;
        }
        assert!(differs, "seeds 8..12 all repeat seed 7");
        assert!(count(&recorder, names::SPECULATIVE_EMITTED_TOTAL) > 0.0);
        assert!(!recorder
            .render()
            .contains(names::SPECULATIVE_FALLBACKS_TOTAL));
        Ok(())
    }

    #[tokio::test]
    async fn host_sampler_features_fall_back_unchanged_and_count() -> Result<()> {
        let recorder = TestRecorder::install();
        let (plain, spec) = twins(CTX, 5)?;
        let base = GenerationConfig {
            temperature: 0.8,
            seed: Some(11),
            max_tokens: 12,
            ..GenerationConfig::default()
        };
        let cases = [
            (
                "mirostat",
                GenerationConfig {
                    mirostat_mode: 2,
                    ..base.clone()
                },
            ),
            (
                "dry",
                GenerationConfig {
                    dry_multiplier: 0.8,
                    ..base.clone()
                },
            ),
            (
                "typical",
                GenerationConfig {
                    typical_p: 0.95,
                    ..base.clone()
                },
            ),
            (
                "dynatemp",
                GenerationConfig {
                    dynatemp_range: 0.5,
                    ..base.clone()
                },
            ),
        ];
        let prompt = REPETITIVE[0];
        for (reason, cfg) in cases {
            let want = alone_tokens(&plain, prompt, &cfg).await?;
            let got = alone_tokens(&spec, prompt, &cfg).await?;
            assert_eq!(shape(&got), shape(&want), "{reason}");
            assert_eq!(fallbacks(&recorder, reason), 1.0, "{reason}");
        }
        let plain = full_vocab_executor(None)?;
        let spec = speculating(full_vocab_executor(None)?, 5);
        let json = GenerationConfig {
            constraint: Some(Constraint::JsonObject),
            ..greedy(8)
        };
        let want = alone_tokens(&plain, &[1, 2, 3], &json).await?;
        assert_eq!(
            shape(&alone_tokens(&spec, &[1, 2, 3], &json).await?),
            shape(&want)
        );
        assert_eq!(fallbacks(&recorder, "grammar"), 1.0);
        assert_eq!(
            fallbacks(&recorder, "disabled"),
            0.0,
            "speculation off never counts"
        );
        assert_eq!(recorder.value(names::SPECULATIVE_EMITTED_TOTAL, &[]), None);
        Ok(())
    }

    #[tokio::test]
    async fn the_batched_route_ignores_the_setting() -> Result<()> {
        let recorder = TestRecorder::install();
        let mut plain = rig(POOL)?;
        let on = speculating(seeded_executor(RIG_SEED, Some(POOL))?, 5);
        let mut on = rig_on(on, POOL, TINY_CTX)?;
        let prompt = [1, 2, 3];
        let mut a = plain.scheduler.submit(prompt.to_vec(), greedy(6))?;
        drain(&mut plain).await?;
        let mut b = on.scheduler.submit(prompt.to_vec(), greedy(6))?;
        drain(&mut on).await?;
        assert_eq!(ids(&mut b)?, ids(&mut a)?);
        assert!(!recorder.render().contains("blazr_speculative_"));
        // The single-sequence paged path speculates and counts no fallback.
        let want = alone_tokens(&plain.executor, &prompt, &greedy(6)).await?;
        let got = alone_tokens(&on.executor, &prompt, &greedy(6)).await?;
        assert_eq!(shape(&got), shape(&want));
        assert_eq!(fallbacks(&recorder, "paged_attention"), 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn a_paged_executor_speculates_as_plain_paged_greedy() -> Result<()> {
        let recorder = TestRecorder::install();
        let pool = PagedPool {
            block_size: 4,
            blocks: 16,
        };
        let plain = seeded_paged(SEED, pool, CTX, false)?;
        let spec = speculating(seeded_paged(SEED, pool, CTX, false)?, 5);
        let prompt = tie_free(&plain, &REPETITIVE, 12).await?;
        let gen = with_logprobs(12);
        let want = alone_tokens(&plain, prompt, &gen).await?;
        let got = alone_tokens(&spec, prompt, &gen).await?;
        assert_eq!(shape(&got), shape(&want));
        assert_logprobs_close(&got, &want, "paged");
        assert!(count(&recorder, names::SPECULATIVE_EMITTED_TOTAL) > 0.0);
        assert!(!recorder
            .render()
            .contains(names::SPECULATIVE_FALLBACKS_TOTAL));
        let free = spec
            .shared_allocator()
            .ok_or_else(|| anyhow!("the executor has no shared allocator"))?
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?
            .free_block_count();
        assert_eq!(free, pool.blocks, "the request returned its blocks");
        Ok(())
    }

    #[tokio::test]
    async fn a_dropped_stream_reports_and_the_next_request_runs() -> Result<()> {
        let recorder = TestRecorder::install();
        let (plain, spec) = twins(CTX, 5)?;
        let prompt = tie_free(&plain, &REPETITIVE, 8).await?;
        {
            let config = greedy(MAX_TOKENS);
            let mut stream = std::pin::pin!(spec.generate(prompt, &config));
            let first = stream.next().await.ok_or_else(|| anyhow!("no token"))??;
            assert_eq!(first.finish_reason, None);
        }
        let emitted = count(&recorder, names::SPECULATIVE_EMITTED_TOTAL);
        assert!(
            (1.0..MAX_TOKENS as f64).contains(&emitted),
            "emitted {emitted}"
        );
        let want = plain.generate_text(prompt, &greedy(8)).await?;
        assert_eq!(
            outcome(&spec.generate_text(prompt, &greedy(8)).await?),
            outcome(&want)
        );
        Ok(())
    }

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
