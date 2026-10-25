//! The speculative plan of one request and the target each plan runs over.

use anyhow::Result;
use async_stream::stream;
use futures::Stream;

use boostr::{DType, RoPEPackedOps, Runtime};

use crate::config::{DrafterSpec, GenerationConfig};
use crate::engine::executor::Executor;
use crate::engine::speculative::target_glue::TargetClient;
use crate::engine::speculative::{ContiguousTarget, Fallback, PagedTarget, SpeculativePlan};
use crate::engine::types::{GeneratedToken, PrefillCancel};
use crate::telemetry::record::record_speculative_fallback;

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
    ///
    /// The target and the draft model run each extend in chunks of
    /// `inference.prefill_chunk_size` tokens and read `cancel` before each
    /// chunk. A cancel ends the stream with no item.
    pub(in crate::engine) fn generate_speculative<'a>(
        &'a self,
        prompt: &'a [u32],
        gen: &'a GenerationConfig,
        plan: SpeculativePlan,
        cancel: PrefillCancel<'a>,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            let SpeculativePlan { config, drafter } = plan;
            let paged = self.config.inference.paged_attention;
            match drafter {
                DrafterSpec::PromptLookup(lookup) if paged => {
                    let target = PagedTarget::new(self, gen, prompt.len())?.with_cancel(cancel);
                    for await item in self.drive_speculative(target, prompt, gen, config, lookup, cancel) {
                        yield item;
                    }
                }
                DrafterSpec::PromptLookup(lookup) => {
                    let target = ContiguousTarget::new(self, gen, prompt.len(), config.max_tokens)?
                        .with_cancel(cancel);
                    for await item in self.drive_speculative(target, prompt, gen, config, lookup, cancel) {
                        yield item;
                    }
                }
                DrafterSpec::Model if paged => {
                    let target = PagedTarget::new(self, gen, prompt.len())?.with_cancel(cancel);
                    let drafter = self.model_drafter(gen, prompt.len(), &config, cancel)?;
                    for await item in self.drive_speculative(target, prompt, gen, config, drafter, cancel) {
                        yield item;
                    }
                }
                DrafterSpec::Model => {
                    let target = ContiguousTarget::new(self, gen, prompt.len(), config.max_tokens)?
                        .with_cancel(cancel);
                    let drafter = self.model_drafter(gen, prompt.len(), &config, cancel)?;
                    for await item in self.drive_speculative(target, prompt, gen, config, drafter, cancel) {
                        yield item;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use boostr::inference::BlockAllocator;

    use super::super::test_rig::{
        assert_logprobs_close, count, shape, speculating, tie_free, twins, with_logprobs, CTX,
        REPETITIVE, SEED,
    };
    use super::*;
    use crate::config::Constraint;
    use crate::engine::requests::batch_test_rig::{
        alone_tokens, drain, greedy, ids, rig, rig_on, POOL, SEED as RIG_SEED, TINY_CTX,
    };
    use crate::engine::test_support::{
        full_vocab_executor, seeded_executor, seeded_paged, PagedPool,
    };
    use crate::telemetry::names;
    use crate::telemetry::test_recorder::TestRecorder;

    fn fallbacks(recorder: &TestRecorder, reason: &str) -> f64 {
        recorder
            .value(names::SPECULATIVE_FALLBACKS_TOTAL, &[("reason", reason)])
            .unwrap_or(0.0)
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
}
