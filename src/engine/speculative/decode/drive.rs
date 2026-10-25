//! The speculative loop over one target and the stats it records.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::inference::speculative::{
    DraftSource, FinishReason as SpeculativeFinish, SpeculativeConfig, SpeculativeGenerator,
    SpeculativeModel, SpeculativeStats, StepOutput,
};
use boostr::{DType, RoPEPackedOps, Runtime};

use crate::config::GenerationConfig;
use crate::engine::executor::Executor;
use crate::engine::speculative::target_glue::{LastLogits, TargetClient};
use crate::engine::types::{FinishReason, GeneratedToken, PrefillCancel};
use crate::telemetry::record::{record_speculative, SpeculativeCounts};
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
    /// The speculative loop over `target`, drafting with `drafter`.
    ///
    /// An error after `cancel` tripped is the stopped extend: the stream
    /// ends with no item.
    pub(super) fn drive_speculative<'a, M, D>(
        &'a self,
        target: M,
        prompt: &'a [u32],
        gen: &'a GenerationConfig,
        config: SpeculativeConfig,
        drafter: D,
        cancel: PrefillCancel<'a>,
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
            if let Err(e) = generator.start(prompt) {
                if cancel.is_cancelled() {
                    return;
                }
                Err::<(), _>(e)?;
            }
            tracing::info!(phase = "prefill_end", backend = "speculative");
            tracing::info!(phase = "decode_start", backend = "speculative", max_tokens, num_draft_tokens);
            loop {
                let StepOutput { tokens, finish } = match generator.step() {
                    Ok(step) => step,
                    Err(_) if cancel.is_cancelled() => return,
                    Err(e) => Err::<StepOutput, _>(e)?,
                };
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

    use futures::StreamExt;

    use super::super::test_rig::{
        assert_logprobs_close, count, margin, shape, speculating, tie_free, twins, with_logprobs,
        CTX, MAX_TOKENS, NEAR_TIE, RANDOM, REPETITIVE,
    };
    use crate::engine::requests::batch_test_rig::{alone_tokens, greedy};
    use crate::engine::test_support::full_vocab_executor;
    use crate::engine::types::GenerationResult;
    use crate::telemetry::names;
    use crate::telemetry::test_recorder::TestRecorder;

    fn outcome(result: &GenerationResult) -> (String, FinishReason, usize) {
        (
            result.text.clone(),
            result.finish_reason,
            result.completion_tokens,
        )
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
}
