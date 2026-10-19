//! The speculative target over an executor's contiguous KV cache.

use boostr::error::Result;
use boostr::inference::speculative::{Draw, SampleRule, SpeculativeModel, Verdict};
use boostr::inference::LayeredKvCache;
use boostr::{DType, Runtime, Tensor};

use super::target_glue::{
    bias_tensor, biased_rows, greedy_ids, inference_error, no_verify_tokens, verify_rows,
    LastLogits, TargetClient,
};
use crate::config::GenerationConfig;
use crate::engine::executor::Executor;
use crate::engine::tensor_parallel::KvCacheSpec;

/// One sequence of an executor's model as a boostr [`SpeculativeModel`].
///
/// It owns a contiguous KV cache built like the plain contiguous decode
/// path builds one. Each extend runs one forward at the cache end. The
/// request's `logit_bias` adds to every logits row before the argmax or the
/// verify, as the plain sampler adds it before its penalties.
pub(in crate::engine) struct ContiguousTarget<'a, R: Runtime<DType = DType>> {
    exec: &'a Executor<R>,
    kv: LayeredKvCache<R>,
    /// The fresh-cache shape. The first forward takes it.
    fresh: Option<KvCacheSpec>,
    /// The `[1, 1, vocab]` logit bias, cast to the logits dtype on first use.
    bias: Option<Tensor<R>>,
    /// Every token the cache holds, in order. It is the penalty context.
    fed: Vec<u32>,
    /// The `[1, M, vocab]` logits of the last extend, before the bias.
    last_logits: Option<Tensor<R>>,
}

impl<'a, R> ContiguousTarget<'a, R>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R>,
{
    /// An empty target for a prompt of `prompt_len` tokens and `max_tokens` new tokens.
    ///
    /// # Errors
    ///
    /// An error when the KV cache or the bias tensor cannot be built.
    pub(in crate::engine) fn new(
        exec: &'a Executor<R>,
        gen: &GenerationConfig,
        prompt_len: usize,
        max_tokens: usize,
    ) -> anyhow::Result<Self> {
        let (kv, fresh) = exec.contiguous_kv_cache(prompt_len, max_tokens)?;
        Ok(Self {
            exec,
            kv,
            fresh: Some(fresh),
            bias: bias_tensor(exec, gen)?,
            fed: Vec::with_capacity(prompt_len + max_tokens),
            last_logits: None,
        })
    }

    /// Every token the cache holds, in order.
    #[cfg(test)]
    pub(in crate::engine) fn fed(&self) -> &[u32] {
        &self.fed
    }

    /// The `[1, M, vocab]` logits of the last extend before the bias, if any.
    ///
    /// Row `i` scores the prefix that ends with fed token `i`. The plain
    /// decode loops report logprobs from unbiased logits too.
    pub(in crate::engine) fn last_logits(&self) -> Option<&Tensor<R>> {
        self.last_logits.as_ref()
    }

    /// One forward over `tokens` at the cache end: the logits, then the
    /// logits with the bias added.
    ///
    /// The cache and `fed` grow by `tokens.len()`.
    fn forward(&mut self, tokens: &[u32]) -> anyhow::Result<(Tensor<R>, Tensor<R>)> {
        let input = self.exec.create_input_tensor(tokens)?;
        let position = self.kv.seq_len();
        let logits =
            self.exec
                .forward_contiguous(&input, &mut self.kv, position, self.fresh.take())?;
        self.fed.extend_from_slice(tokens);
        biased_rows(self.exec, &mut self.bias, logits, tokens.len())
    }
}

impl<R> LastLogits<R> for ContiguousTarget<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R>,
{
    fn last_logits(&self) -> Option<&Tensor<R>> {
        ContiguousTarget::last_logits(self)
    }
}

impl<R> SpeculativeModel for ContiguousTarget<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R>,
{
    fn vocab_size(&self) -> usize {
        self.exec.vocab_size()
    }

    fn len(&self) -> usize {
        self.kv.seq_len()
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        self.kv.truncate(len)?;
        self.fed.truncate(len);
        Ok(())
    }

    fn extend_greedy(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let (logits, biased) = self.forward(tokens).map_err(inference_error)?;
        let ids = greedy_ids(self.exec, &biased).map_err(inference_error)?;
        self.last_logits = Some(logits);
        Ok(ids)
    }

    fn extend_sampled(&mut self, tokens: &[u32], rule: &SampleRule, draw: Draw) -> Result<Verdict> {
        if tokens.is_empty() {
            return Err(no_verify_tokens());
        }
        let (logits, biased) = self.forward(tokens).map_err(inference_error)?;
        let verdict = verify_rows(self.exec, &biased, &self.fed, tokens, rule, draw)?;
        self.last_logits = Some(logits);
        Ok(verdict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use anyhow::anyhow;
    use boostr::error::Error;

    use boostr::inference::speculative::{
        PromptLookup, SpeculativeConfig, SpeculativeGenerator, SpeculativeStats,
    };
    use boostr::CpuRuntime;

    use crate::engine::requests::batch_test_rig::alone_tokens;
    use crate::engine::speculative::{sample_rule, ContiguousTarget};
    use crate::engine::test_support::seeded_executor_ctx;

    const CTX: usize = 64;
    const SEED: u64 = 0x5bec;
    const MAX_TOKENS: usize = 40;
    /// Smallest top-2 logit margin of a compared greedy row.
    const NEAR_TIE: f32 = 1e-4;

    type Exec = Executor<CpuRuntime>;

    /// Plain prompts and repetitive ones where prompt lookup finds matches.
    fn prompts() -> Vec<Vec<u32>> {
        vec![
            vec![1],
            vec![5, 9, 2],
            vec![3, 7, 11, 3, 7, 11, 3, 7, 11],
            vec![4, 4, 4, 4, 4, 4],
            vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
        ]
    }

    fn greedy_plain() -> GenerationConfig {
        GenerationConfig {
            max_tokens: MAX_TOKENS,
            repeat_penalty: 1.0,
            ..GenerationConfig::greedy()
        }
    }

    async fn plain_ids(
        exec: &Exec,
        prompt: &[u32],
        gen: &GenerationConfig,
    ) -> anyhow::Result<Vec<u32>> {
        let tokens = alone_tokens(exec, prompt, gen).await?;
        Ok(tokens.into_iter().map(|t| t.token_id).collect())
    }

    /// Speculative output of `prompt` under `gen` with `k` drafts per step.
    fn speculative(
        exec: &Exec,
        prompt: &[u32],
        gen: &GenerationConfig,
        k: usize,
    ) -> anyhow::Result<(Vec<u32>, SpeculativeStats)> {
        let budget =
            crate::engine::token_budget(gen.max_tokens, exec.config().max_seq_len(), prompt.len());
        let target = ContiguousTarget::new(exec, gen, prompt.len(), budget)?;
        let config = SpeculativeConfig {
            num_draft_tokens: k,
            max_tokens: budget,
            eos_token_ids: exec.eos_ids(),
            sampling: sample_rule(gen),
        };
        let mut generator = SpeculativeGenerator::new(target, PromptLookup::default(), config)?;
        let (ids, _, stats) = generator.generate(prompt)?;
        Ok((ids, stats))
    }

    /// One forward over `prompt ++ output[..-1]`: the greedy ids of the rows
    /// that pick `output`, and the smallest top-2 margin of those rows.
    fn one_forward(exec: &Exec, prompt: &[u32], output: &[u32]) -> anyhow::Result<(Vec<u32>, f32)> {
        let gen = greedy_plain();
        let mut target = ContiguousTarget::new(exec, &gen, prompt.len(), output.len())?;
        let mut seq = prompt.to_vec();
        seq.extend_from_slice(&output[..output.len().saturating_sub(1)]);
        let ids = target.extend_greedy(&seq)?;
        let logits = target
            .last_logits()
            .ok_or_else(|| anyhow!("no logits after an extend"))?
            .to_vec::<f32>();
        let first = prompt.len() - 1;
        let margin = logits
            .chunks(exec.vocab_size())
            .skip(first)
            .map(|row| {
                let mut sorted = row.to_vec();
                sorted.sort_by(|a, b| b.total_cmp(a));
                sorted[0] - sorted[1]
            })
            .fold(f32::INFINITY, f32::min);
        Ok((ids[first..].to_vec(), margin))
    }

    #[tokio::test]
    async fn greedy_speculation_matches_plain_greedy() -> anyhow::Result<()> {
        let exec = seeded_executor_ctx(SEED, CTX)?;
        let gen = greedy_plain();
        let mut cases = Vec::new();
        for prompt in prompts() {
            let want = plain_ids(&exec, &prompt, &gen).await?;
            let (ids, margin) = one_forward(&exec, &prompt, &want)?;
            if margin < NEAR_TIE {
                continue;
            }
            assert_eq!(ids, want, "one multi-token forward, prompt {prompt:?}");
            cases.push((prompt, want));
        }
        assert!(
            cases.len() >= 3,
            "only {} prompts have no near-tie",
            cases.len()
        );
        for k in [1, 2, 5, 8] {
            let mut accepted = 0;
            for (prompt, want) in &cases {
                let (got, stats) = speculative(&exec, prompt, &gen, k)?;
                assert_eq!(&got, want, "k {k}, prompt {prompt:?}");
                assert!(stats.target_forwards <= want.len());
                accepted += stats.accepted;
            }
            assert!(accepted > 0, "k {k}: prompt lookup accepted no draft");
        }
        Ok(())
    }

    #[tokio::test]
    async fn greedy_with_penalties_matches_the_plain_path() -> anyhow::Result<()> {
        let exec = seeded_executor_ctx(SEED, CTX)?;
        let defaults = GenerationConfig {
            max_tokens: MAX_TOKENS,
            ..GenerationConfig::greedy()
        };
        assert_eq!((defaults.repeat_penalty, defaults.repeat_last_n), (1.1, 64));
        let short_window = GenerationConfig {
            repeat_last_n: 4,
            presence_penalty: 0.3,
            frequency_penalty: 0.2,
            ..defaults.clone()
        };
        for gen in [defaults, short_window] {
            assert!(
                sample_rule(&gen).is_some(),
                "penalties take the verify path"
            );
            for prompt in prompts() {
                let want = plain_ids(&exec, &prompt, &gen).await?;
                for k in [1, 5] {
                    let (got, _) = speculative(&exec, &prompt, &gen, k)?;
                    assert_eq!(got, want, "k {k}, prompt {prompt:?}, gen {gen:?}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn truncate_len_and_vocab_follow_the_contract() -> anyhow::Result<()> {
        let exec = seeded_executor_ctx(SEED, CTX)?;
        let vocab = exec.vocab_size();
        let mut target = ContiguousTarget::new(&exec, &greedy_plain(), 6, 8)?;
        assert_eq!(target.vocab_size(), vocab);
        assert!(target.is_empty());

        let ids = target.extend_greedy(&[1, 2, 3, 4])?;
        assert_eq!(ids.len(), 4);
        assert!(ids.iter().all(|&id| (id as usize) < vocab));
        assert_eq!((target.len(), target.fed()), (4, &[1, 2, 3, 4][..]));
        let full = target.last_logits().ok_or_else(|| anyhow!("logits"))?;
        assert_eq!(full.shape(), &[1, 4, vocab]);
        let full = full.to_vec::<f32>();

        target.truncate(2)?;
        assert_eq!((target.len(), target.fed()), (2, &[1, 2][..]));
        assert!(
            target.truncate(3).is_err(),
            "a truncate past len is an error"
        );

        // After the rollback the rows equal the rows of the first forward.
        target.extend_greedy(&[3, 4])?;
        let again = target
            .last_logits()
            .ok_or_else(|| anyhow!("logits"))?
            .to_vec::<f32>();
        let diff = again
            .iter()
            .zip(&full[2 * vocab..])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-4, "rows after a rollback differ by {diff}");

        assert!(target.extend_greedy(&[])?.is_empty());
        assert_eq!((target.len(), target.fed()), (4, &[1, 2, 3, 4][..]));
        Ok(())
    }

    #[test]
    fn the_sampled_extend_keeps_fed_in_step_with_the_cache() -> anyhow::Result<()> {
        let exec = seeded_executor_ctx(SEED, CTX)?;
        let gen = GenerationConfig {
            max_tokens: 8,
            ..GenerationConfig::greedy()
        };
        let rule = sample_rule(&gen).ok_or_else(|| anyhow!("penalties map to a rule"))?;
        let mut target = ContiguousTarget::new(&exec, &gen, 3, 8)?;
        target.extend_greedy(&[5, 9])?;
        let verdict = target.extend_sampled(&[2, 6, 6], &rule, Draw::for_step(0, 0))?;
        assert!(verdict.accepted <= 2);
        assert!((verdict.next as usize) < exec.vocab_size());
        assert_eq!((target.len(), target.fed()), (5, &[5, 9, 2, 6, 6][..]));
        let err = target.extend_sampled(&[], &rule, Draw::for_step(0, 1));
        assert!(matches!(
            err,
            Err(Error::InvalidArgument { arg: "tokens", .. })
        ));
        Ok(())
    }

    #[test]
    fn the_logit_bias_reaches_every_row() -> anyhow::Result<()> {
        let exec = seeded_executor_ctx(SEED, CTX)?;
        let gen = GenerationConfig {
            logit_bias: HashMap::from([(7, 100.0)]),
            ..greedy_plain()
        };
        let mut target = ContiguousTarget::new(&exec, &gen, 4, 4)?;
        assert_eq!(target.extend_greedy(&[1, 2, 3])?, vec![7, 7, 7]);
        Ok(())
    }
}
