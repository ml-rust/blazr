//! The speculative target over an executor's paged KV pool.

use anyhow::anyhow;

use boostr::error::{Error, Result};
use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::speculative::{
    Draw, SampleRule, SpeculativeModel, SpeculativeProbModel, Verdict,
};
use boostr::{DType, RoPEPackedOps, Runtime, Tensor};

use super::target_glue::{
    bias_tensor, biased_rows, chunked_logits, greedy_ids, inference_error, no_verify_tokens,
    verify_rows, LastLogits, TargetClient,
};
use crate::config::GenerationConfig;
use crate::engine::executor::Executor;
use crate::engine::executors::executor_cache::prefix_cache_allocate;
use crate::engine::executors::paged_pool::PoolLease;
use crate::engine::requests::kv_recompute::{block_ids, block_slots};
use crate::engine::types::PrefillCancel;

/// One sequence of an executor's model as a boostr [`SpeculativeModel`], over
/// blocks of the executor's paged KV pool.
///
/// The target leases pool blocks as the sequence grows and returns them on
/// truncate and on drop. The first greedy extend starts the prompt. Its
/// prompt tokens lease their blocks through the prefix cache and skip the
/// leading tokens whose KV the reused blocks hold. Tokens past the prompt
/// length never reach the prefix cache. The request's `logit_bias` adds to
/// every logits row before the argmax or the verify.
pub(in crate::engine) struct PagedTarget<'a, R: Runtime<DType = DType>> {
    exec: &'a Executor<R>,
    pool: &'a LayeredPagedKvCache<R>,
    lease: PoolLease<'a>,
    /// Tokens in the request's prompt.
    prompt_len: usize,
    /// Positions whose KV the pool holds for this sequence.
    len: usize,
    /// Leading positions that blocks of the prefix cache hold. A truncate
    /// never cuts into them, because a later write would change shared KV.
    shared: usize,
    /// Every token the cache holds, in order. It is the penalty context.
    fed: Vec<u32>,
    /// The `[1, 1, vocab]` logit bias, cast to the logits dtype on first use.
    bias: Option<Tensor<R>>,
    /// The `[1, M, vocab]` logits of the last extend, before the bias.
    last_logits: Option<Tensor<R>>,
    /// Read before each forward chunk.
    cancel: PrefillCancel<'a>,
}

impl<'a, R> PagedTarget<'a, R>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R> + RoPEPackedOps<R>,
{
    /// An empty target for a prompt of `prompt_len` tokens. It holds no pool
    /// block yet.
    ///
    /// # Errors
    ///
    /// An error when paged attention is off, or the pool or the bias tensor
    /// cannot be built.
    pub(in crate::engine) fn new(
        exec: &'a Executor<R>,
        gen: &GenerationConfig,
        prompt_len: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            exec,
            pool: exec.paged_pool()?,
            lease: exec.pool_lease()?,
            prompt_len,
            len: 0,
            shared: 0,
            fed: Vec::new(),
            bias: bias_tensor(exec, gen)?,
            last_logits: None,
            cancel: PrefillCancel::never(),
        })
    }

    /// Stop an extend at its next chunk boundary once `cancel` trips.
    pub(in crate::engine) fn with_cancel(mut self, cancel: PrefillCancel<'a>) -> Self {
        self.cancel = cancel;
        self
    }

    /// Every token the cache holds, in order.
    #[cfg(test)]
    pub(in crate::engine) fn fed(&self) -> &[u32] {
        &self.fed
    }

    /// The `[1, M, vocab]` logits of the last extend before the bias, if any.
    ///
    /// Row `i` scores the prefix that ends with the `i`-th token that ran. A
    /// prompt extend that reuses cached blocks runs only the tokens after
    /// them, so its rows start at the first token that ran.
    pub(in crate::engine) fn last_logits(&self) -> Option<&Tensor<R>> {
        self.last_logits.as_ref()
    }

    /// The extend that fills an empty sequence is the prompt.
    fn is_first_extend(&self) -> bool {
        self.len == 0 && self.lease.blocks().is_empty()
    }

    /// Lease the blocks of the first extend `tokens` through the prefix cache.
    ///
    /// Only the leading tokens inside the prompt reach the prefix cache. The
    /// drafts after them lease their blocks in the forward. Returns the count
    /// of leading tokens whose KV the reused blocks hold. The last token
    /// always runs, so the forward yields its logits.
    fn lease_prompt(&mut self, tokens: &[u32]) -> anyhow::Result<usize> {
        let prompt = tokens.get(..self.prompt_len).unwrap_or(tokens);
        if prompt.is_empty() {
            return Ok(0);
        }
        let cached = prefix_cache_allocate(&mut self.lease, prompt)?;
        #[cfg(feature = "cuda")]
        {
            crate::engine::executors::executor_cache::gpu_prefix_cache_insert(
                &self.exec.gpu_prefix_cache,
                prompt,
                self.lease.block_size(),
                self.lease.blocks(),
            );
            self.lease
                .hold_gpu_prefix(crate::engine::executors::paged_pool::GpuPrefixRelease {
                    cache: &self.exec.gpu_prefix_cache,
                    prompt_tokens: prompt.to_vec(),
                });
        }
        let start = cached.min(tokens.len().saturating_sub(1));
        self.shared = start;
        self.len = start;
        self.fed
            .extend_from_slice(tokens.get(..start).unwrap_or_default());
        Ok(start)
    }

    /// The forward over `tokens` at the sequence end, in chunks: the logits,
    /// then the logits with the bias added.
    ///
    /// The lease covers the new positions. The sequence and `fed` grow by
    /// `tokens.len()`.
    fn forward(&mut self, tokens: &[u32]) -> anyhow::Result<(Tensor<R>, Tensor<R>)> {
        let start = self.len;
        let end = start + tokens.len();
        self.lease.cover(end)?;
        let (exec, pool) = (self.exec, self.pool);
        let blocks = self.lease.blocks();
        let block_size = self.lease.block_size();
        let table = block_ids(blocks);
        let block_table = Tensor::from_slice(&table, &[1, table.len()], exec.device())?;
        let logits = chunked_logits(exec, &self.cancel, tokens.len(), |r| {
            let (from, to) = (start + r.start, start + r.end);
            let slots = block_slots(blocks, block_size, from..to);
            let slot_mapping = Tensor::from_slice(&slots, &[r.len()], exec.device())?;
            let input = exec.create_input_tensor(&tokens[r])?;
            exec.model()?
                .forward_with_paged_kv_cache(
                    &input,
                    pool,
                    &slot_mapping,
                    &block_table,
                    to,
                    from,
                    None,
                    None,
                )
                .map_err(|e| anyhow!("Paged forward over positions {from}..{to} failed: {e}"))
        })?;
        self.len = end;
        self.fed.extend_from_slice(tokens);
        biased_rows(self.exec, &mut self.bias, logits, tokens.len())
    }

    /// The greedy ids of `tokens`, one per token.
    ///
    /// A prompt extend skips the cached leading tokens, so the forward has no
    /// logits for them. Their ids are the next prompt tokens. The caller of
    /// a prompt extend reads none of them.
    fn greedy(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<u32>> {
        let skipped = if self.is_first_extend() {
            self.lease_prompt(tokens)?
        } else {
            0
        };
        let (logits, biased) = self.forward(tokens.get(skipped..).unwrap_or_default())?;
        let ran = greedy_ids(self.exec, &biased)?;
        self.last_logits = Some(logits);
        let mut ids: Vec<u32> = tokens.iter().skip(1).take(skipped).copied().collect();
        ids.extend(ran);
        Ok(ids)
    }
}

impl<R> LastLogits<R> for PagedTarget<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R> + RoPEPackedOps<R>,
{
    fn last_logits(&self) -> Option<&Tensor<R>> {
        PagedTarget::last_logits(self)
    }
}

impl<R> SpeculativeModel for PagedTarget<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R> + RoPEPackedOps<R>,
{
    fn vocab_size(&self) -> usize {
        self.exec.vocab_size()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(Error::InvalidArgument {
                arg: "len",
                reason: format!(
                    "cannot cut to {len}; the sequence holds {} positions",
                    self.len
                ),
            });
        }
        if len < self.shared {
            return Err(Error::InvalidArgument {
                arg: "len",
                reason: format!(
                    "cannot cut to {len}; the first {} positions sit in shared prefix blocks",
                    self.shared
                ),
            });
        }
        self.len = len;
        self.fed.truncate(len);
        self.lease.shrink_to(len).map_err(inference_error)
    }

    fn extend_greedy(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        self.greedy(tokens).map_err(inference_error)
    }

    fn extend_sampled(&mut self, tokens: &[u32], rule: &SampleRule, draw: Draw) -> Result<Verdict> {
        self.extend_sampled_with_probs(tokens, rule, draw, None)
    }
}

impl<R> SpeculativeProbModel<R> for PagedTarget<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TargetClient<R> + RoPEPackedOps<R>,
{
    fn extend_sampled_with_probs(
        &mut self,
        tokens: &[u32],
        rule: &SampleRule,
        draw: Draw,
        draft_probs: Option<&Tensor<R>>,
    ) -> Result<Verdict> {
        if tokens.is_empty() {
            return Err(no_verify_tokens());
        }
        let (logits, biased) = self.forward(tokens).map_err(inference_error)?;
        let verdict = verify_rows(
            self.exec,
            &biased,
            &self.fed,
            tokens,
            rule,
            draw,
            draft_probs,
        )?;
        self.last_logits = Some(logits);
        Ok(verdict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;
    use boostr::inference::speculative::{PromptLookup, SpeculativeConfig, SpeculativeGenerator};
    use boostr::inference::BlockAllocator;
    use boostr::CpuRuntime;

    use crate::engine::speculative::{sample_rule, ContiguousTarget};
    use crate::engine::test_support::{seeded_paged, PagedPool};

    const POOL: PagedPool = PagedPool {
        block_size: 4,
        blocks: 16,
    };
    const CTX: usize = 64;
    const SEED: u64 = 0x5bec;
    const MAX_TOKENS: usize = 40;
    /// Smallest top-2 logit margin of a compared greedy row.
    const NEAR_TIE: f32 = 1e-4;

    type Exec = Executor<CpuRuntime>;

    fn executor(prefix_cache: bool) -> Result<Exec> {
        seeded_paged(SEED, POOL, CTX, prefix_cache)
    }

    fn free_blocks(exec: &Exec) -> Result<usize> {
        let shared = exec
            .shared_allocator()
            .ok_or_else(|| anyhow!("the executor has no shared allocator"))?;
        let free = shared
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?
            .free_block_count();
        Ok(free)
    }

    fn prompts() -> Vec<Vec<u32>> {
        vec![
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

    fn sampled() -> GenerationConfig {
        GenerationConfig {
            temperature: 0.8,
            top_k: 8,
            seed: Some(11),
            ..greedy_plain()
        }
    }

    /// Speculative output of `prompt` over `model`, with `k` drafts per step.
    fn run<M: SpeculativeModel>(
        exec: &Exec,
        model: M,
        prompt: &[u32],
        gen: &GenerationConfig,
        k: usize,
    ) -> Result<Vec<u32>> {
        let budget =
            crate::engine::token_budget(gen.max_tokens, exec.config().max_seq_len(), prompt.len());
        let config = SpeculativeConfig {
            num_draft_tokens: k,
            max_tokens: budget,
            eos_token_ids: exec.eos_ids(),
            sampling: sample_rule(gen),
        };
        let mut generator = SpeculativeGenerator::new(model, PromptLookup::default(), config)?;
        Ok(generator.generate(prompt)?.0)
    }

    fn contiguous<'a>(
        exec: &'a Exec,
        gen: &GenerationConfig,
        prompt: &[u32],
    ) -> Result<ContiguousTarget<'a, CpuRuntime>> {
        let budget =
            crate::engine::token_budget(gen.max_tokens, exec.config().max_seq_len(), prompt.len());
        ContiguousTarget::new(exec, gen, prompt.len(), budget)
    }

    /// The smallest top-2 margin of the rows that pick `output` after `prompt`.
    fn min_margin(exec: &Exec, prompt: &[u32], output: &[u32]) -> Result<f32> {
        let mut target = ContiguousTarget::new(exec, &greedy_plain(), prompt.len(), output.len())?;
        let mut seq = prompt.to_vec();
        seq.extend_from_slice(&output[..output.len().saturating_sub(1)]);
        target.extend_greedy(&seq)?;
        let logits = target
            .last_logits()
            .ok_or_else(|| anyhow!("no logits after an extend"))?
            .to_vec::<f32>();
        Ok(logits
            .chunks(exec.vocab_size())
            .skip(prompt.len() - 1)
            .map(|row| {
                let mut sorted = row.to_vec();
                sorted.sort_by(|a, b| b.total_cmp(a));
                sorted[0] - sorted[1]
            })
            .fold(f32::INFINITY, f32::min))
    }

    fn rows<M: LastLogits<CpuRuntime>>(target: &M) -> Result<Vec<f32>> {
        Ok(target
            .last_logits()
            .ok_or_else(|| anyhow!("no logits after an extend"))?
            .to_vec::<f32>())
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn greedy_matches_the_contiguous_target() -> Result<()> {
        let exec = executor(false)?;
        let gen = greedy_plain();
        let mut compared = 0;
        for prompt in prompts() {
            let want = run(&exec, contiguous(&exec, &gen, &prompt)?, &prompt, &gen, 4)?;
            if min_margin(&exec, &prompt, &want)? < NEAR_TIE {
                continue;
            }
            for k in [1, 5] {
                let target = PagedTarget::new(&exec, &gen, prompt.len())?;
                assert_eq!(
                    run(&exec, target, &prompt, &gen, k)?,
                    want,
                    "k {k}, prompt {prompt:?}"
                );
            }
            compared += 1;
        }
        assert!(compared >= 3, "only {compared} prompts have no near-tie");
        assert_eq!(free_blocks(&exec)?, POOL.blocks, "every target dropped");
        Ok(())
    }

    #[test]
    fn sampled_matches_the_contiguous_target() -> Result<()> {
        let exec = executor(false)?;
        let gen = sampled();
        for prompt in prompts() {
            let want = run(&exec, contiguous(&exec, &gen, &prompt)?, &prompt, &gen, 4)?;
            let target = PagedTarget::new(&exec, &gen, prompt.len())?;
            assert_eq!(
                run(&exec, target, &prompt, &gen, 4)?,
                want,
                "prompt {prompt:?}"
            );
        }

        let rule = sample_rule(&gen).ok_or_else(|| anyhow!("a sampled rule"))?;
        let mut a = contiguous(&exec, &gen, &[5, 9, 2])?;
        let mut b = PagedTarget::new(&exec, &gen, 3)?;
        a.extend_greedy(&[5, 9])?;
        b.extend_greedy(&[5, 9])?;
        let draw = Draw::for_step(7, 0);
        assert_eq!(
            b.extend_sampled(&[2, 6, 6], &rule, draw)?,
            a.extend_sampled(&[2, 6, 6], &rule, draw)?
        );
        assert_eq!((b.len(), b.fed()), (5, &[5, 9, 2, 6, 6][..]));
        assert!(matches!(
            b.extend_sampled(&[], &rule, draw),
            Err(Error::InvalidArgument { arg: "tokens", .. })
        ));
        Ok(())
    }

    #[test]
    fn a_rollback_across_a_block_boundary_equals_a_fresh_sequence() -> Result<()> {
        let exec = executor(false)?;
        let gen = greedy_plain();
        let mut target = PagedTarget::new(&exec, &gen, 3)?;
        target.extend_greedy(&[1, 2, 3])?;
        target.extend_greedy(&[4, 5, 6, 7, 8])?;
        assert_eq!(target.len(), 8);
        target.truncate(2)?;
        assert_eq!((target.len(), target.fed()), (2, &[1, 2][..]));
        assert!(target.truncate(3).is_err(), "a cut past len is an error");
        target.extend_greedy(&[3, 9, 10, 11, 12])?;
        let again = rows(&target)?;

        let mut fresh = PagedTarget::new(&exec, &gen, 7)?;
        fresh.extend_greedy(&[1, 2, 3, 9, 10, 11, 12])?;
        let vocab = exec.vocab_size();
        let diff = max_diff(&again, &rows(&fresh)?[2 * vocab..]);
        assert!(diff < 1e-4, "rows after a rollback differ by {diff}");
        Ok(())
    }

    #[test]
    fn truncate_returns_the_blocks_past_the_kept_tokens() -> Result<()> {
        let exec = executor(false)?;
        let mut target = PagedTarget::new(&exec, &greedy_plain(), 5)?;
        target.extend_greedy(&[1, 2, 3, 4, 5])?;
        target.extend_greedy(&[6, 7, 8])?;
        target.extend_greedy(&[9, 10, 11, 12, 13])?;
        let needed = |tokens: usize| tokens.div_ceil(POOL.block_size);
        assert_eq!(free_blocks(&exec)?, POOL.blocks - needed(13));
        target.truncate(6)?;
        assert_eq!(free_blocks(&exec)?, POOL.blocks - needed(6));
        target.truncate(0)?;
        assert_eq!(free_blocks(&exec)?, POOL.blocks);
        target.extend_greedy(&[1, 2, 3, 4, 5])?;
        drop(target);
        assert_eq!(free_blocks(&exec)?, POOL.blocks);
        Ok(())
    }

    #[test]
    fn a_second_request_reuses_the_prompt_blocks_of_the_first() -> Result<()> {
        let exec = executor(true)?;
        let gen = greedy_plain();
        let prompt: Vec<u32> = (0..17).map(|i| (i * 7 + 3) % 31 + 1).collect();

        let mut first = PagedTarget::new(&exec, &gen, prompt.len())?;
        let want = first.extend_greedy(&prompt)?;
        let want_row = rows(&first)?;
        drop(first);
        let stats = exec
            .prefix_cache_stats()
            .ok_or_else(|| anyhow!("the prefix cache is on"))?;
        assert_eq!(stats.cached_blocks, prompt.len() / POOL.block_size);
        assert_eq!(stats.cache_hits, 0);

        let mut second = PagedTarget::new(&exec, &gen, prompt.len())?;
        let got = second.extend_greedy(&prompt)?;
        let stats = exec
            .prefix_cache_stats()
            .ok_or_else(|| anyhow!("the prefix cache is on"))?;
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.blocks_from_cache, 4);
        assert_eq!((second.len(), second.fed()), (17, &prompt[..]));
        assert_eq!(got.len(), prompt.len(), "one id per fed token");
        assert_eq!(got.last(), want.last());
        let vocab = exec.vocab_size();
        let diff = max_diff(&rows(&second)?, &want_row[16 * vocab..]);
        assert!(
            diff < 1e-4,
            "the reused prefix changed the logits by {diff}"
        );
        assert!(
            second.truncate(8).is_err(),
            "a cut into shared blocks is an error"
        );
        second.truncate(16)?;
        Ok(())
    }

    #[test]
    fn drafts_past_the_prompt_never_reach_the_prefix_cache() -> Result<()> {
        let exec = executor(true)?;
        let mut target = PagedTarget::new(&exec, &greedy_plain(), 1)?;
        let tokens: Vec<u32> = (1..=13).collect();
        target.extend_greedy(&tokens)?;
        assert_eq!(target.len(), tokens.len());
        let stats = exec
            .prefix_cache_stats()
            .ok_or_else(|| anyhow!("the prefix cache is on"))?;
        assert_eq!(stats.cached_blocks, 0, "a 1-token prompt fills no block");
        drop(target);
        assert_eq!(free_blocks(&exec)?, POOL.blocks);
        Ok(())
    }

    #[test]
    fn an_exhausted_pool_is_a_clean_error() -> Result<()> {
        let exec = executor(false)?;
        let too_long = vec![1; POOL.blocks * POOL.block_size + 1];
        let mut target = PagedTarget::new(&exec, &greedy_plain(), too_long.len())?;
        let err = target.extend_greedy(&too_long);
        assert!(
            matches!(&err, Err(e) if e.to_string().contains("KV blocks")),
            "got {err:?}"
        );
        assert_eq!(target.len(), 0);
        assert_eq!(free_blocks(&exec)?, POOL.blocks);
        assert_eq!(
            target.extend_greedy(&[1, 2])?.len(),
            2,
            "the target recovers"
        );
        Ok(())
    }
}
