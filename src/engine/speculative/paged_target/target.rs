//! The [`PagedTarget`] type: its prompt lease through the prefix cache and
//! its chunked forward over the paged KV pool.

use anyhow::anyhow;

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::{DType, RoPEPackedOps, Runtime, Tensor};

use crate::config::GenerationConfig;
use crate::engine::executor::Executor;
use crate::engine::executors::executor_cache::prefix_cache_allocate;
use crate::engine::executors::paged_pool::PoolLease;
use crate::engine::requests::kv_recompute::{block_ids, block_slots};
use crate::engine::speculative::target_glue::{
    bias_tensor, biased_rows, chunked_logits, greedy_ids, TargetClient,
};
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
///
/// [`SpeculativeModel`]: boostr::inference::speculative::SpeculativeModel
pub(in crate::engine) struct PagedTarget<'a, R: Runtime<DType = DType>> {
    pub(super) exec: &'a Executor<R>,
    pool: &'a LayeredPagedKvCache<R>,
    pub(super) lease: PoolLease<'a>,
    /// Tokens in the request's prompt.
    prompt_len: usize,
    /// Positions whose KV the pool holds for this sequence.
    pub(super) len: usize,
    /// Leading positions that blocks of the prefix cache hold. A truncate
    /// never cuts into them, because a later write would change shared KV.
    pub(super) shared: usize,
    /// Every token the cache holds, in order. It is the penalty context.
    pub(super) fed: Vec<u32>,
    /// The `[1, 1, vocab]` logit bias, cast to the logits dtype on first use.
    bias: Option<Tensor<R>>,
    /// The `[1, M, vocab]` logits of the last extend, before the bias.
    pub(super) last_logits: Option<Tensor<R>>,
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
    pub(super) fn forward(&mut self, tokens: &[u32]) -> anyhow::Result<(Tensor<R>, Tensor<R>)> {
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
    pub(super) fn greedy(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<u32>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;
    use boostr::inference::speculative::SpeculativeModel;

    use super::super::test_rig::{executor, free_blocks, greedy_plain, max_diff, rows, POOL};

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
