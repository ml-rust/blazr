//! The boostr speculative traits over a [`PagedTarget`]: greedy extend,
//! truncate, and the sampled verify.

use boostr::error::{Error, Result};
use boostr::inference::speculative::{
    Draw, SampleRule, SpeculativeModel, SpeculativeProbModel, Verdict,
};
use boostr::{DType, RoPEPackedOps, Runtime, Tensor};

use super::target::PagedTarget;
use crate::engine::speculative::target_glue::{
    inference_error, no_verify_tokens, verify_rows, LastLogits, TargetClient,
};

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

    use anyhow::{anyhow, Result};
    use boostr::inference::speculative::{PromptLookup, SpeculativeConfig, SpeculativeGenerator};
    use boostr::CpuRuntime;

    use super::super::test_rig::{executor, free_blocks, greedy_plain, max_diff, rows, Exec, POOL};
    use crate::config::GenerationConfig;
    use crate::engine::speculative::{sample_rule, ContiguousTarget};

    /// Smallest top-2 logit margin of a compared greedy row.
    const NEAR_TIE: f32 = 1e-4;

    fn prompts() -> Vec<Vec<u32>> {
        vec![
            vec![5, 9, 2],
            vec![3, 7, 11, 3, 7, 11, 3, 7, 11],
            vec![4, 4, 4, 4, 4, 4],
            vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
        ]
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
}
