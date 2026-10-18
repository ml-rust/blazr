//! Per-sequence sampling state for the continuous batching engine.
//!
//! Each batched sequence carries its own [`SamplerState`], as the
//! single-sequence loop does. Every row samples through
//! [`Executor::sample_step`], so the grammar, mirostat and the penalty
//! history apply per sequence, and a row produces the ids it produces alone.
//!
//! A state starts at the sequence's first prefill. [`BatchSeqs::end_seq`] is
//! the one place a state ends. Preemption keeps the state: mirostat's
//! running estimate depends on the logits each step saw, so a replay over the
//! generated ids cannot rebuild it.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use boostr::inference::scheduler::SequenceId;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, GrammarDfaOps, NormalizationOps, Runtime,
    SamplingOps, SamplingScratch, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::engine::executor::Executor;
use crate::engine::sampler_step::SamplerState;

/// One batched sequence's sampling policy and state.
struct BatchSeq<R: Runtime<DType = DType>> {
    /// The config the request was submitted with.
    gen_config: GenerationConfig,
    /// History, mirostat state and grammar state, as the single-sequence loop
    /// carries them.
    sampler: SamplerState<R>,
    /// The scheduler freed this sequence's KV blocks. The next decode must
    /// recompute them first.
    kv_stale: bool,
}

/// The sampling state of every sequence the batching engine carries.
pub(in crate::engine) struct BatchSeqs<R: Runtime<DType = DType>> {
    seqs: HashMap<SequenceId, BatchSeq<R>>,
    /// Device buffers one sample writes into. Rows sample one after another
    /// and each reads its id back before the next starts, so one scratch
    /// serves every row.
    scratch: SamplingScratch<R>,
    /// KV recomputes this engine ran, for the preemption tests.
    #[cfg(test)]
    pub(super) recomputes: usize,
}

impl<R: Runtime<DType = DType>> BatchSeqs<R> {
    /// An empty set whose scratch is sized for `vocab_size` on `device`.
    ///
    /// # Errors
    ///
    /// An error when the scratch allocation fails.
    pub(in crate::engine) fn new(vocab_size: usize, device: &R::Device) -> Result<Self> {
        let scratch = SamplingScratch::new(vocab_size, device).map_err(|e| {
            anyhow!("allocating the batch sampling scratch for a vocab of {vocab_size} failed: {e}")
        })?;
        Ok(Self {
            seqs: HashMap::new(),
            scratch,
            #[cfg(test)]
            recomputes: 0,
        })
    }

    /// Whether `seq_id` has a sampling state here.
    pub(in crate::engine) fn contains(&self, seq_id: SequenceId) -> bool {
        self.seqs.contains_key(&seq_id)
    }

    /// Drop `seq_id`'s state. Every exit path of a sequence calls this.
    pub(in crate::engine) fn end_seq(&mut self, seq_id: SequenceId) {
        self.seqs.remove(&seq_id);
    }

    /// Drop the state of every sequence `is_live` rejects, through
    /// [`Self::end_seq`].
    pub(in crate::engine) fn end_gone(&mut self, is_live: impl Fn(SequenceId) -> bool) {
        let gone: Vec<SequenceId> = self
            .seqs
            .keys()
            .copied()
            .filter(|&id| !is_live(id))
            .collect();
        for seq_id in gone {
            self.end_seq(seq_id);
        }
    }

    /// Mark `preempted` as holding no KV. Ids with no state are skipped.
    pub(in crate::engine) fn mark_stale(&mut self, preempted: &[SequenceId]) {
        for seq_id in preempted {
            if let Some(seq) = self.seqs.get_mut(seq_id) {
                seq.kv_stale = true;
            }
        }
    }

    /// Whether `seq_id` needs a KV recompute, clearing the mark.
    pub(in crate::engine) fn take_stale(&mut self, seq_id: SequenceId) -> bool {
        let stale = self
            .seqs
            .get_mut(&seq_id)
            .is_some_and(|seq| std::mem::replace(&mut seq.kv_stale, false));
        #[cfg(test)]
        if stale {
            self.recomputes += 1;
        }
        stale
    }

    /// Sequences that hold a state, for the cleanup tests.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.seqs.len()
    }
}

impl<R: Runtime<DType = DType>> BatchSeqs<R>
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
    /// Start `seq_id`'s state from its prompt and its request's config.
    ///
    /// The history starts as the prompt. The grammar state starts from the
    /// DFA admission compiled, as the single-sequence loop does it.
    ///
    /// # Errors
    ///
    /// An error when the constraint does not compile or does not upload.
    pub(in crate::engine) fn begin(
        &mut self,
        executor: &Executor<R>,
        seq_id: SequenceId,
        prompt_tokens: &[u32],
        gen_config: GenerationConfig,
    ) -> Result<()> {
        let sampler = executor.sampler_state(prompt_tokens, &gen_config)?;
        self.seqs.insert(
            seq_id,
            BatchSeq {
                gen_config,
                sampler,
                kv_stale: false,
            },
        );
        Ok(())
    }

    /// Sample `seq_id`'s next id from its `[1, seq, vocab]` logits row.
    ///
    /// The id goes into the sequence's history and advances its grammar.
    ///
    /// # Errors
    ///
    /// An error when `seq_id` has no state, or when the sample fails.
    pub(in crate::engine) fn sample(
        &mut self,
        executor: &Executor<R>,
        seq_id: SequenceId,
        logits: &Tensor<R>,
    ) -> Result<u32> {
        let seq = self.seqs.get_mut(&seq_id).ok_or_else(|| {
            anyhow!(
                "sequence {seq_id} has no sampler state. A batched sequence must pass \
                 prefill in this engine before it samples."
            )
        })?;
        executor.sample_step(
            logits,
            &mut seq.sampler,
            &seq.gen_config,
            Some(&self.scratch),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::CpuRuntime;

    use crate::engine::test_support::tiny_executor;

    /// `[1, 1, vocab]` logits that peak at `peak`.
    fn peaked(exec: &Executor<CpuRuntime>, peak: usize) -> Result<Tensor<CpuRuntime>> {
        let vocab = exec.vocab_size();
        let mut row = vec![0.0f32; vocab];
        row[peak] = 20.0;
        Ok(Tensor::from_slice(&row, &[1, 1, vocab], exec.device())?)
    }

    fn greedy() -> GenerationConfig {
        GenerationConfig {
            repeat_penalty: 1.0,
            ..GenerationConfig::greedy()
        }
    }

    #[test]
    fn each_sequence_keeps_its_own_history() -> Result<()> {
        let exec = tiny_executor()?;
        let mut seqs = BatchSeqs::new(exec.vocab_size(), exec.device())?;
        seqs.begin(&exec, 1, &[1, 2], greedy())?;
        seqs.begin(&exec, 2, &[3], greedy())?;
        assert_eq!(seqs.sample(&exec, 1, &peaked(&exec, 7)?)?, 7);
        assert_eq!(seqs.sample(&exec, 2, &peaked(&exec, 9)?)?, 9);
        let history = |id| seqs.seqs.get(&id).map(|s| s.sampler.token_history.clone());
        assert_eq!(history(1), Some(vec![1, 2, 7]));
        assert_eq!(history(2), Some(vec![3, 9]));
        Ok(())
    }

    #[test]
    fn sampling_an_unknown_sequence_is_an_error() -> Result<()> {
        let exec = tiny_executor()?;
        let mut seqs = BatchSeqs::new(exec.vocab_size(), exec.device())?;
        let err = seqs.sample(&exec, 5, &peaked(&exec, 1)?);
        assert!(err.is_err_and(|e| e.to_string().contains("sequence 5")));
        Ok(())
    }

    #[test]
    fn stale_marks_clear_once_and_end_gone_drops_dead_sequences() -> Result<()> {
        let exec = tiny_executor()?;
        let mut seqs = BatchSeqs::new(exec.vocab_size(), exec.device())?;
        seqs.begin(&exec, 1, &[1], greedy())?;
        seqs.begin(&exec, 2, &[2], greedy())?;
        seqs.mark_stale(&[1, 99]);
        assert!(seqs.take_stale(1));
        assert!(!seqs.take_stale(1));
        assert!(!seqs.take_stale(2));
        assert!(!seqs.take_stale(99));

        seqs.end_gone(|id| id == 2);
        assert!(!seqs.contains(1));
        assert!(seqs.contains(2));
        seqs.end_seq(2);
        assert_eq!(seqs.len(), 0);
        Ok(())
    }
}
