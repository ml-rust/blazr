//! Engine types shared across executor, sampling, and cuda_graphs modules.

use std::ops::Range;

/// A probe a single-sequence prefill reads before each prompt chunk.
///
/// `inference.prefill_chunk_size` above 0 splits a prompt into chunks of
/// that many tokens. The prefill reads the probe before every chunk, so a
/// request whose caller left stops at the next chunk boundary and drops its
/// KV cache. A forward that runs is never interrupted, so a chunk size of 0
/// prefills the whole prompt in one forward that runs to its end.
#[derive(Clone, Copy, Default)]
pub struct PrefillCancel<'a> {
    probe: Option<&'a (dyn Fn() -> bool + Sync)>,
}

impl<'a> PrefillCancel<'a> {
    /// A prefill no caller can stop.
    pub fn never() -> Self {
        Self { probe: None }
    }

    /// A prefill that stops once `probe` returns `true`.
    pub fn when(probe: &'a (dyn Fn() -> bool + Sync)) -> Self {
        Self { probe: Some(probe) }
    }

    /// Whether the caller asked the prefill to stop.
    pub fn is_cancelled(&self) -> bool {
        self.probe.is_some_and(|probe| probe())
    }

    /// Run `forward` over `range` in chunks of `chunk` positions, or over
    /// all of `range` at once when `chunk` is 0.
    ///
    /// The probe is read before each chunk. Returns the last chunk's output,
    /// or `None` when the probe stopped the prefill or `range` is empty.
    ///
    /// # Errors
    ///
    /// The first error `forward` returns.
    pub(crate) fn run<T>(
        &self,
        range: Range<usize>,
        chunk: usize,
        mut forward: impl FnMut(Range<usize>) -> anyhow::Result<T>,
    ) -> anyhow::Result<Option<T>> {
        let step = if chunk == 0 { range.len() } else { chunk };
        let mut last = None;
        let mut start = range.start;
        while start < range.end {
            if self.is_cancelled() {
                tracing::debug!(
                    prefilled = start - range.start,
                    total = range.len(),
                    "prefill stopped at a chunk boundary: the caller left"
                );
                return Ok(None);
            }
            let end = (start + step).min(range.end);
            last = Some(forward(start..end)?);
            start = end;
        }
        Ok(last)
    }
}

/// Why generation stopped
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit an EOS token
    Eos,
    /// Reached max_tokens limit
    Length,
    /// Matched a stop sequence
    Stop,
}

impl FinishReason {
    /// OpenAI API string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Eos => "stop",
            FinishReason::Length => "length",
            FinishReason::Stop => "stop",
        }
    }
}

/// A single token's log probability entry
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    /// Token ID
    pub token_id: u32,
    /// Decoded text for this token
    pub text: String,
    /// Log probability
    pub logprob: f32,
}

/// A generated token with metadata
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    /// Token ID
    pub token_id: u32,
    /// Decoded text
    pub text: String,
    /// Log probability of the chosen token (if logprobs enabled)
    pub logprob: Option<f32>,
    /// Top-N alternative tokens with their log probabilities (if logprobs enabled)
    pub top_logprobs: Option<Vec<TokenLogprob>>,
    /// Set on the final token to indicate why generation ended
    pub finish_reason: Option<FinishReason>,
}

/// Result of a complete (non-streaming) generation
#[derive(Debug)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,
    /// Number of prompt tokens
    pub prompt_tokens: usize,
    /// Number of completion tokens generated
    pub completion_tokens: usize,
    /// Why generation finished
    pub finish_reason: FinishReason,
    /// Time to first token (prefill duration) in milliseconds
    pub prompt_eval_duration_ms: u64,
    /// Per-token logprobs (when logprobs enabled)
    pub token_logprobs: Option<Vec<GeneratedToken>>,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// The ranges `cancel` runs over `range` in chunks of `chunk`.
    fn ranges(cancel: PrefillCancel<'_>, range: Range<usize>, chunk: usize) -> Vec<Range<usize>> {
        let mut seen = Vec::new();
        let last = cancel
            .run(range, chunk, |r| {
                seen.push(r.clone());
                Ok(r)
            })
            .ok()
            .flatten();
        assert_eq!(last, seen.last().cloned(), "run returns the last chunk");
        seen
    }

    #[test]
    fn chunks_cover_the_range_and_zero_runs_it_whole() {
        let never = PrefillCancel::never();
        assert_eq!(ranges(never, 2..9, 3), [2..5, 5..8, 8..9]);
        assert_eq!(ranges(never, 2..9, 0), vec![2..9]);
        assert_eq!(ranges(never, 4..4, 2), Vec::<Range<usize>>::new());
    }

    #[test]
    fn a_cancel_stops_before_the_next_chunk() {
        let reads = AtomicUsize::new(0);
        let probe = || reads.fetch_add(1, Ordering::SeqCst) >= 2;
        let cancel = PrefillCancel::when(&probe);
        let mut ran = Vec::new();
        let out = cancel
            .run(0..10, 2, |r| {
                ran.push(r);
                Ok(())
            })
            .ok();
        assert_eq!(out, Some(None), "a stopped prefill returns no output");
        assert_eq!(ran, [0..2, 2..4], "two of five chunks ran");
        assert_eq!(reads.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn a_cancel_before_an_unchunked_prefill_skips_its_forward() {
        let probe = || true;
        let mut ran = 0;
        let out = PrefillCancel::when(&probe).run(0..10, 0, |_| {
            ran += 1;
            Ok(())
        });
        assert!(matches!(out, Ok(None)));
        assert_eq!(ran, 0);
    }
}
