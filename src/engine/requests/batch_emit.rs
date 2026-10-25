//! One batched sequence's token, from its logits to its caller.
//!
//! The prefill and the batched decode step both emit through
//! [`emit_token`], so the finish reason, the logprobs and the stop sequences
//! apply as the single-sequence loop applies them.

use anyhow::Result;

use boostr::inference::scheduler::SequenceId;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, GrammarDfaOps, NormalizationOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use super::cancel_marks::CancelPhase;
use super::request_cancel::end_cancelled;
use super::request_scheduler::RequestScheduler;
use super::seq_sampler::BatchSeqs;
use crate::engine::executor::Executor;
use crate::engine::types::{FinishReason, GeneratedToken};

/// Sample `seq_id`'s next token from `logits`, send it, and end the sequence
/// when the token finishes it.
///
/// The finish reason is `Eos` or `Length` as [`Executor::decode_finish_reason`]
/// gives it, or `Stop` when the token completes a stop sequence. A sequence
/// that finishes frees its sampler state, its KV blocks and its request at
/// once, so no later step decodes for it. Returns whether it ended.
///
/// A cancelled sequence ends before it samples. A send to a closed channel
/// ends the sequence as a cancel in `phase`. Either way nothing decodes
/// for it again.
///
/// # Errors
///
/// An error when the sample, the logprobs or the scheduler update fails.
pub(super) async fn emit_token<R: Runtime<DType = DType>>(
    executor: &Executor<R>,
    scheduler: &RequestScheduler,
    seqs: &mut BatchSeqs<R>,
    seq_id: SequenceId,
    logits: &Tensor<R>,
    phase: CancelPhase,
) -> Result<bool>
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
    if !scheduler.is_live(seq_id) {
        end_cancelled(scheduler, seqs, seq_id, phase)?;
        return Ok(true);
    }
    let sampled = seqs.sample(executor, seq_id, logits)?;

    // The scheduler also ends a sequence that fills its context. The
    // admission budget makes that the `max_tokens` step, so it reads `Length`.
    let full = scheduler.append_token(seq_id, sampled.id)?;
    let finish = sampled.finish.or(full.then_some(FinishReason::Length));
    let cut = scheduler.token_text(seq_id, executor.tokenizer(), sampled.id, finish);

    let token = GeneratedToken {
        token_id: sampled.id,
        text: cut.text,
        logprob: sampled.logprob,
        top_logprobs: sampled.top_logprobs,
        finish_reason: cut.finish,
    };
    let delivered = match scheduler.get_token_sender(seq_id) {
        Some(tx) => tx.send(Ok(token)).await.is_ok(),
        None => false,
    };
    if !delivered {
        // The caller stopped listening: no later step decodes for it.
        end_cancelled(scheduler, seqs, seq_id, phase)?;
        return Ok(true);
    }
    scheduler.append_to_history(seq_id, sampled.id);

    let ended = cut.finish.is_some();
    if ended {
        seqs.end_seq(seq_id);
        scheduler.finish_sequence(seq_id)?;
    }
    Ok(ended)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use anyhow::anyhow;

    use super::super::batch_test_rig::{
        alone, alone_tokens, drain, free_blocks, greedy, ids, outcome, rig, rig_on, tokens, Rig,
        POOL,
    };
    use crate::config::GenerationConfig;
    use crate::engine::test_support::{full_vocab_executor, piece_id, FULL_VOCAB_CTX};
    use crate::engine::types::TokenLogprob;

    /// Logprobs of the two paths agree to this much.
    const LOGPROB_TOL: f32 = 1e-5;

    fn close(got: Option<f32>, want: Option<f32>) -> bool {
        matches!((got, want), (Some(g), Some(w)) if (g - w).abs() <= LOGPROB_TOL)
    }

    fn same_top(got: &[TokenLogprob], want: &[TokenLogprob]) -> bool {
        got.len() == want.len()
            && got.iter().zip(want).all(|(g, w)| {
                g.token_id == w.token_id
                    && g.text == w.text
                    && close(Some(g.logprob), Some(w.logprob))
            })
    }

    fn with_stops(stops: &[&str], cfg: GenerationConfig) -> GenerationConfig {
        GenerationConfig {
            stop_sequences: stops.iter().map(|s| s.to_string()).collect(),
            ..cfg
        }
    }

    /// The text, finish reason and token count of the single-sequence
    /// non-streaming path, which applies the stops after the decode loop.
    async fn alone_outcome(
        rig: &Rig,
        prompt: &[u32],
        cfg: &GenerationConfig,
    ) -> Result<(String, FinishReason, usize)> {
        let result = rig.executor.generate_text(prompt, cfg).await?;
        Ok((result.text, result.finish_reason, result.completion_tokens))
    }

    /// The text the seeded llama produces alone for `prompt`. Every id under
    /// 32 is one ASCII character, so a byte range is a token range.
    async fn alone_text(rig: &Rig, prompt: &[u32], cfg: &GenerationConfig) -> Result<String> {
        let tokens = alone_tokens(&rig.executor, prompt, cfg).await?;
        let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
        if text.len() != tokens.len() || !text.is_ascii() {
            return Err(anyhow!("{text:?} is not one ASCII character per token"));
        }
        Ok(text)
    }

    #[tokio::test]
    async fn logprobs_match_the_unbatched_run_and_only_the_asking_row_gets_them() -> Result<()> {
        let mut rig = rig(POOL)?;
        let (asking, plain) = ([1, 2, 3], [7, 8, 9, 10]);
        let asking_cfg = GenerationConfig {
            logprobs: true,
            top_logprobs: 3,
            ..greedy(4)
        };
        let expected = alone_tokens(&rig.executor, &asking, &asking_cfg).await?;

        let mut a = rig.scheduler.submit(asking.to_vec(), asking_cfg)?;
        let mut b = rig.scheduler.submit(plain.to_vec(), greedy(4))?;
        rig.engine.tick().await;
        assert_eq!(rig.engine.seqs.len(), 2, "both rows share one batch");
        drain(&mut rig).await?;

        let got = tokens(&mut a)?;
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(g.token_id, e.token_id);
            assert!(
                close(g.logprob, e.logprob),
                "{:?} vs {:?}",
                g.logprob,
                e.logprob
            );
            let (g_top, e_top) = (g.top_logprobs.as_deref(), e.top_logprobs.as_deref());
            assert!(
                matches!((g_top, e_top), (Some(gt), Some(et)) if et.len() == 3 && same_top(gt, et)),
                "{g_top:?} vs {e_top:?}"
            );
        }
        let plain_got = tokens(&mut b)?;
        assert_eq!(plain_got.len(), 4);
        assert!(plain_got
            .iter()
            .all(|t| t.logprob.is_none() && t.top_logprobs.is_none()));
        Ok(())
    }

    #[tokio::test]
    async fn max_tokens_one_finishes_with_length_as_unbatched() -> Result<()> {
        let mut rig = rig(POOL)?;
        let prompt = [1, 2, 3];
        let cfg = greedy(1);
        let expected = alone_tokens(&rig.executor, &prompt, &cfg).await?;
        assert_eq!(
            expected.last().and_then(|t| t.finish_reason),
            Some(FinishReason::Length)
        );

        let mut handle = rig.scheduler.submit(prompt.to_vec(), cfg)?;
        drain(&mut rig).await?;
        let got = tokens(&mut handle)?;

        assert_eq!(got.len(), 1);
        assert_eq!(got[0].token_id, expected[0].token_id);
        assert_eq!(got[0].finish_reason, Some(FinishReason::Length));
        assert_eq!(rig.engine.seqs.len(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn eos_on_the_first_token_finishes_with_eos_as_unbatched() -> Result<()> {
        let mut rig = rig_on(full_vocab_executor(Some(POOL))?, POOL, FULL_VOCAB_CTX)?;
        let eos = rig
            .executor
            .tokenizer()
            .eos_token_id()
            .ok_or_else(|| anyhow!("llama3 states no end-of-sequence token"))?;
        let cfg = GenerationConfig {
            logit_bias: HashMap::from([(eos, 30.0)]),
            ..greedy(8)
        };
        let expected = alone_tokens(&rig.executor, &[1], &cfg).await?;

        let mut handle = rig.scheduler.submit(vec![1], cfg)?;
        drain(&mut rig).await?;
        let got = tokens(&mut handle)?;

        assert_eq!(outcome(&got), outcome(&expected));
        assert_eq!(outcome(&got).1, FinishReason::Eos);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].token_id, eos);
        Ok(())
    }

    /// Two rows in one batch, each with a stop that spans two tokens.
    #[tokio::test]
    async fn rows_with_different_split_stops_match_their_unbatched_text() -> Result<()> {
        let mut rig = rig(POOL)?;
        let start = free_blocks(&rig)?;
        let (first, second) = ([1, 2, 3], [7, 8, 9, 10]);
        let first_text = alone_text(&rig, &first, &greedy(6)).await?;
        let second_text = alone_text(&rig, &second, &greedy(6)).await?;
        let first_cfg = with_stops(&[&first_text[1..3]], greedy(6));
        let second_cfg = with_stops(&[&second_text[2..4]], greedy(6));
        let first_alone = alone_outcome(&rig, &first, &first_cfg).await?;
        let second_alone = alone_outcome(&rig, &second, &second_cfg).await?;
        assert_eq!(first_alone.1, FinishReason::Stop);
        assert_eq!(second_alone.1, FinishReason::Stop);

        let mut a = rig.scheduler.submit(first.to_vec(), first_cfg)?;
        let mut b = rig.scheduler.submit(second.to_vec(), second_cfg)?;
        drain(&mut rig).await?;

        assert_eq!(outcome(&tokens(&mut a)?), first_alone);
        assert_eq!(outcome(&tokens(&mut b)?), second_alone);
        assert!(!rig.scheduler.has_work());
        assert_eq!(rig.engine.seqs.len(), 0);
        assert_eq!(
            free_blocks(&rig)?,
            start,
            "a stopped row must free its blocks"
        );
        Ok(())
    }

    /// Every token is the piece `the`. The stop `h` sits inside the first
    /// token, and the stop `et` spans the first two.
    #[tokio::test]
    async fn stops_inside_a_token_and_at_the_first_token_match_the_unbatched_text() -> Result<()> {
        let mut rig = rig_on(full_vocab_executor(Some(POOL))?, POOL, FULL_VOCAB_CTX)?;
        let start = free_blocks(&rig)?;
        let the = piece_id(&rig.executor, b"the")?;
        let base = GenerationConfig {
            logit_bias: HashMap::from([(the, 30.0)]),
            ..greedy(4)
        };
        let inside = with_stops(&["h"], base.clone());
        let across = with_stops(&["et"], base);
        let inside_alone = alone_outcome(&rig, &[1], &inside).await?;
        let across_alone = alone_outcome(&rig, &[1], &across).await?;
        assert_eq!(inside_alone, ("t".to_string(), FinishReason::Stop, 1));
        assert_eq!(across_alone, ("th".to_string(), FinishReason::Stop, 2));

        let mut a = rig.scheduler.submit(vec![1], inside)?;
        let mut b = rig.scheduler.submit(vec![1], across)?;
        drain(&mut rig).await?;

        assert_eq!(outcome(&tokens(&mut a)?), inside_alone);
        assert_eq!(outcome(&tokens(&mut b)?), across_alone);
        assert!(!rig.scheduler.has_work());
        assert_eq!(free_blocks(&rig)?, start);
        Ok(())
    }

    #[tokio::test]
    async fn a_stopped_row_frees_its_blocks_while_the_other_decodes_on() -> Result<()> {
        let mut rig = rig(POOL)?;
        let start = free_blocks(&rig)?;
        let (stopping, running) = ([1, 2, 3], [7, 8, 9, 10]);
        let stopping_text = alone_text(&rig, &stopping, &greedy(6)).await?;
        // The stop is the first two characters, so it completes at the second token.
        let stopping_cfg = with_stops(&[&stopping_text[..2]], greedy(6));
        let stopping_alone = alone_outcome(&rig, &stopping, &stopping_cfg).await?;
        assert_eq!(stopping_alone.1, FinishReason::Stop);
        assert_eq!(stopping_alone.2, 2);
        let running_alone = alone(&rig.executor, &running, &greedy(8)).await?;

        let mut a = rig.scheduler.submit(stopping.to_vec(), stopping_cfg)?;
        let mut b = rig.scheduler.submit(running.to_vec(), greedy(8))?;
        for _ in 0..8 {
            if !rig.scheduler.is_live(a.seq_id) {
                break;
            }
            rig.engine.tick().await;
        }
        assert!(!rig.scheduler.is_live(a.seq_id), "the stop ends the row");
        assert!(rig.scheduler.is_live(b.seq_id), "the other row decodes on");
        assert_eq!(rig.engine.seqs.len(), 1);
        let held = rig
            .scheduler
            .get_block_table(b.seq_id)
            .map_or(0, |blocks| blocks.len());
        assert_eq!(
            free_blocks(&rig)?,
            start - held,
            "the stopped row's blocks are free"
        );

        drain(&mut rig).await?;
        assert_eq!(outcome(&tokens(&mut a)?), stopping_alone);
        assert_eq!(ids(&mut b)?, running_alone);
        assert_eq!(free_blocks(&rig)?, start);
        Ok(())
    }

    /// The tiny llama holds 16 tokens, so a 12-token prompt decodes 4.
    #[tokio::test]
    async fn max_tokens_is_clamped_to_the_context_as_unbatched() -> Result<()> {
        let mut rig = rig(POOL)?;
        let prompt: Vec<u32> = (1..=12).collect();
        let cfg = greedy(100);
        let expected = alone_tokens(&rig.executor, &prompt, &cfg).await?;
        assert_eq!(expected.len(), 4);

        let mut handle = rig.scheduler.submit(prompt, cfg)?;
        drain(&mut rig).await?;
        let got = tokens(&mut handle)?;

        let ids_of = |t: &[GeneratedToken]| t.iter().map(|t| t.token_id).collect::<Vec<_>>();
        assert_eq!(ids_of(&got), ids_of(&expected));
        assert_eq!(outcome(&got), outcome(&expected));
        assert_eq!(outcome(&got).1, FinishReason::Length);
        Ok(())
    }
}
