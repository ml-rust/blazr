//! Batched decode step for continuous batching
//!
//! Contains the core logic for batched single-token decoding. Each row
//! samples through its own sequence's sampler state and emits through
//! `emit_token`.

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::scheduler::{ScheduledBatch, SequenceId};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, KvSeqLen, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use super::batch_emit::emit_token;
use super::cancel_marks::CancelPhase;
use super::kv_recompute::{block_ids, block_slot, recompute_kv};
use super::request_cancel::end_cancelled;
use super::request_scheduler::RequestScheduler;
use super::seq_sampler::BatchSeqs;
use crate::engine::executor::Executor;
use crate::engine::types::PrefillCancel;

/// Per-sequence data collected before the batched decode forward pass.
pub(in crate::engine) struct DecodeSeqData {
    pub seq_id: SequenceId,
    pub last_token: u32,
    pub slot: i32,
    pub block_table: Vec<i32>,
    pub seq_len: usize,
}

/// Process decode step for multiple sequences (true batched single-token generation).
///
/// Concatenates all decode sequence tokens into one input tensor [N, 1], builds a
/// single padded block table [N, max_num_blocks], and runs ONE forward pass. The
/// output logits [N, 1, vocab_size] are then split per sequence for sampling.
///
/// A sequence `seqs` marks stale first refills its KV blocks from its
/// history, in forwards of at most `chunk_size` tokens. A cancel read
/// before each of those forwards ends the sequence at that chunk.
pub(in crate::engine) async fn process_decode_batch<R: Runtime<DType = DType>>(
    executor: &Executor<R>,
    scheduler: &RequestScheduler,
    paged_cache: &LayeredPagedKvCache<R>,
    seqs: &mut BatchSeqs<R>,
    decode_seqs: &[SequenceId],
    batch: &ScheduledBatch,
    chunk_size: usize,
) -> Result<()>
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
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    let block_size = paged_cache.block_size();
    let device = executor.device();

    // Collect per-sequence data needed for the batched forward pass.
    // Sequences missing any required data are skipped.
    let mut seq_data: Vec<DecodeSeqData> = Vec::with_capacity(decode_seqs.len());

    for &seq_id in decode_seqs {
        // A cancelled row leaves before the forward, so it takes no slot.
        if !scheduler.is_live(seq_id) {
            end_cancelled(scheduler, seqs, seq_id, CancelPhase::Decode)?;
            continue;
        }
        let token_history = match scheduler.get_token_history(seq_id) {
            Some(h) => h,
            None => continue,
        };
        let blocks = match batch.block_tables.get(&seq_id) {
            Some(b) => b,
            None => continue,
        };

        if !seqs.contains(seq_id) {
            return Err(anyhow!(
                "sequence {seq_id} reached batched decode with no sampler state. A sequence \
                 must pass prefill in this engine before it decodes."
            ));
        }

        let seq_len = token_history.len();
        let last_token = *token_history.last().unwrap_or(&0);

        let token_pos = seq_len - 1;
        // A resumed sequence's blocks are new: refill every position before
        // the one this step writes. A cancel stops the refill at its next
        // chunk and frees the sequence.
        if seqs.take_stale(seq_id) {
            let gone = || !scheduler.is_live(seq_id);
            let filled = recompute_kv(
                executor,
                paged_cache,
                &token_history[..token_pos],
                blocks,
                chunk_size,
                PrefillCancel::when(&gone),
            )?;
            if !filled {
                end_cancelled(scheduler, seqs, seq_id, CancelPhase::Decode)?;
                continue;
            }
        }
        let slot = block_slot(blocks, block_size, token_pos);
        let block_table = block_ids(blocks);

        seq_data.push(DecodeSeqData {
            seq_id,
            last_token,
            slot,
            block_table,
            seq_len,
        });
    }

    if seq_data.is_empty() {
        return Ok(());
    }

    let n = seq_data.len();

    // Find max block table length for padding.
    let max_num_blocks = seq_data
        .iter()
        .map(|s| s.block_table.len())
        .max()
        .unwrap_or(0);

    // Build batched tensors.
    // input: [N, 1] i64
    let tokens_flat: Vec<i64> = seq_data.iter().map(|s| s.last_token as i64).collect();
    let input = Tensor::from_slice(&tokens_flat, &[n, 1], device)?;

    // slot_mapping: [N]
    let slots_flat: Vec<i32> = seq_data.iter().map(|s| s.slot).collect();
    let slot_mapping = Tensor::from_slice(&slots_flat, &[n], device)?;

    // block_table: [N, max_num_blocks] — shorter tables padded with 0
    let mut bt_flat: Vec<i32> = vec![0i32; n * max_num_blocks];
    for (i, s) in seq_data.iter().enumerate() {
        let row_start = i * max_num_blocks;
        for (j, &b) in s.block_table.iter().enumerate() {
            bt_flat[row_start + j] = b;
        }
    }
    let block_table_tensor = Tensor::from_slice(&bt_flat, &[n, max_num_blocks], device)?;

    // max_seq_len only sizes the gathered KV buffer and the KV-write
    // position; it must not bound any row's own attention range, since a
    // shorter row's real keys end before it.
    let max_seq_len = seq_data.iter().map(|s| s.seq_len).max().unwrap_or(1);

    // Each row's own key end (its seq_len), so a shorter row does not
    // attend past its real cached keys into another row's blocks.
    let ends_flat: Vec<i32> = seq_data.iter().map(|s| s.seq_len as i32).collect();
    let kv_ends = Tensor::from_slice(&ends_flat, &[n], device)?;

    // Each row's own RoPE position (its last token's index), so a shorter
    // row's query rotates at its real position instead of the batch max
    // `max_seq_len - 1`.
    let positions_flat: Vec<i32> = seq_data.iter().map(|s| (s.seq_len - 1) as i32).collect();
    let rope_positions = Tensor::from_slice(&positions_flat, &[n], device)?;

    // Single batched forward pass — model dispatches paged decode attention per (batch, head).
    seqs.count_forward();
    let batch_logits = executor
        .model()?
        .forward_with_paged_kv_cache(
            &input,
            paged_cache,
            &slot_mapping,
            &block_table_tensor,
            max_seq_len,
            max_seq_len - 1,
            Some(KvSeqLen::PerRow(&kv_ends)),
            Some(&rope_positions),
        )
        .map_err(|e| anyhow!("batched decode forward failed: {}", e))?;

    // Split logits and sample one token per sequence.
    for (batch_idx, s) in seq_data.iter().enumerate() {
        let seq_id = s.seq_id;

        // Slice logits for this sequence: batch_logits[batch_idx..batch_idx+1, :, :]
        let seq_logits = batch_logits
            .narrow(0, batch_idx, 1)
            .map_err(|e| anyhow!("narrow logits for seq {}: {}", seq_id, e))?;

        // A request cancelled during the forward ends here, with no token.
        emit_token(
            executor,
            scheduler,
            seqs,
            seq_id,
            &seq_logits,
            CancelPhase::Decode,
        )
        .await?;
    }

    Ok(())
}
