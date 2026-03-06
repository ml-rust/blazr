//! Batched decode step for continuous batching
//!
//! Extracted from `batch_engine` to keep file sizes within limits.
//! Contains the core logic for batched single-token decoding.

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::scheduler::{ScheduledBatch, SequenceId};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use super::executor::Executor;
use super::request_scheduler::RequestScheduler;
use super::types::{FinishReason, GeneratedToken};

/// Per-sequence data collected before the batched decode forward pass.
pub(super) struct DecodeSeqData {
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
pub(super) async fn process_decode_batch<R: Runtime<DType = DType>>(
    executor: &Executor<R>,
    scheduler: &RequestScheduler,
    paged_cache: &LayeredPagedKvCache<R>,
    decode_seqs: &[SequenceId],
    batch: &ScheduledBatch,
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
        + ModelClient<R>,
{
    let block_size = paged_cache.block_size();
    let device = executor.device();

    // Collect per-sequence data needed for the batched forward pass.
    // Sequences missing any required data are skipped.
    let mut seq_data: Vec<DecodeSeqData> = Vec::with_capacity(decode_seqs.len());

    for &seq_id in decode_seqs {
        let token_history = match scheduler.get_token_history(seq_id) {
            Some(h) => h,
            None => continue,
        };
        let blocks = match batch.block_tables.get(&seq_id) {
            Some(b) => b,
            None => continue,
        };

        // gen_config is looked up later during sampling (not needed for the forward pass)
        // but we still check it exists so we don't run a forward pass for an unserviceable seq.
        if scheduler.get_gen_config(seq_id).is_none() {
            continue;
        }

        let seq_len = token_history.len();
        let last_token = *token_history.last().unwrap_or(&0);

        let token_pos = seq_len - 1;
        let block_idx = token_pos / block_size;
        let block_offset = token_pos % block_size;
        let slot = if block_idx < blocks.len() {
            (blocks[block_idx] as i32) * (block_size as i32) + (block_offset as i32)
        } else {
            -1
        };

        let block_table: Vec<i32> = blocks.iter().map(|&b| b as i32).collect();

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
    let input = Tensor::from_slice(&tokens_flat, &[n, 1], device);

    // slot_mapping: [N]
    let slots_flat: Vec<i32> = seq_data.iter().map(|s| s.slot).collect();
    let slot_mapping = Tensor::from_slice(&slots_flat, &[n], device);

    // block_table: [N, max_num_blocks] — shorter tables padded with 0
    let mut bt_flat: Vec<i32> = vec![0i32; n * max_num_blocks];
    for (i, s) in seq_data.iter().enumerate() {
        let row_start = i * max_num_blocks;
        for (j, &b) in s.block_table.iter().enumerate() {
            bt_flat[row_start + j] = b;
        }
    }
    let block_table_tensor = Tensor::from_slice(&bt_flat, &[n, max_num_blocks], device);

    // Use the maximum seq_len across all sequences for the attention kernel.
    let max_seq_len = seq_data.iter().map(|s| s.seq_len).max().unwrap_or(1);

    // Single batched forward pass — model dispatches paged decode attention per (batch, head).
    let batch_logits = executor
        .model()
        .forward_with_paged_kv_cache(
            &input,
            paged_cache,
            &slot_mapping,
            &block_table_tensor,
            max_seq_len,
            max_seq_len - 1,
        )
        .map_err(|e| anyhow!("batched decode forward failed: {}", e))?;

    // Split logits and sample one token per sequence.
    for (batch_idx, s) in seq_data.iter().enumerate() {
        let seq_id = s.seq_id;

        // Slice logits for this sequence: batch_logits[batch_idx..batch_idx+1, :, :]
        let seq_logits = batch_logits
            .narrow(0, batch_idx, 1)
            .map_err(|e| anyhow!("narrow logits for seq {}: {}", seq_id, e))?;

        let token_history = match scheduler.get_token_history(seq_id) {
            Some(h) => h,
            None => continue,
        };
        let gen_config = match scheduler.get_gen_config(seq_id) {
            Some(gc) => gc,
            None => continue,
        };

        let token_gpu =
            executor.logits_to_token_on_device(&seq_logits, &token_history, &gen_config)?;
        let event = token_gpu
            .record_event()
            .map_err(|e| anyhow!("record event failed for seq {}: {}", seq_id, e))?;
        let token_id = Executor::<R>::read_token_id(&token_gpu, event)?;

        let is_eos = executor.tokenizer().is_eos(token_id);
        let text = if is_eos {
            String::new()
        } else {
            executor.tokenizer().decode(&[token_id]).unwrap_or_default()
        };

        let finished = scheduler.append_token(seq_id, token_id)?;
        let finish = if is_eos {
            Some(FinishReason::Eos)
        } else if finished {
            Some(FinishReason::Length)
        } else {
            None
        };

        if let Some(tx) = scheduler.get_token_sender(seq_id) {
            let token = GeneratedToken {
                token_id,
                text,
                logprob: None,
                top_logprobs: None,
                finish_reason: finish,
            };
            let _ = tx.send(Ok(token)).await;
        }

        scheduler.append_to_history(seq_id, token_id);

        if finish.is_some() {
            scheduler.finish_sequence(seq_id)?;
        }
    }

    Ok(())
}
