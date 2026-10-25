//! Paged-cache slot mapping and the KV recompute of a resumed sequence.
//!
//! Preemption frees a sequence's KV blocks. On resume the scheduler hands it
//! new blocks that hold no keys or values. Before the sequence decodes again,
//! [`recompute_kv`] runs its context through the model to fill them.

use std::ops::Range;

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::BlockId;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::engine::executor::Executor;

/// The paged-cache slot of `position`, or -1 past the end of `blocks`, which
/// the kernels skip.
pub(in crate::engine) fn block_slot(blocks: &[BlockId], block_size: usize, position: usize) -> i32 {
    match blocks.get(position / block_size) {
        Some(&block) => (block as i32) * (block_size as i32) + ((position % block_size) as i32),
        None => -1,
    }
}

/// The paged-cache slot of each position in `positions`, as [`block_slot`].
pub(in crate::engine) fn block_slots(
    blocks: &[BlockId],
    block_size: usize,
    positions: Range<usize>,
) -> Vec<i32> {
    positions
        .map(|i| block_slot(blocks, block_size, i))
        .collect()
}

/// A block table as the `i32` ids the kernels read.
pub(in crate::engine) fn block_ids(blocks: &[BlockId]) -> Vec<i32> {
    blocks.iter().map(|&b| b as i32).collect()
}

/// Fill `blocks` with the keys and values of `tokens` at positions
/// `0..tokens.len()`.
///
/// `chunk` caps the tokens one forward takes. A `chunk` of 0 runs every token
/// in one forward. The logits are dropped.
///
/// # Errors
///
/// An error when a forward fails.
pub(in crate::engine) fn recompute_kv<R: Runtime<DType = DType>>(
    executor: &Executor<R>,
    paged_cache: &LayeredPagedKvCache<R>,
    tokens: &[u32],
    blocks: &[BlockId],
    chunk: usize,
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
    let device = executor.device();
    let block_size = paged_cache.block_size();
    let step = if chunk == 0 { tokens.len() } else { chunk };
    let bt_vec = block_ids(blocks);
    let block_table = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], device)?;

    let mut start = 0;
    while start < tokens.len() {
        let end = (start + step).min(tokens.len());
        let slots = block_slots(blocks, block_size, start..end);
        let slot_tensor = Tensor::from_slice(&slots, &[end - start], device)?;
        let input = executor.create_input_tensor(&tokens[start..end])?;
        executor
            .model()?
            .forward_with_paged_kv_cache(
                &input,
                paged_cache,
                &slot_tensor,
                &block_table,
                end,
                start,
                None,
                None,
            )
            .map_err(|e| {
                anyhow!(
                    "KV recompute of a resumed sequence failed at positions {start}..{end} of \
                     {}: {e}",
                    tokens.len()
                )
            })?;
        start = end;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::CpuRuntime;

    use crate::engine::test_support::seeded_executor;

    const BLOCK_SIZE: usize = 4;

    #[test]
    fn slots_follow_the_block_table_and_run_off_to_minus_one() {
        assert_eq!(block_slots(&[3, 1], 4, 2..6), vec![14, 15, 4, 5]);
        assert_eq!(block_slots(&[3], 4, 3..5), vec![15, -1]);
    }

    /// The logits of `token` decoded at `pos` over the KV in `blocks`.
    fn decode_logits(
        exec: &Executor<CpuRuntime>,
        cache: &LayeredPagedKvCache<CpuRuntime>,
        blocks: &[BlockId],
        token: u32,
        pos: usize,
    ) -> Result<Vec<f32>> {
        let slots = block_slots(blocks, BLOCK_SIZE, pos..pos + 1);
        let slot_tensor = Tensor::from_slice(&slots, &[1], exec.device())?;
        let bt: Vec<i32> = blocks.iter().map(|&b| b as i32).collect();
        let block_table = Tensor::from_slice(&bt, &[1, bt.len()], exec.device())?;
        let input = exec.create_input_tensor(&[token])?;
        let logits = exec.model()?.forward_with_paged_kv_cache(
            &input,
            cache,
            &slot_tensor,
            &block_table,
            pos + 1,
            pos,
            None,
            None,
        )?;
        Ok(logits.to_vec())
    }

    #[test]
    fn recomputed_blocks_decode_like_the_original_prefill() -> Result<()> {
        let exec = seeded_executor(3, None)?;
        let model = exec.model()?;
        let cache = LayeredPagedKvCache::<CpuRuntime>::new(
            model.num_layers(),
            8,
            BLOCK_SIZE,
            model.num_kv_heads().unwrap_or(1),
            model.head_dim().unwrap_or(1),
            DType::F32,
            exec.device(),
        )?;
        let context = [1, 2, 3, 4, 5];
        let next = 6;

        // The original prefill writes blocks 0 and 1. Chunks of 2 refill 4 and 5.
        recompute_kv(&exec, &cache, &context, &[0, 1], 0)?;
        recompute_kv(&exec, &cache, &context, &[4, 5], 2)?;

        let original = decode_logits(&exec, &cache, &[0, 1], next, context.len())?;
        let refilled = decode_logits(&exec, &cache, &[4, 5], next, context.len())?;
        let empty = decode_logits(&exec, &cache, &[2, 3], next, context.len())?;

        let max_diff = original
            .iter()
            .zip(&refilled)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "refilled blocks differ by {max_diff}");
        assert_ne!(original, empty, "decoding over empty blocks must differ");
        Ok(())
    }
}
