//! The seeded paged executor and the helpers the [`PagedTarget`] tests share.
//!
//! [`PagedTarget`]: super::PagedTarget

use anyhow::{anyhow, Result};
use boostr::inference::BlockAllocator;
use boostr::CpuRuntime;

use crate::config::GenerationConfig;
use crate::engine::executor::Executor;
use crate::engine::speculative::target_glue::LastLogits;
use crate::engine::test_support::{seeded_paged, PagedPool};

pub(super) const POOL: PagedPool = PagedPool {
    block_size: 4,
    blocks: 16,
};
const CTX: usize = 64;
const SEED: u64 = 0x5bec;
const MAX_TOKENS: usize = 40;

pub(super) type Exec = Executor<CpuRuntime>;

pub(super) fn executor(prefix_cache: bool) -> Result<Exec> {
    seeded_paged(SEED, POOL, CTX, prefix_cache)
}

pub(super) fn free_blocks(exec: &Exec) -> Result<usize> {
    let shared = exec
        .shared_allocator()
        .ok_or_else(|| anyhow!("the executor has no shared allocator"))?;
    let free = shared
        .lock()
        .map_err(|e| anyhow!("lock poisoned: {e}"))?
        .free_block_count();
    Ok(free)
}

pub(super) fn greedy_plain() -> GenerationConfig {
    GenerationConfig {
        max_tokens: MAX_TOKENS,
        repeat_penalty: 1.0,
        ..GenerationConfig::greedy()
    }
}

pub(super) fn rows<M: LastLogits<CpuRuntime>>(target: &M) -> Result<Vec<f32>> {
    Ok(target
        .last_logits()
        .ok_or_else(|| anyhow!("no logits after an extend"))?
        .to_vec::<f32>())
}

pub(super) fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
