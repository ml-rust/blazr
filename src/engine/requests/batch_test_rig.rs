//! The batching engine over a tiny CPU llama, shared by the engine's and
//! the server's unit tests, with the single-sequence runs they compare against.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use futures::StreamExt;

use boostr::inference::memory::CpuBlockAllocator;
use boostr::inference::scheduler::SchedulerConfig;
use boostr::CpuRuntime;

use super::batch_engine::BatchEngine;
use super::request_registry::RequestHandle;
use super::request_scheduler::RequestScheduler;
use crate::config::GenerationConfig;
use crate::engine::executor::Executor;
use crate::engine::test_support::{seeded_executor, PagedPool};
use crate::engine::types::{FinishReason, GeneratedToken};

/// Seed of the rig's tiny llama weights.
pub(crate) const SEED: u64 = 0x5eed;
/// Ticks a test allows before it calls the engine stuck.
const TICK_LIMIT: usize = 64;
/// A pool large enough that no sequence is preempted.
pub(crate) const POOL: PagedPool = PagedPool {
    block_size: 4,
    blocks: 16,
};
/// The context of [`rig`]'s tiny llama.
pub(crate) const TINY_CTX: usize = 16;

pub(crate) struct Rig {
    pub executor: Arc<Executor<CpuRuntime>>,
    pub scheduler: Arc<RequestScheduler>,
    pub engine: BatchEngine<CpuRuntime>,
}

/// A batching engine over the seeded tiny llama, admitting two sequences.
pub(crate) fn rig(pool: PagedPool) -> Result<Rig> {
    rig_on(seeded_executor(SEED, Some(pool))?, pool, TINY_CTX)
}

/// A batching engine over `executor`, admitting two sequences of up to
/// `max_seq_len` tokens.
pub(crate) fn rig_on(
    executor: Executor<CpuRuntime>,
    pool: PagedPool,
    max_seq_len: usize,
) -> Result<Rig> {
    let executor = Arc::new(executor);
    let config = SchedulerConfig {
        max_batch_size: 2,
        max_batch_tokens: 4096,
        max_seq_len,
        block_size: pool.block_size,
        ..Default::default()
    };
    let allocator = CpuBlockAllocator::new(pool.blocks, pool.block_size);
    let scheduler = RequestScheduler::new(allocator, config);
    let engine =
        BatchEngine::new(Arc::clone(&executor), Arc::clone(&scheduler))?.with_prefill_chunk_size(0);
    Ok(Rig {
        executor,
        scheduler,
        engine,
    })
}

/// Greedy decoding with no repetition penalty.
pub(crate) fn greedy(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        repeat_penalty: 1.0,
        ..GenerationConfig::greedy()
    }
}

/// The tokens `prompt` produces alone through the single-sequence loop.
pub(crate) async fn alone_tokens(
    executor: &Executor<CpuRuntime>,
    prompt: &[u32],
    cfg: &GenerationConfig,
) -> Result<Vec<GeneratedToken>> {
    let mut stream = std::pin::pin!(executor.generate(prompt, cfg));
    let mut tokens = Vec::new();
    while let Some(item) = stream.next().await {
        tokens.push(item?);
    }
    Ok(tokens)
}

/// The ids `prompt` produces alone through the single-sequence loop.
pub(super) async fn alone(
    executor: &Executor<CpuRuntime>,
    prompt: &[u32],
    cfg: &GenerationConfig,
) -> Result<Vec<u32>> {
    let tokens = alone_tokens(executor, prompt, cfg).await?;
    Ok(tokens.into_iter().map(|t| t.token_id).collect())
}

/// Run one engine step.
pub(crate) async fn tick(rig: &mut Rig) {
    rig.engine.tick().await;
}

/// Tick until the scheduler holds no work.
pub(crate) async fn drain(rig: &mut Rig) -> Result<()> {
    for _ in 0..TICK_LIMIT {
        if !rig.scheduler.has_work() {
            return Ok(());
        }
        rig.engine.tick().await;
    }
    Err(anyhow!(
        "the batch engine still had work after {TICK_LIMIT} ticks"
    ))
}

/// Every item `handle` received so far.
pub(super) fn received(handle: &mut RequestHandle) -> Vec<Result<GeneratedToken>> {
    let mut items = Vec::new();
    while let Ok(item) = handle.token_rx.try_recv() {
        items.push(item);
    }
    items
}

/// The tokens `handle` received so far. A streamed error fails the call.
pub(super) fn tokens(handle: &mut RequestHandle) -> Result<Vec<GeneratedToken>> {
    received(handle).into_iter().collect()
}

/// The ids `handle` received so far. A streamed error fails the call.
pub(crate) fn ids(handle: &mut RequestHandle) -> Result<Vec<u32>> {
    Ok(tokens(handle)?.into_iter().map(|t| t.token_id).collect())
}

/// The response text, finish reason and token count `tokens` add up to.
///
/// A response with no finish token reads as `Length`, as the HTTP
/// collectors read it.
pub(super) fn outcome(tokens: &[GeneratedToken]) -> (String, FinishReason, usize) {
    let text = tokens.iter().map(|t| t.text.as_str()).collect();
    let finish = tokens
        .last()
        .and_then(|t| t.finish_reason)
        .unwrap_or(FinishReason::Length);
    (text, finish, tokens.len())
}

/// Free blocks in the rig's scheduler pool.
pub(crate) fn free_blocks(rig: &Rig) -> Result<usize> {
    rig.scheduler
        .block_stats()
        .map(|stats| stats.free_blocks)
        .ok_or_else(|| anyhow!("the scheduler lock is poisoned"))
}
