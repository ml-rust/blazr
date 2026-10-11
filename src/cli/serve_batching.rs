//! Picks the scheduler `blazr serve` runs behind the HTTP handlers.
//!
//! Two kinds exist, and a model takes at most one:
//!
//! - Models that need GDN state decode in `qwen35` lanes: [`LaneScheduler`]
//!   plus the [`LaneEngine`] task. Paged attention plays no part here.
//! - Every other model with paged attention on decodes in the paged
//!   continuous batch: [`RequestScheduler`] plus the [`BatchEngine`] task.
//!
//! With `inference.max_batch_size` at 1 neither is built, and the handlers
//! take the single-sequence path.
//!
//! With `server.slo` P95 targets, both schedulers stop growing the batch
//! while a target is breached.

use std::sync::Arc;

use anyhow::Result;
use colored::Colorize;

use crate::engine::{BatchEngine, Executor, GenerationScheduler, RequestScheduler, SloBatchPolicy};

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

/// Which scheduler a model's decode path takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchingChoice {
    /// Handlers generate one sequence at a time, no scheduler built.
    Single,
    /// `LaneScheduler` plus the `LaneEngine` task (GDN-state models).
    Lanes,
    /// `RequestScheduler` plus the `BatchEngine` task (paged continuous batch).
    Paged,
}

/// Decides the batching path from the facts `configure_batching` reads off
/// the executor: does the model need GDN state, what is
/// `inference.max_batch_size`, and is paged attention on.
pub fn batching_choice(
    needs_gdn_state: bool,
    max_batch_size: usize,
    paged: bool,
) -> BatchingChoice {
    if max_batch_size <= 1 {
        return BatchingChoice::Single;
    }
    if needs_gdn_state {
        return BatchingChoice::Lanes;
    }
    if !paged {
        return BatchingChoice::Single;
    }
    BatchingChoice::Paged
}

/// Build the scheduler `executor`'s model calls for, spawn the task that
/// drives it, and return it for the server state.
///
/// `Ok(None)` means the handlers generate one sequence at a time. `device`
/// names the executor's device in the startup line.
///
/// # Errors
///
/// An error when the chosen engine cannot be built for this model.
pub async fn configure_batching(
    executor: Arc<Executor<ServerRuntime>>,
    slo_policy: Option<Arc<SloBatchPolicy>>,
    device: &str,
) -> Result<Option<Arc<dyn GenerationScheduler>>> {
    let max_batch = executor.config().inference.max_batch_size;
    let choice = batching_choice(
        executor.model().needs_gdn_state(),
        max_batch,
        executor.config().inference.paged_attention,
    );

    match choice {
        BatchingChoice::Single => Ok(None),
        BatchingChoice::Lanes => lane_batching(executor, max_batch, slo_policy, device).await,
        BatchingChoice::Paged => paged_batching(executor, max_batch, slo_policy, device).map(Some),
    }
}

/// `max_batch` `qwen35` decode lanes, driven by a [`LaneEngine`] task.
#[cfg(feature = "cuda")]
async fn lane_batching(
    executor: Arc<Executor<ServerRuntime>>,
    max_batch: usize,
    slo_policy: Option<Arc<SloBatchPolicy>>,
    device: &str,
) -> Result<Option<Arc<dyn GenerationScheduler>>> {
    use crate::engine::{LaneEngine, LaneScheduler};

    let lane_scheduler = LaneScheduler::with_slo_policy(max_batch, slo_policy)?;
    let engine = LaneEngine::open(executor, Arc::clone(&lane_scheduler)).await?;
    let mode = engine.mode();
    tokio::spawn(engine.run());

    eprintln!(
        "  Batching: {} (lanes={}, decode={:?}, device={})",
        "lanes".green(),
        max_batch,
        mode,
        device,
    );

    let configured: Arc<dyn GenerationScheduler> = lane_scheduler;
    Ok(Some(configured))
}

/// Lane decode is a CUDA path, so a CPU build serves these models one
/// sequence at a time whatever `max_batch_size` says.
#[cfg(not(feature = "cuda"))]
async fn lane_batching(
    executor: Arc<Executor<ServerRuntime>>,
    max_batch: usize,
    _slo_policy: Option<Arc<SloBatchPolicy>>,
    _device: &str,
) -> Result<Option<Arc<dyn GenerationScheduler>>> {
    tracing::warn!(
        "model {} decodes in lanes, which needs a CUDA build; inference.max_batch_size = {} is \
         ignored and requests run one at a time. Rebuild with --features cuda to batch them.",
        executor.model().model_type(),
        max_batch,
    );
    Ok(None)
}

/// A paged continuous batch, driven by a [`BatchEngine`] task.
fn paged_batching(
    executor: Arc<Executor<ServerRuntime>>,
    max_batch: usize,
    slo_policy: Option<Arc<SloBatchPolicy>>,
    device: &str,
) -> Result<Arc<dyn GenerationScheduler>> {
    let block_size = executor.config().inference.block_size;
    let kv_pool = executor.config().inference.kv_pool_blocks;
    let max_ctx = executor.config().inference.max_context_len.unwrap_or(4096);

    let total_blocks = if kv_pool > 0 {
        kv_pool
    } else {
        (max_ctx * max_batch / block_size).max(256)
    };

    let allocator = boostr::inference::memory::CpuBlockAllocator::new(total_blocks, block_size);
    let sched_config = boostr::inference::scheduler::SchedulerConfig {
        max_batch_size: max_batch,
        max_batch_tokens: max_ctx,
        block_size,
        ..Default::default()
    };

    // Create prefix cache if enabled
    let prefix_cache = executor.config().inference.prefix_cache.then(|| {
        let cache_config = boostr::inference::prefix_cache::PrefixCacheConfig {
            enabled: true,
            block_size,
            max_cached_blocks: executor
                .config()
                .inference
                .max_cached_blocks
                .max(total_blocks / 4),
            ..Default::default()
        };
        boostr::inference::prefix_cache::PrefixCache::new(allocator.clone(), cache_config)
    });
    let request_scheduler =
        RequestScheduler::build(allocator, sched_config, prefix_cache, slo_policy);

    let batch_engine = BatchEngine::new(executor, Arc::clone(&request_scheduler))?;
    tokio::spawn(async move {
        batch_engine.run().await;
    });

    eprintln!(
        "  Batching: {} (max_batch={}, blocks={}, device={})",
        "continuous".green(),
        max_batch,
        total_blocks,
        device,
    );

    let configured: Arc<dyn GenerationScheduler> = request_scheduler;
    Ok(configured)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_batch_zero_gives_single_regardless_of_gdn_or_paged() {
        assert_eq!(batching_choice(false, 0, false), BatchingChoice::Single);
        assert_eq!(batching_choice(true, 0, true), BatchingChoice::Single);
    }

    #[test]
    fn max_batch_one_gives_single_regardless_of_gdn_or_paged() {
        assert_eq!(batching_choice(false, 1, false), BatchingChoice::Single);
        assert_eq!(batching_choice(true, 1, true), BatchingChoice::Single);
    }

    #[test]
    fn gdn_model_gives_lanes_whether_paged_is_true_or_false() {
        assert_eq!(batching_choice(true, 4, true), BatchingChoice::Lanes);
        assert_eq!(batching_choice(true, 4, false), BatchingChoice::Lanes);
    }

    #[test]
    fn non_gdn_model_with_paged_off_gives_single() {
        assert_eq!(batching_choice(false, 4, false), BatchingChoice::Single);
    }

    #[test]
    fn non_gdn_model_with_paged_on_gives_paged() {
        assert_eq!(batching_choice(false, 4, true), BatchingChoice::Paged);
    }
}
