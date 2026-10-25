//! The executor's paged KV pool and the block lease of one paged request.
//!
//! The shared block allocator hands out ids of the one pool cache. A request
//! holds its ids in a [`PoolLease`], which returns them when it drops. A
//! stream that ends early, by a stop sequence, an error or a dropped client,
//! returns them the same way.

use std::sync::Mutex;

use anyhow::{anyhow, Result};

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::{BlockAllocator, BlockId, CpuBlockAllocator};
use boostr::inference::prefix_cache::PrefixCache;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};

use super::executor::Executor;

/// The prefix cache of an executor, as the lease reads it.
type SharedPrefixCache = Option<Mutex<PrefixCache<CpuBlockAllocator>>>;

impl<R: Runtime<DType = DType>> Executor<R>
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
        + boostr::quant::traits::DequantOps<R>,
{
    /// The paged KV cache that holds every block of the shared allocator.
    ///
    /// The first call creates it with exactly the allocator's block count, so
    /// every id the allocator hands out is a block of this cache.
    ///
    /// # Errors
    ///
    /// An error when paged attention is off, the allocator lock is poisoned,
    /// or the cache tensors fail to allocate.
    pub(crate) fn paged_pool(&self) -> Result<&LayeredPagedKvCache<R>> {
        if let Some(pool) = self.paged_pool.get() {
            return Ok(pool);
        }
        let shared = self
            .shared_allocator
            .as_ref()
            .ok_or_else(|| anyhow!("the paged KV pool needs inference.paged_attention = true"))?;
        let total_blocks = shared
            .lock()
            .map_err(|e| anyhow!("Block allocator lock poisoned: {e}"))?
            .total_blocks();
        let model = self.model()?;
        let pool = LayeredPagedKvCache::new(
            model.num_layers(),
            total_blocks,
            self.config.inference.block_size,
            model.num_kv_heads().ok_or_else(|| {
                anyhow!(
                    "The model reports no KV head count, so paged attention cannot size its pool."
                )
            })?,
            model.head_dim().ok_or_else(|| {
                anyhow!(
                    "The model reports no head dimension, so paged attention cannot size its pool."
                )
            })?,
            self.config.kv_cache_dtype()?,
            &self.device,
        )?;
        // A concurrent first call can set the pool first. Its pool has the
        // same shape, so this one is dropped.
        let _ = self.paged_pool.set(pool);
        self.paged_pool
            .get()
            .ok_or_else(|| anyhow!("the paged KV pool was not stored"))
    }

    /// A lease on the pool blocks of one request, with none held yet.
    ///
    /// # Errors
    ///
    /// An error when paged attention is off or the allocator lock is poisoned.
    pub(crate) fn pool_lease(&self) -> Result<PoolLease<'_>> {
        let shared = self
            .shared_allocator
            .as_ref()
            .ok_or_else(|| anyhow!("a paged request needs inference.paged_attention = true"))?;
        let allocator = shared
            .lock()
            .map_err(|e| anyhow!("Block allocator lock poisoned: {e}"))?
            .clone();
        Ok(PoolLease {
            blocks: Vec::new(),
            allocator,
            block_size: self.config.inference.block_size,
            prefix_cache: &self.prefix_cache,
            prefix_seq: None,
            prefix_count: 0,
            #[cfg(feature = "cuda")]
            gpu_release: None,
        })
    }
}

/// The pool blocks one paged request holds, in position order.
///
/// Drop returns every block. Blocks the prefix cache handed out go back
/// through it. The allocator takes back the rest.
pub(crate) struct PoolLease<'a> {
    blocks: Vec<BlockId>,
    allocator: CpuBlockAllocator,
    block_size: usize,
    prefix_cache: &'a SharedPrefixCache,
    prefix_seq: Option<u64>,
    /// The leading blocks of `blocks` that the prefix cache handed out.
    prefix_count: usize,
    #[cfg(feature = "cuda")]
    gpu_release: Option<GpuPrefixRelease<'a>>,
}

/// The GPU prefix cache references a lease drops with its blocks.
#[cfg(feature = "cuda")]
pub(crate) struct GpuPrefixRelease<'a> {
    pub cache: &'a Option<Mutex<boostr::inference::prefix_cache::GpuPrefixCache>>,
    pub prompt_tokens: Vec<u32>,
}

impl<'a> PoolLease<'a> {
    /// The leased block ids, in position order.
    pub(crate) fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Tokens per block.
    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    /// The executor's prefix cache.
    pub(crate) fn prefix_cache(&self) -> &'a SharedPrefixCache {
        self.prefix_cache
    }

    /// Take `blocks`, which the prefix cache handed out for sequence `seq`.
    ///
    /// They must come before any block [`Self::cover`] allocates. The prefix
    /// cache takes them back on drop. It does not free a block it did not
    /// hand out, so the lease frees those itself.
    pub(crate) fn hold_prefix_blocks(&mut self, seq: u64, blocks: Vec<BlockId>) {
        self.prefix_seq = Some(seq);
        self.prefix_count = blocks.len();
        self.blocks = blocks;
    }

    /// Release the GPU prefix cache references of `release` on drop.
    ///
    /// References an earlier call took go back first, so a second prompt on
    /// one lease leaves no count behind.
    #[cfg(feature = "cuda")]
    pub(crate) fn hold_gpu_prefix(&mut self, release: GpuPrefixRelease<'a>) {
        if let Some(earlier) = self.gpu_release.replace(release) {
            super::executor_cache::gpu_prefix_cache_release(
                earlier.cache,
                &earlier.prompt_tokens,
                self.block_size,
            );
        }
    }

    /// Allocate blocks until the lease covers positions `0..tokens`.
    ///
    /// # Errors
    ///
    /// An error when the pool holds too few free blocks.
    pub(crate) fn cover(&mut self, tokens: usize) -> Result<()> {
        let needed = tokens.div_ceil(self.block_size);
        if needed <= self.blocks.len() {
            return Ok(());
        }
        let extra = self
            .allocator
            .allocate(needed - self.blocks.len())
            .map_err(|e| anyhow!("Failed to allocate KV blocks for {tokens} tokens: {e}"))?;
        self.blocks.extend(extra);
        Ok(())
    }

    /// Free the blocks past those that hold positions `0..tokens`.
    ///
    /// The prefix blocks stay, whatever `tokens` is. Nothing happens when the
    /// lease holds no block past that point.
    ///
    /// # Errors
    ///
    /// An error when the allocator refuses the free. The lease keeps the
    /// blocks and drop frees them.
    pub(crate) fn shrink_to(&mut self, tokens: usize) -> Result<()> {
        let keep = tokens.div_ceil(self.block_size).max(self.prefix_count);
        if keep >= self.blocks.len() {
            return Ok(());
        }
        self.allocator
            .free(&self.blocks[keep..])
            .map_err(|e| anyhow!("Failed to free KV blocks past {tokens} tokens: {e}"))?;
        self.blocks.truncate(keep);
        Ok(())
    }
}

impl Drop for PoolLease<'_> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if let Some(release) = self.gpu_release.take() {
            super::executor_cache::gpu_prefix_cache_release(
                release.cache,
                &release.prompt_tokens,
                self.block_size,
            );
        }
        let (prefix, own) = self.blocks.split_at(self.prefix_count);
        if let (Some(pc), Some(seq)) = (self.prefix_cache, self.prefix_seq) {
            // A poisoned prefix cache keeps its blocks. Freeing them to the
            // allocator while the cache still maps them hands one block to
            // two sequences.
            if let Ok(mut cache) = pc.lock() {
                let _ = cache.release_blocks(seq, prefix);
            }
        }
        if !own.is_empty() {
            let _ = self.allocator.free(own);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::engine::test_support::{
        free_pool_blocks as free_blocks, seeded_executor, PagedPool,
    };

    const POOL: PagedPool = PagedPool {
        block_size: 4,
        blocks: 16,
    };

    #[test]
    fn the_pool_holds_every_block_of_the_shared_allocator() -> Result<()> {
        let exec = seeded_executor(1, Some(POOL))?;
        let pool = exec.paged_pool()?;
        let k_cache = pool.layer(0).k_cache();
        assert_eq!(k_cache.shape()[0], POOL.blocks);
        Ok(())
    }

    #[test]
    fn a_dropped_lease_returns_its_blocks() -> Result<()> {
        let exec = seeded_executor(1, Some(POOL))?;
        let mut lease = exec.pool_lease()?;
        lease.cover(9)?;
        assert_eq!(lease.blocks().len(), 3);
        lease.cover(12)?;
        assert_eq!(lease.blocks().len(), 3, "12 tokens fit in 3 blocks");
        assert_eq!(free_blocks(&exec)?, POOL.blocks - 3);
        drop(lease);
        assert_eq!(free_blocks(&exec)?, POOL.blocks);
        Ok(())
    }

    #[test]
    fn shrink_to_frees_the_blocks_past_the_kept_tokens() -> Result<()> {
        let exec = seeded_executor(1, Some(POOL))?;
        let mut lease = exec.pool_lease()?;
        lease.cover(14)?;
        assert_eq!(lease.blocks().len(), 4);
        let kept = lease.blocks()[..2].to_vec();
        lease.shrink_to(5)?;
        assert_eq!(lease.blocks(), &kept[..], "5 tokens need 2 blocks");
        assert_eq!(free_blocks(&exec)?, POOL.blocks - 2);
        lease.shrink_to(8)?;
        assert_eq!(
            free_blocks(&exec)?,
            POOL.blocks - 2,
            "nothing past 8 tokens"
        );
        lease.shrink_to(0)?;
        assert!(lease.blocks().is_empty());
        assert_eq!(free_blocks(&exec)?, POOL.blocks);
        Ok(())
    }

    #[test]
    fn shrink_to_never_frees_a_prefix_block() -> Result<()> {
        let exec = seeded_executor(1, Some(POOL))?;
        let shared = exec
            .shared_allocator()
            .ok_or_else(|| anyhow!("no shared allocator"))?;
        let allocator = shared
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?
            .clone();
        let prefix = allocator.allocate(2).map_err(|e| anyhow!("{e}"))?;
        let mut lease = exec.pool_lease()?;
        lease.hold_prefix_blocks(7, prefix.clone());
        lease.cover(16)?;
        assert_eq!(lease.blocks().len(), 4);
        lease.shrink_to(0)?;
        assert_eq!(lease.blocks(), &prefix[..], "the prefix blocks stay");
        assert_eq!(free_blocks(&exec)?, POOL.blocks - 2);
        // The executor has no prefix cache, so the drop leaves the prefix blocks.
        drop(lease);
        assert_eq!(free_blocks(&exec)?, POOL.blocks - 2);
        allocator.free(&prefix).map_err(|e| anyhow!("{e}"))?;
        assert_eq!(free_blocks(&exec)?, POOL.blocks);
        Ok(())
    }
}
