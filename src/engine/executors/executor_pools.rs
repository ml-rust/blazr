//! The shared KV block pool and prefix caches an executor starts with.

use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};

use boostr::inference::memory::{BlockTable, CpuBlockAllocator};
use boostr::inference::prefix_cache::{PrefixCache, PrefixCacheConfig};
use boostr::{DType, Runtime};

use crate::config::BlazrConfig;

/// The block allocator and prefix caches of one executor.
pub(crate) struct KvPools {
    /// Shared block allocator for paged attention.
    pub(crate) shared_allocator: Option<Arc<Mutex<CpuBlockAllocator>>>,
    /// Shared prefix cache for paged attention.
    pub(crate) prefix_cache: Option<Mutex<PrefixCache<CpuBlockAllocator>>>,
    /// GPU prefix cache, opt-in by config.
    #[cfg(feature = "cuda")]
    pub(crate) gpu_prefix_cache: Option<Mutex<boostr::inference::prefix_cache::GpuPrefixCache>>,
}

impl KvPools {
    /// No pool and no cache: the state of an executor that holds no weights.
    pub(crate) fn none() -> Self {
        Self {
            shared_allocator: None,
            prefix_cache: None,
            #[cfg(feature = "cuda")]
            gpu_prefix_cache: None,
        }
    }

    /// The pools `config.inference` asks for, sized for `num_ctx` tokens per
    /// sequence when no explicit pool size is set.
    ///
    /// # Errors
    ///
    /// An error when the allocator lock is poisoned.
    pub(crate) fn new<R: Runtime<DType = DType>>(
        config: &BlazrConfig,
        num_ctx: usize,
        device: &R::Device,
    ) -> Result<Self> {
        let inference = &config.inference;
        let shared_allocator = if inference.paged_attention {
            let block_size = inference.block_size;
            let total_blocks = if inference.kv_pool_blocks > 0 {
                inference.kv_pool_blocks
            } else {
                // Enough for max_batch_size requests at max context, plus a
                // fifth for the prefix cache and fragmentation.
                let max_ctx = inference.max_context_len.unwrap_or(num_ctx);
                let blocks_per_seq = BlockTable::blocks_needed(max_ctx, block_size);
                let base = blocks_per_seq * inference.max_batch_size.max(1);
                base + base / 5 + 64
            };
            tracing::info!(
                "Shared KV block pool: {} blocks × {} tokens/block = {} token slots",
                total_blocks,
                block_size,
                total_blocks * block_size
            );
            Some(Arc::new(Mutex::new(CpuBlockAllocator::new(
                total_blocks,
                block_size,
            ))))
        } else {
            None
        };

        let prefix_cache = if inference.prefix_cache && inference.paged_attention {
            let block_size = inference.block_size;
            let max_cached = inference.max_cached_blocks;
            let allocator = if let Some(ref shared) = shared_allocator {
                shared
                    .lock()
                    .map_err(|e| anyhow!("Block allocator lock poisoned: {e}"))?
                    .clone()
            } else {
                CpuBlockAllocator::new(max_cached + 1024, block_size)
            };
            let cache_config = PrefixCacheConfig {
                enabled: true,
                max_cached_blocks: max_cached,
                block_size,
                ..PrefixCacheConfig::default()
            };
            tracing::info!(
                "Prefix cache enabled: max_cached_blocks={}, block_size={}",
                max_cached,
                block_size
            );
            Some(Mutex::new(PrefixCache::new(allocator, cache_config)))
        } else {
            None
        };

        #[cfg(not(feature = "cuda"))]
        let _ = device;
        Ok(Self {
            shared_allocator,
            prefix_cache,
            #[cfg(feature = "cuda")]
            gpu_prefix_cache: gpu_prefix_cache::<R>(config, device),
        })
    }
}

/// The GPU prefix cache on the executor's GPU, when the config opts in.
#[cfg(feature = "cuda")]
fn gpu_prefix_cache<R: Runtime<DType = DType>>(
    config: &BlazrConfig,
    device: &R::Device,
) -> Option<Mutex<boostr::inference::prefix_cache::GpuPrefixCache>> {
    use boostr::runtime::Device;
    let inference = &config.inference;
    if !(inference.gpu_prefix_cache && inference.prefix_cache && inference.paged_attention) {
        return None;
    }
    let capacity = inference.max_cached_blocks;
    let block_size = inference.block_size;
    let ram_tier = inference.gpu_prefix_cache_ram_tier;
    tracing::info!(
        "GPU prefix cache enabled: capacity={}, block_size={}, ram_tier={}",
        capacity,
        block_size,
        ram_tier,
    );
    // The CudaDevice matches the executor's GPU through Device::id().
    let cuda_device = boostr::CudaDevice::new(device.id());
    Some(Mutex::new(
        boostr::inference::prefix_cache::GpuPrefixCache::new(
            capacity,
            block_size,
            cuda_device,
            ram_tier,
        ),
    ))
}
