//! Data parallelism: multi-replica inference with request-level sharding.
//!
//! Runs multiple identical copies of a model (replicas) and distributes
//! incoming requests across them. Supports round-robin and cache-aware routing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};

use super::Executor;
use crate::engine::cache_router::CacheAwareRouter;

/// Data-parallel model serving across multiple replicas
pub struct DataParallelGroup<R: Runtime<DType = DType>> {
    /// Replicas of the same model
    replicas: Vec<Arc<Executor<R>>>,
    /// Cache-aware router (if enabled) or round-robin fallback
    router: Router<R>,
    /// Model name
    model_name: String,
}

enum Router<R: Runtime<DType = DType>> {
    RoundRobin(AtomicUsize),
    CacheAware(CacheAwareRouter<R>),
}

impl<R: Runtime<DType = DType>> DataParallelGroup<R>
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
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    /// Create a new data-parallel group from pre-loaded replicas.
    pub fn new(model_name: String, replicas: Vec<Arc<Executor<R>>>) -> Result<Self> {
        if replicas.is_empty() {
            return Err(anyhow!("DataParallelGroup requires at least one replica"));
        }
        Ok(Self {
            replicas,
            router: Router::RoundRobin(AtomicUsize::new(0)),
            model_name,
        })
    }

    /// Create with cache-aware routing
    pub fn with_cache_aware_routing(
        model_name: String,
        replicas: Vec<Arc<Executor<R>>>,
    ) -> Result<Self> {
        if replicas.is_empty() {
            return Err(anyhow!("DataParallelGroup requires at least one replica"));
        }
        let router = CacheAwareRouter::new(replicas.clone());
        Ok(Self {
            replicas,
            router: Router::CacheAware(router),
            model_name,
        })
    }

    /// Select the best replica for a request.
    ///
    /// With round-robin: cycles through replicas.
    /// With cache-aware: picks replica most likely to have warm KV cache for the prefix.
    pub fn select(&self, prompt_prefix: Option<&str>) -> Arc<Executor<R>> {
        match &self.router {
            Router::RoundRobin(counter) => {
                let idx = counter.fetch_add(1, Ordering::Relaxed) % self.replicas.len();
                Arc::clone(&self.replicas[idx])
            }
            Router::CacheAware(router) => router.route(prompt_prefix),
        }
    }

    /// Number of replicas in this group
    pub fn num_replicas(&self) -> usize {
        self.replicas.len()
    }

    /// Model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get all replicas
    pub fn replicas(&self) -> &[Arc<Executor<R>>] {
        &self.replicas
    }

    /// Notify the router that a request with this prefix was served by a replica.
    /// Only meaningful with cache-aware routing.
    pub fn record_served(&self, replica_idx: usize, prompt_prefix: &str) {
        if let Router::CacheAware(ref router) = self.router {
            router.record(replica_idx, prompt_prefix);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full tests require model loading. These test the routing logic only.

    #[test]
    fn test_round_robin_counter() {
        let counter = AtomicUsize::new(0);
        let n = 3;
        for expected in 0..9 {
            let idx = counter.fetch_add(1, Ordering::Relaxed) % n;
            assert_eq!(idx, expected % n);
        }
    }
}
