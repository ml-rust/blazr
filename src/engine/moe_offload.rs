//! Asymmetric MoE expert offloading (CPU+GPU hybrid).
//!
//! Enables running large MoE models (e.g., DeepSeek-V3 671B) on consumer hardware
//! by keeping dense layers and hot experts on GPU while cold experts reside on CPU.
//!
//! # Architecture
//!
//! ```text
//! GPU VRAM                    CPU RAM
//! ┌─────────────────┐        ┌─────────────────┐
//! │ Embedding       │        │ Cold Experts    │
//! │ Dense Layers    │        │ (prefetched     │
//! │ Hot Experts     │◄─copy──│  one step ahead)│
//! │ LM Head         │        │                 │
//! └─────────────────┘        └─────────────────┘
//! ```
//!
//! # Expert Placement Strategy
//!
//! - Tracks router activation frequency per expert per layer
//! - Hot experts (top-K by frequency) stay on GPU
//! - Cold experts live on CPU, prefetched when router predicts them
//! - Rebalancing: periodically re-classify hot/cold based on frequency shifts

use std::sync::atomic::{AtomicUsize, Ordering};

pub use crate::engine::moe_offload_types::{
    ExpertFrequencyTracker, ExpertTransfer, LayerExpertPlacement, MoeOffloadConfig,
    OffloadStrategy, ResolvedPlacement, TransferDirection,
};

/// Manages MoE expert offloading across all layers.
///
/// Tracks per-layer expert frequency, manages hot/cold classification,
/// and coordinates expert weight transfers between devices.
pub struct MoeOffloadManager {
    config: MoeOffloadConfig,
    /// Per-layer frequency trackers
    layer_trackers: Vec<ExpertFrequencyTracker>,
    /// Per-layer placement
    placements: Vec<LayerExpertPlacement>,
    /// Number of experts per layer
    num_experts: usize,
    /// Number of layers
    num_layers: usize,
    /// Number of GPU experts (from resolved placement)
    gpu_expert_budget: usize,
    /// Forward pass counter for rebalancing
    forward_count: AtomicUsize,
    /// Whether rebalancing is needed
    needs_rebalance: bool,
}

impl MoeOffloadManager {
    /// Create a new offload manager for an MoE model.
    pub fn new(
        config: MoeOffloadConfig,
        num_layers: usize,
        num_experts: usize,
        placement: &ResolvedPlacement,
    ) -> Self {
        let mut layer_trackers = Vec::with_capacity(num_layers);
        let mut placements = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            layer_trackers.push(ExpertFrequencyTracker::new(config.frequency_window));

            // Initial placement: first N experts on GPU, rest on CPU
            let gpu_experts: Vec<usize> =
                (0..placement.gpu_expert_count.min(num_experts)).collect();
            let cpu_experts: Vec<usize> =
                (placement.gpu_expert_count.min(num_experts)..num_experts).collect();
            placements.push(LayerExpertPlacement {
                gpu_experts,
                cpu_experts,
            });
        }

        Self {
            config,
            layer_trackers,
            placements,
            num_experts,
            num_layers,
            gpu_expert_budget: placement.gpu_expert_count,
            forward_count: AtomicUsize::new(0),
            needs_rebalance: false,
        }
    }

    /// Record expert activations for a layer after a forward pass.
    ///
    /// `activated_experts` contains the expert indices that were selected by the router.
    pub fn record_activations(&mut self, layer_idx: usize, activated_experts: &[usize]) {
        if layer_idx < self.num_layers {
            self.layer_trackers[layer_idx].record_batch(activated_experts);
        }
    }

    /// Call after each forward pass. Returns the set of expert transfers to execute,
    /// or an empty vec if no rebalancing was performed.
    pub fn step(&mut self) -> Vec<ExpertTransfer> {
        let count = self.forward_count.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_multiple_of(self.config.rebalance_interval) {
            return self.rebalance();
        }
        Vec::new()
    }

    /// Execute a set of expert weight transfers.
    ///
    /// The callback `transfer_fn` is provided by the caller (executor) and handles
    /// the actual tensor copy between devices. After each successful transfer the
    /// placement tracking is updated so `is_on_gpu` stays accurate.
    pub fn execute_transfers<F>(&mut self, transfers: &[ExpertTransfer], mut transfer_fn: F)
    where
        F: FnMut(&ExpertTransfer) -> bool,
    {
        for transfer in transfers {
            let success = transfer_fn(transfer);
            if success {
                // Update placement tracking to reflect the completed transfer.
                match transfer.direction {
                    TransferDirection::CpuToGpu => {
                        self.placements[transfer.layer]
                            .cpu_experts
                            .retain(|&e| e != transfer.expert_id);
                        if !self.placements[transfer.layer]
                            .gpu_experts
                            .contains(&transfer.expert_id)
                        {
                            self.placements[transfer.layer]
                                .gpu_experts
                                .push(transfer.expert_id);
                        }
                    }
                    TransferDirection::GpuToCpu => {
                        self.placements[transfer.layer]
                            .gpu_experts
                            .retain(|&e| e != transfer.expert_id);
                        if !self.placements[transfer.layer]
                            .cpu_experts
                            .contains(&transfer.expert_id)
                        {
                            self.placements[transfer.layer]
                                .cpu_experts
                                .push(transfer.expert_id);
                        }
                    }
                }
                tracing::info!(
                    layer = transfer.layer,
                    expert = transfer.expert_id,
                    direction = ?transfer.direction,
                    "Expert transfer executed"
                );
            } else {
                tracing::warn!(
                    layer = transfer.layer,
                    expert = transfer.expert_id,
                    direction = ?transfer.direction,
                    "Expert transfer failed"
                );
            }
        }
    }

    /// Rebalance expert placement based on accumulated frequency data.
    ///
    /// For each layer, picks the top-K most frequent experts for GPU placement.
    /// Returns the set of (layer, expert) pairs that need to be transferred.
    pub fn rebalance(&mut self) -> Vec<ExpertTransfer> {
        let mut transfers = Vec::new();

        for layer_idx in 0..self.num_layers {
            let hot = self.layer_trackers[layer_idx].top_k(self.gpu_expert_budget);
            let old_gpu: std::collections::HashSet<usize> = self.placements[layer_idx]
                .gpu_experts
                .iter()
                .copied()
                .collect();
            let new_gpu: std::collections::HashSet<usize> = hot.iter().copied().collect();

            // Experts moving GPU → CPU
            for &expert in &old_gpu {
                if !new_gpu.contains(&expert) {
                    transfers.push(ExpertTransfer {
                        layer: layer_idx,
                        expert_id: expert,
                        direction: TransferDirection::GpuToCpu,
                    });
                }
            }

            // Experts moving CPU → GPU
            for &expert in &new_gpu {
                if !old_gpu.contains(&expert) {
                    transfers.push(ExpertTransfer {
                        layer: layer_idx,
                        expert_id: expert,
                        direction: TransferDirection::CpuToGpu,
                    });
                }
            }

            // Update placement
            self.placements[layer_idx].gpu_experts = hot;
            self.placements[layer_idx].cpu_experts = (0..self.num_experts)
                .filter(|e| !new_gpu.contains(e))
                .collect();
        }

        self.needs_rebalance = false;
        transfers
    }

    /// Get the current placement for a layer.
    pub fn placement(&self, layer_idx: usize) -> &LayerExpertPlacement {
        &self.placements[layer_idx]
    }

    /// Check if an expert is on GPU for a given layer.
    pub fn is_on_gpu(&self, layer_idx: usize, expert_id: usize) -> bool {
        self.placements[layer_idx].gpu_experts.contains(&expert_id)
    }

    /// Predict which experts will be needed for the next token based on
    /// current frequency data. Used for async prefetch.
    pub fn predict_next_experts(&self, layer_idx: usize, top_k: usize) -> Vec<usize> {
        self.layer_trackers[layer_idx].top_k(top_k)
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get the number of experts.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get the GPU expert budget.
    pub fn gpu_expert_budget(&self) -> usize {
        self.gpu_expert_budget
    }

    /// Get frequency tracker for a layer.
    pub fn tracker(&self, layer_idx: usize) -> &ExpertFrequencyTracker {
        &self.layer_trackers[layer_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_tracker() {
        let mut tracker = ExpertFrequencyTracker::new(100);
        for _ in 0..10 {
            tracker.record(0);
        }
        for _ in 0..5 {
            tracker.record(1);
        }
        tracker.record(2);

        let top = tracker.top_k(2);
        assert_eq!(top[0], 0);
        assert_eq!(top[1], 1);
        assert!(tracker.frequency(0) > tracker.frequency(1));
    }

    #[test]
    fn test_frequency_tracker_decay() {
        let mut tracker = ExpertFrequencyTracker::new(10);
        // Fill past window
        for _ in 0..12 {
            tracker.record(0);
        }
        // Counts should have been halved
        assert!(tracker.frequency(0) > 0.0);
    }

    #[test]
    fn test_resolved_placement() {
        let config = MoeOffloadConfig {
            strategy: OffloadStrategy::Hybrid,
            gpu_experts: 4,
            ..Default::default()
        };
        let placement = config.resolve_strategy(8, 1_000_000, 100_000_000);
        assert_eq!(placement.gpu_expert_count, 4);
        assert_eq!(placement.cpu_expert_count, 4);
        assert!(placement.is_hybrid());
    }

    #[test]
    fn test_auto_strategy_all_fit() {
        let config = MoeOffloadConfig {
            strategy: OffloadStrategy::Auto,
            ..Default::default()
        };
        // 8 experts * 100MB each = 800MB, VRAM = 2GB → all fit
        let placement = config.resolve_strategy(8, 100_000_000, 2_000_000_000);
        assert!(placement.is_all_gpu());
    }

    #[test]
    fn test_auto_strategy_partial_fit() {
        let config = MoeOffloadConfig {
            strategy: OffloadStrategy::Auto,
            ..Default::default()
        };
        // 8 experts * 500MB each = 4GB, VRAM = 2GB → hybrid
        let placement = config.resolve_strategy(8, 500_000_000, 2_000_000_000);
        assert!(placement.is_hybrid());
        assert!(placement.gpu_expert_count < 8);
        assert!(placement.gpu_expert_count > 0);
    }

    #[test]
    fn test_offload_manager_rebalance() {
        let config = MoeOffloadConfig {
            strategy: OffloadStrategy::Hybrid,
            gpu_experts: 2,
            rebalance_interval: 5,
            ..Default::default()
        };
        let placement = ResolvedPlacement {
            gpu_expert_count: 2,
            cpu_expert_count: 2,
        };
        let mut manager = MoeOffloadManager::new(config, 1, 4, &placement);

        // Initially experts 0,1 on GPU; 2,3 on CPU
        assert!(manager.is_on_gpu(0, 0));
        assert!(manager.is_on_gpu(0, 1));
        assert!(!manager.is_on_gpu(0, 2));
        assert!(!manager.is_on_gpu(0, 3));

        // Record lots of activations for experts 2 and 3
        for _ in 0..50 {
            manager.record_activations(0, &[2, 3]);
        }

        // Rebalance
        let transfers = manager.rebalance();
        assert!(!transfers.is_empty());

        // Now experts 2,3 should be on GPU
        assert!(manager.is_on_gpu(0, 2));
        assert!(manager.is_on_gpu(0, 3));
    }

    #[test]
    fn test_offload_manager_step_triggers_rebalance() {
        let config = MoeOffloadConfig {
            strategy: OffloadStrategy::Hybrid,
            gpu_experts: 2,
            rebalance_interval: 3,
            ..Default::default()
        };
        let placement = ResolvedPlacement {
            gpu_expert_count: 2,
            cpu_expert_count: 2,
        };
        let mut manager = MoeOffloadManager::new(config, 1, 4, &placement);

        assert!(manager.step().is_empty()); // 1
        assert!(manager.step().is_empty()); // 2
        assert!(!manager.step().is_empty()); // 3 → rebalance
    }
}
