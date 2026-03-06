//! Types for MoE expert offloading: strategies, placement, transfers, and frequency tracking.

use std::collections::HashMap;

/// Offloading strategy for MoE expert placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffloadStrategy {
    /// All experts on GPU (requires sufficient VRAM)
    Gpu,
    /// All experts on CPU (slower but fits any hardware)
    Cpu,
    /// Hybrid: hot experts on GPU, cold on CPU with async prefetch
    Hybrid,
    /// Auto-detect based on available VRAM
    #[default]
    Auto,
}

/// Configuration for MoE expert offloading.
#[derive(Debug, Clone)]
pub struct MoeOffloadConfig {
    /// Offloading strategy
    pub strategy: OffloadStrategy,
    /// Number of experts to keep on GPU (for Hybrid mode)
    /// If 0, auto-computed from available VRAM.
    pub gpu_experts: usize,
    /// Enable async prefetch of predicted next experts
    pub async_prefetch: bool,
    /// Window size for tracking expert activation frequency
    pub frequency_window: usize,
    /// How often to rebalance hot/cold classification (in forward passes)
    pub rebalance_interval: usize,
}

impl MoeOffloadConfig {
    /// Create from InferenceConfig settings.
    pub fn from_inference_config(config: &crate::config::InferenceConfig) -> Option<Self> {
        let strategy = match config.moe_offload.as_deref() {
            Some("gpu") => OffloadStrategy::Gpu,
            Some("cpu") => OffloadStrategy::Cpu,
            Some("hybrid") => OffloadStrategy::Hybrid,
            Some("auto") => OffloadStrategy::Auto,
            Some(_) | None => return None,
        };
        Some(Self {
            strategy,
            gpu_experts: config.moe_gpu_experts,
            async_prefetch: true,
            frequency_window: 1000,
            rebalance_interval: 100,
        })
    }

    /// Resolve the effective strategy given model info and available resources.
    pub fn resolve_strategy(
        &self,
        num_experts: usize,
        expert_size_bytes: usize,
        available_vram_bytes: usize,
    ) -> ResolvedPlacement {
        match self.strategy {
            OffloadStrategy::Gpu => ResolvedPlacement {
                gpu_expert_count: num_experts,
                cpu_expert_count: 0,
            },
            OffloadStrategy::Cpu => ResolvedPlacement {
                gpu_expert_count: 0,
                cpu_expert_count: num_experts,
            },
            OffloadStrategy::Hybrid => {
                let gpu_count = if self.gpu_experts > 0 {
                    self.gpu_experts.min(num_experts)
                } else {
                    // Auto: fit as many experts as VRAM allows, reserving 20% for activations
                    let usable = (available_vram_bytes as f64 * 0.8) as usize;
                    (usable / expert_size_bytes.max(1)).min(num_experts)
                };
                ResolvedPlacement {
                    gpu_expert_count: gpu_count,
                    cpu_expert_count: num_experts.saturating_sub(gpu_count),
                }
            }
            OffloadStrategy::Auto => {
                let total_expert_bytes = num_experts * expert_size_bytes;
                if total_expert_bytes <= (available_vram_bytes as f64 * 0.8) as usize {
                    // Everything fits on GPU
                    ResolvedPlacement {
                        gpu_expert_count: num_experts,
                        cpu_expert_count: 0,
                    }
                } else {
                    // Hybrid: fit what we can
                    let usable = (available_vram_bytes as f64 * 0.8) as usize;
                    let gpu_count = (usable / expert_size_bytes.max(1)).min(num_experts);
                    ResolvedPlacement {
                        gpu_expert_count: gpu_count,
                        cpu_expert_count: num_experts.saturating_sub(gpu_count),
                    }
                }
            }
        }
    }
}

impl Default for MoeOffloadConfig {
    fn default() -> Self {
        Self {
            strategy: OffloadStrategy::Auto,
            gpu_experts: 0,
            async_prefetch: true,
            frequency_window: 1000,
            rebalance_interval: 100,
        }
    }
}

/// Resolved expert placement after considering VRAM budget.
#[derive(Debug, Clone)]
pub struct ResolvedPlacement {
    pub gpu_expert_count: usize,
    pub cpu_expert_count: usize,
}

impl ResolvedPlacement {
    pub fn is_hybrid(&self) -> bool {
        self.gpu_expert_count > 0 && self.cpu_expert_count > 0
    }

    pub fn is_all_gpu(&self) -> bool {
        self.cpu_expert_count == 0
    }

    pub fn is_all_cpu(&self) -> bool {
        self.gpu_expert_count == 0
    }
}

/// Tracks expert activation frequency for hot/cold classification.
///
/// Uses exponential decay windowing to adapt to changing routing patterns.
pub struct ExpertFrequencyTracker {
    /// Activation counts per expert in the current window
    counts: HashMap<usize, usize>,
    /// Total activations in the current window
    total: usize,
    /// Window size (decay at this threshold)
    window: usize,
}

impl ExpertFrequencyTracker {
    pub fn new(window: usize) -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
            window,
        }
    }

    /// Record an expert activation.
    pub fn record(&mut self, expert_id: usize) {
        *self.counts.entry(expert_id).or_insert(0) += 1;
        self.total += 1;

        // Decay counts by half when window is reached
        if self.total >= self.window {
            for count in self.counts.values_mut() {
                *count /= 2;
            }
            self.total /= 2;
        }
    }

    /// Record multiple expert activations at once (from a batch).
    pub fn record_batch(&mut self, expert_ids: &[usize]) {
        for &id in expert_ids {
            self.record(id);
        }
    }

    /// Get the top-K most frequently activated experts.
    pub fn top_k(&self, k: usize) -> Vec<usize> {
        let mut sorted: Vec<(usize, usize)> = self.counts.iter().map(|(&e, &c)| (e, c)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(k).map(|(e, _)| e).collect()
    }

    /// Get activation frequency for a specific expert.
    pub fn frequency(&self, expert_id: usize) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        *self.counts.get(&expert_id).unwrap_or(&0) as f32 / self.total as f32
    }

    /// Reset all counts.
    pub fn reset(&mut self) {
        self.counts.clear();
        self.total = 0;
    }
}

/// Per-layer expert placement state.
#[derive(Debug, Clone)]
pub struct LayerExpertPlacement {
    /// Which expert indices are currently on GPU
    pub gpu_experts: Vec<usize>,
    /// Which expert indices are currently on CPU
    pub cpu_experts: Vec<usize>,
}

/// Describes a required expert weight transfer between devices.
#[derive(Debug, Clone)]
pub struct ExpertTransfer {
    /// Layer index
    pub layer: usize,
    /// Expert index within the layer
    pub expert_id: usize,
    /// Direction of transfer
    pub direction: TransferDirection,
}

/// Direction of an expert weight transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Move expert weights from GPU to CPU (eviction)
    GpuToCpu,
    /// Move expert weights from CPU to GPU (prefetch/promote)
    CpuToGpu,
}
