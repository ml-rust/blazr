//! Benchmark methodology configuration.
//!
//! Standardized benchmark matrix for reproducible performance measurement.
//! Defines workload profiles, hardware context, and comparison methodology.

use serde::{Deserialize, Serialize};

/// Standardized benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Model under test
    pub model: ModelSpec,
    /// Workload profiles to run
    pub workloads: Vec<WorkloadProfile>,
    /// Number of warmup iterations (excluded from measurement)
    #[serde(default = "default_warmup")]
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    #[serde(default = "default_iterations")]
    pub measurement_iterations: usize,
    /// Concurrency levels to sweep
    #[serde(default = "default_concurrency")]
    pub concurrency_levels: Vec<usize>,
    /// Hardware context (auto-detected if not specified)
    #[serde(default)]
    pub hardware: Option<HardwareContext>,
}

fn default_warmup() -> usize {
    3
}
fn default_iterations() -> usize {
    10
}
fn default_concurrency() -> Vec<usize> {
    vec![1]
}

/// Model specification for benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Model name/path
    pub name: String,
    /// Format: "safetensors", "gguf", "awq", "gptq"
    pub format: String,
    /// Quantization level (e.g., "Q4_K_M", "INT4", "F16")
    pub quantization: String,
    /// Parameter count (for normalization)
    #[serde(default)]
    pub params_billions: Option<f64>,
}

/// Workload profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadProfile {
    /// Profile name (e.g., "short_prompt", "long_context")
    pub name: String,
    /// Prompt token count
    pub prompt_tokens: usize,
    /// Max output tokens
    pub output_tokens: usize,
    /// Whether to stream
    #[serde(default)]
    pub stream: bool,
}

/// Hardware context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareContext {
    /// GPU model (e.g., "RTX 4090", "A100 80GB")
    pub gpu: Option<String>,
    /// CPU model
    pub cpu: Option<String>,
    /// System RAM (GB)
    pub ram_gb: Option<f64>,
    /// VRAM (GB)
    pub vram_gb: Option<f64>,
    /// CUDA version
    pub cuda_version: Option<String>,
    /// Driver version
    pub driver_version: Option<String>,
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Configuration used
    pub config: BenchmarkConfig,
    /// Per-workload results
    pub results: Vec<WorkloadResult>,
    /// Timestamp
    pub timestamp: String,
    /// blazr version
    pub version: String,
}

/// Results for one workload profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadResult {
    /// Profile name
    pub profile: String,
    /// Concurrency level
    pub concurrency: usize,
    /// Metrics
    pub metrics: BenchMetrics,
}

/// Measured metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchMetrics {
    /// Time to first token (ms): P50, P95, P99
    pub ttft_p50_ms: f64,
    pub ttft_p95_ms: f64,
    pub ttft_p99_ms: f64,
    /// Inter-token latency (ms): P50, P95, P99
    pub itl_p50_ms: f64,
    pub itl_p95_ms: f64,
    pub itl_p99_ms: f64,
    /// Decode throughput (tokens/sec): median
    pub decode_tps_median: f64,
    /// End-to-end latency (ms): P50, P95, P99
    pub e2e_p50_ms: f64,
    pub e2e_p95_ms: f64,
    pub e2e_p99_ms: f64,
    /// Peak VRAM usage (MB)
    pub peak_vram_mb: Option<f64>,
    /// Requests per second at this concurrency
    pub requests_per_second: f64,
}

/// Standard workload profiles
impl WorkloadProfile {
    /// Short prompt, short output (chatbot single-turn)
    pub fn short() -> Self {
        Self {
            name: "short".to_string(),
            prompt_tokens: 32,
            output_tokens: 64,
            stream: true,
        }
    }

    /// Medium prompt (chat with context)
    pub fn medium() -> Self {
        Self {
            name: "medium".to_string(),
            prompt_tokens: 128,
            output_tokens: 256,
            stream: true,
        }
    }

    /// Long prompt (document analysis)
    pub fn long() -> Self {
        Self {
            name: "long".to_string(),
            prompt_tokens: 512,
            output_tokens: 256,
            stream: true,
        }
    }

    /// Very long context
    pub fn long_context() -> Self {
        Self {
            name: "long_context".to_string(),
            prompt_tokens: 2048,
            output_tokens: 128,
            stream: true,
        }
    }

    /// Code generation (medium prompt, long output)
    pub fn code_gen() -> Self {
        Self {
            name: "code_gen".to_string(),
            prompt_tokens: 256,
            output_tokens: 512,
            stream: true,
        }
    }

    /// Standard benchmark matrix (all profiles)
    pub fn standard_matrix() -> Vec<Self> {
        vec![
            Self::short(),
            Self::medium(),
            Self::long(),
            Self::long_context(),
            Self::code_gen(),
        ]
    }
}

impl BenchmarkConfig {
    /// Standard concurrency sweep levels
    pub fn standard_concurrency() -> Vec<usize> {
        vec![1, 2, 4, 8, 16, 32]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_matrix() {
        let profiles = WorkloadProfile::standard_matrix();
        assert_eq!(profiles.len(), 5);
        assert_eq!(profiles[0].name, "short");
        assert_eq!(profiles[4].name, "code_gen");
    }

    #[test]
    fn test_benchmark_config_serde() {
        let config = BenchmarkConfig {
            model: ModelSpec {
                name: "llama-3.2-1b".to_string(),
                format: "safetensors".to_string(),
                quantization: "F16".to_string(),
                params_billions: Some(1.0),
            },
            workloads: WorkloadProfile::standard_matrix(),
            warmup_iterations: 3,
            measurement_iterations: 10,
            concurrency_levels: vec![1, 4, 16],
            hardware: None,
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: BenchmarkConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model.name, "llama-3.2-1b");
        assert_eq!(parsed.workloads.len(), 5);
    }
}
