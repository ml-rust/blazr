//! Inference configuration settings

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Device configuration for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DeviceConfig {
    /// Simple device string (e.g., "cuda:0", "cpu")
    Simple(String),
    /// Detailed device configuration
    Detailed {
        /// Device type: "cuda", "cpu"
        device_type: String,
        /// Device ID (for multi-GPU)
        #[serde(default)]
        device_id: usize,
    },
}

impl Default for DeviceConfig {
    fn default() -> Self {
        DeviceConfig::Simple("cuda:0".to_string())
    }
}

impl DeviceConfig {
    /// Get device type ("cuda" or "cpu")
    pub fn device_type(&self) -> &str {
        match self {
            DeviceConfig::Simple(s) => {
                if s.starts_with("cuda") {
                    "cuda"
                } else {
                    "cpu"
                }
            }
            DeviceConfig::Detailed { device_type, .. } => device_type,
        }
    }

    /// Get device ID (for multi-GPU)
    pub fn device_id(&self) -> usize {
        match self {
            DeviceConfig::Simple(s) => s
                .strip_prefix("cuda:")
                .and_then(|id| id.parse().ok())
                .unwrap_or(0),
            DeviceConfig::Detailed { device_id, .. } => *device_id,
        }
    }

    /// Check if using CUDA
    pub fn is_cuda(&self) -> bool {
        self.device_type() == "cuda"
    }
}

/// Inference-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Path to model weights (SafeTensors, GGUF, etc.)
    #[serde(default)]
    pub weights_path: Option<PathBuf>,

    /// Device configuration
    #[serde(default)]
    pub device: DeviceConfig,

    /// Maximum context length (overrides model default if smaller)
    #[serde(default)]
    pub max_context_len: Option<usize>,

    /// Data type for inference (f32, f16, bf16)
    #[serde(default = "default_dtype")]
    pub dtype: String,

    /// Enable flash attention if available
    #[serde(default = "default_true")]
    pub flash_attention: bool,

    /// Enable KV cache
    #[serde(default = "default_true")]
    pub kv_cache: bool,

    /// Maximum batch size for batched inference
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Enable paged attention (vLLM-style block-based KV cache)
    #[serde(default)]
    pub paged_attention: bool,

    /// Block size for paged attention (tokens per block)
    #[serde(default = "default_block_size")]
    pub block_size: usize,

    /// Number of KV cache blocks to pre-allocate for paged attention.
    /// If 0, auto-computed from max_context_len.
    #[serde(default)]
    pub num_blocks: usize,
}

fn default_dtype() -> String {
    "f16".to_string()
}

fn default_true() -> bool {
    true
}

fn default_max_batch_size() -> usize {
    1
}

fn default_block_size() -> usize {
    16
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            weights_path: None,
            device: DeviceConfig::default(),
            max_context_len: None,
            dtype: default_dtype(),
            flash_attention: false, // TEMP: Disabled for debugging
            kv_cache: true,
            max_batch_size: 1,
            paged_attention: false,
            block_size: default_block_size(),
            num_blocks: 0,
        }
    }
}
