//! Configuration system for blazr
//!
//! BlazrConfig is a superset of boostr's UniversalConfig, adding
//! inference-specific and server settings.

mod generation;
mod inference;
mod server;

pub use generation::GenerationConfig;
pub use inference::{DeviceConfig, InferenceConfig};
pub use server::ServerConfig;

use std::path::Path;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use boostr::model::UniversalConfig;
use boostr::DType;

/// Parse a dtype string into a `DType`.
///
/// Accepts short and long forms: "f32"/"float32", "f16"/"float16", "bf16"/"bfloat16".
/// Returns an error for unknown strings or when f16/bf16 is requested without the `f16` feature.
pub fn parse_dtype(s: &str) -> Result<DType> {
    match s {
        "f32" | "float32" => Ok(DType::F32),
        #[cfg(feature = "f16")]
        "f16" | "float16" => Ok(DType::F16),
        #[cfg(feature = "f16")]
        "bf16" | "bfloat16" => Ok(DType::BF16),
        #[cfg(not(feature = "f16"))]
        "f16" | "float16" | "bf16" | "bfloat16" => Err(anyhow!(
            "dtype '{}' requested but the 'f16' feature is not enabled; \
             rebuild with `--features f16` or use 'f32'",
            s
        )),
        other => Err(anyhow!("unknown dtype: '{}'", other)),
    }
}

/// Blazr configuration
///
/// Extends boostr's UniversalConfig with inference and server settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlazrConfig {
    /// Model architecture (from boostr::model::UniversalConfig)
    #[serde(flatten)]
    pub model: UniversalConfig,

    /// Inference-specific settings
    #[serde(default)]
    pub inference: InferenceConfig,

    /// Server settings (only for `blazr serve`)
    #[serde(default)]
    pub server: Option<ServerConfig>,

    /// Default generation settings
    #[serde(default)]
    pub generation: GenerationConfig,
}

impl BlazrConfig {
    /// Load configuration from a YAML file
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from a JSON file
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Create from a UniversalConfig with default inference settings
    pub fn from_universal(model: UniversalConfig) -> Self {
        Self {
            model,
            inference: InferenceConfig::default(),
            server: None,
            generation: GenerationConfig::default(),
        }
    }

    /// Create from a UniversalConfig with specified dtype
    pub fn from_universal_with_dtype(model: UniversalConfig, dtype: &str) -> Self {
        let inference = InferenceConfig {
            dtype: dtype.to_string(),
            ..Default::default()
        };
        Self {
            model,
            inference,
            server: None,
            generation: GenerationConfig::default(),
        }
    }

    /// Get the inference dtype
    pub fn dtype(&self) -> &str {
        &self.inference.dtype
    }

    /// Get the model type
    pub fn model_type(&self) -> &str {
        &self.model.model_type
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.model.hidden_size
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.model.num_layers
    }

    /// Get maximum sequence length (respecting inference override)
    pub fn max_seq_len(&self) -> usize {
        self.inference
            .max_context_len
            .unwrap_or(self.model.max_seq_len)
    }
}

impl Default for BlazrConfig {
    fn default() -> Self {
        Self {
            model: UniversalConfig {
                model_type: String::new(),
                vocab_size: 0,
                hidden_size: 0,
                num_layers: 0,
                max_seq_len: 0,
                intermediate_size: None,
                rms_norm_eps: 1e-5,
                attention: None,
                ssm: None,
                moe: None,
                hybrid_layers: None,
                tie_word_embeddings: false,
            },
            inference: InferenceConfig::default(),
            server: None,
            generation: GenerationConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blazr_config_yaml() {
        let yaml = r#"
model_type: mistral
vocab_size: 32000
hidden_size: 4096
num_layers: 32
max_seq_len: 8192
intermediate_size: 14336

attention:
  num_heads: 32
  num_kv_heads: 8
  rope_theta: 10000.0
  sliding_window: 4096

inference:
  weights_path: ./model.safetensors
  device: cuda:0
  dtype: f16
  max_context_len: 4096

server:
  port: 8080
  host: 0.0.0.0
  max_concurrent_requests: 16

generation:
  max_tokens: 2048
  temperature: 0.7
"#;
        let config: BlazrConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.model_type(), "mistral");
        assert_eq!(config.vocab_size(), 32000);
        assert_eq!(config.max_seq_len(), 4096); // Uses inference override
        assert!(config.server.is_some());
        assert_eq!(config.server.unwrap().port, 8080);
    }

    #[test]
    fn test_parse_dtype_f32() {
        assert_eq!(parse_dtype("f32").unwrap(), DType::F32);
        assert_eq!(parse_dtype("float32").unwrap(), DType::F32);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_parse_dtype_f16() {
        assert_eq!(parse_dtype("f16").unwrap(), DType::F16);
        assert_eq!(parse_dtype("float16").unwrap(), DType::F16);
        assert_eq!(parse_dtype("bf16").unwrap(), DType::BF16);
        assert_eq!(parse_dtype("bfloat16").unwrap(), DType::BF16);
    }

    #[cfg(not(feature = "f16"))]
    #[test]
    fn test_parse_dtype_f16_without_feature() {
        assert!(parse_dtype("f16").is_err());
        assert!(parse_dtype("bf16").is_err());
        assert!(parse_dtype("float16").is_err());
        assert!(parse_dtype("bfloat16").is_err());
    }

    #[test]
    fn test_parse_dtype_unknown() {
        assert!(parse_dtype("int8").is_err());
        assert!(parse_dtype("").is_err());
        assert!(parse_dtype("F32").is_err());
    }
}
