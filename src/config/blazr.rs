use std::path::Path;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use boostr::model::UniversalConfig;
use boostr::DType;

use super::InferenceConfig;

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
            "dtype '{}' needs the 'f16' build feature. Rebuild with `--features f16`, or use \
             'f32'",
            s
        )),
        other => Err(anyhow!("unknown dtype: '{}'", other)),
    }
}

/// A model's configuration: boostr's `UniversalConfig` plus the per-model
/// `inference` section of its `config.json`.
///
/// Server and generation settings live in `~/.blazr/config.yaml`. A model
/// `config.json` that still holds `server` or `generation` parses, and
/// those sections are ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlazrConfig {
    /// Model architecture (from boostr::model::UniversalConfig)
    #[serde(flatten)]
    pub model: UniversalConfig,

    /// Inference-specific settings
    #[serde(default)]
    pub inference: InferenceConfig,
}

impl BlazrConfig {
    /// Load configuration from a YAML file
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: Self = serde_saphyr::from_str(&content)?;
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
        }
    }

    /// Create from a UniversalConfig with specified dtype
    pub fn from_universal_with_dtype(model: UniversalConfig, dtype: &str) -> Self {
        let inference = InferenceConfig {
            dtype: dtype.to_string(),
            ..Default::default()
        };
        Self { model, inference }
    }

    /// Get the inference dtype
    pub fn dtype(&self) -> &str {
        &self.inference.dtype
    }

    /// The compute dtype the loader resolved. The weights hold it, and the
    /// KV cache and every activation buffer take it.
    ///
    /// # Errors
    ///
    /// An error when `inference.dtype` is not f32, bf16 or f16.
    pub fn compute_dtype(&self) -> Result<DType> {
        parse_dtype(&self.inference.dtype)
    }

    /// The KV cache dtype: always [`Self::compute_dtype`]. The attention
    /// append writes activations into the cache, so the two must agree.
    ///
    /// # Errors
    ///
    /// As [`Self::compute_dtype`].
    pub fn kv_cache_dtype(&self) -> Result<DType> {
        self.compute_dtype()
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
                grow_vocab: false,
                vision: None,
                audio: None,
                gdn: None,
                qwen35_attention: None,
                hadamard: None,
                block: Default::default(),
                quant_formats: Vec::new(),
            },
            inference: InferenceConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GenerationConfig, ServerConfig};

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
  max_context_len: 4096

server:
  port: 8080

generation:
  temperature: 0.7
"#;
        // Old model configs carry `server` and `generation`: they still parse.
        let config: BlazrConfig = serde_saphyr::from_str(yaml).unwrap();
        assert_eq!(config.model_type(), "mistral");
        assert_eq!(config.vocab_size(), 32000);
        assert_eq!(config.max_seq_len(), 4096); // Uses inference override
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

    #[test]
    fn test_minimal_config_yaml() {
        let yaml = r#"
model_type: llama
vocab_size: 128
hidden_size: 64
num_layers: 2
max_seq_len: 512
"#;
        let config: BlazrConfig = serde_saphyr::from_str(yaml).unwrap();
        assert_eq!(config.model_type(), "llama");
        assert_eq!(config.vocab_size(), 128);
        assert_eq!(config.num_layers(), 2);
        // No inference override, should use model's max_seq_len
        assert_eq!(config.max_seq_len(), 512);
    }

    #[test]
    fn test_config_default_values() {
        let config = BlazrConfig::default();
        assert_eq!(config.model_type(), "");
        assert_eq!(config.vocab_size(), 0);
        assert_eq!(config.hidden_size(), 0);
        assert_eq!(config.num_layers(), 0);
        assert_eq!(config.dtype(), "f16"); // default dtype is f16
    }

    #[test]
    fn test_config_json_parsing() {
        let json = r#"{
            "model_type": "mistral",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_layers": 32,
            "max_seq_len": 8192,
            "inference": { "dtype": "f32", "max_batch_size": 4 }
        }"#;
        let config: BlazrConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type(), "mistral");
        assert_eq!(config.inference.max_batch_size, 4);
        // The loader sets dtype from the checkpoint: the key is ignored.
        assert_eq!(config.dtype(), "f16");
    }

    #[test]
    fn test_config_max_seq_len_override() {
        let yaml = r#"
model_type: test
vocab_size: 100
hidden_size: 64
num_layers: 1
max_seq_len: 8192
inference:
  max_context_len: 2048
"#;
        let config: BlazrConfig = serde_saphyr::from_str(yaml).unwrap();
        // inference.max_context_len should override model.max_seq_len
        assert_eq!(config.max_seq_len(), 2048);
    }

    #[test]
    fn test_generation_config_presets() {
        let greedy = GenerationConfig::greedy();
        assert!(greedy.is_greedy());
        assert_eq!(greedy.temperature, 0.0);

        let creative = GenerationConfig::creative();
        assert!(!creative.is_greedy());
        assert!(creative.temperature > 1.0);
        assert_eq!(creative.top_k, 50);

        let balanced = GenerationConfig::balanced();
        assert_eq!(balanced.temperature, 0.7);
        assert_eq!(balanced.top_p, 0.9);
    }

    #[test]
    fn test_generation_config_has_penalties() {
        let mut config = GenerationConfig::default();
        assert!(config.has_penalties()); // default repeat_penalty = 1.1

        config.repeat_penalty = 1.0;
        assert!(!config.has_penalties());

        config.frequency_penalty = 0.5;
        assert!(config.has_penalties());

        config.frequency_penalty = 0.0;
        config.presence_penalty = 0.3;
        assert!(config.has_penalties());
    }

    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.max_concurrent_requests, 16);
        assert_eq!(config.request_timeout_secs, 300);
        assert!(config.cors_enabled);
        assert!(config.request_logging);
        assert_eq!(config.max_body_size, 10 * 1024 * 1024);
        assert_eq!(config.addr(), "0.0.0.0:8080");
    }

    #[test]
    fn test_generation_config_yaml_partial() {
        // Only some fields specified, rest should be defaults
        let config: GenerationConfig = serde_saphyr::from_str("temperature: 0.0\n").unwrap();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.max_tokens, 2048); // default
        assert_eq!(config.top_p, 1.0); // default
    }

    #[test]
    fn test_from_universal_config() {
        let universal = UniversalConfig {
            model_type: "llama".to_string(),
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            max_seq_len: 8192,
            intermediate_size: Some(14336),
            rms_norm_eps: 1e-5,
            attention: None,
            ssm: None,
            moe: None,
            hybrid_layers: None,
            tie_word_embeddings: false,
            grow_vocab: false,
            vision: None,
            audio: None,
            gdn: None,
            qwen35_attention: None,
            hadamard: None,
            block: Default::default(),
            quant_formats: Vec::new(),
        };
        let config = BlazrConfig::from_universal(universal);
        assert_eq!(config.model_type(), "llama");
        assert_eq!(config.vocab_size(), 32000);
        assert_eq!(config.dtype(), "f16"); // default inference dtype

        let config2 = BlazrConfig::from_universal_with_dtype(
            UniversalConfig {
                model_type: "mistral".to_string(),
                vocab_size: 32000,
                hidden_size: 4096,
                num_layers: 32,
                max_seq_len: 8192,
                intermediate_size: None,
                rms_norm_eps: 1e-5,
                attention: None,
                ssm: None,
                moe: None,
                hybrid_layers: None,
                tie_word_embeddings: false,
                grow_vocab: false,
                vision: None,
                audio: None,
                gdn: None,
                qwen35_attention: None,
                hadamard: None,
                block: Default::default(),
                quant_formats: Vec::new(),
            },
            "f16",
        );
        assert_eq!(config2.dtype(), "f16");
    }
}
