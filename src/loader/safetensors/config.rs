//! Configuration loading and creation from detected architecture

use std::path::Path;

use anyhow::Result;

use boostr::model::{
    AttentionConfig, HuggingFaceConfig, HybridConfig, MoeConfig, SsmConfig, UniversalConfig,
};

use boostr::{DType, Runtime};

use crate::config::{
    compute_dtype, dtype_name, torch_dtype, BlazrConfig, DtypeRequest, InferenceConfig,
};
use crate::model::detect::DetectedConfig;

/// The dtype a model's `config.json` names in `torch_dtype`. `None` when
/// the file, the key or a known value is missing.
pub fn model_torch_dtype(dir: &Path) -> Option<DType> {
    let content = std::fs::read_to_string(dir.join("config.json")).ok()?;
    torch_dtype_of(&content)
}

/// The `torch_dtype` that the text of a `config.json` names.
pub fn torch_dtype_of(config_json: &str) -> Option<DType> {
    let json = serde_json::from_str::<serde_json::Value>(config_json).ok()?;
    json.get("torch_dtype")
        .and_then(|v| v.as_str())
        .and_then(torch_dtype)
}

/// The compute dtype of a regular SafeTensors checkpoint in `dir` on
/// runtime `R`: `request`, then `torch_dtype`, then the backend default.
pub fn checkpoint_compute_dtype<R: Runtime<DType = DType>>(
    dir: Option<&Path>,
    request: Option<&DtypeRequest>,
) -> DType {
    compute_dtype::<R>(request, dir.and_then(model_torch_dtype))
}

/// The `inference` section of a model's `config.json`, read before any
/// weight loads. `None` when the file is missing or is not a blazr config.
///
/// `path` is the model directory or a file inside it.
pub fn model_config_inference(path: &Path) -> Option<InferenceConfig> {
    let dir = if path.is_dir() { path } else { path.parent()? };
    BlazrConfig::from_json(dir.join("config.json"))
        .ok()
        .map(|config| config.inference)
}

/// Load config or create from detected architecture, with `dtype` as the
/// compute dtype the caller resolved.
pub fn load_or_create_config(
    dir: &Path,
    detected: &DetectedConfig,
    dtype: DType,
) -> Result<BlazrConfig> {
    let content = std::fs::read_to_string(dir.join("config.json")).ok();
    config_from_text(content.as_deref(), detected, dtype)
}

/// [`load_or_create_config`] over the text of a `config.json`. The text is
/// a blazr config, a boostr `UniversalConfig`, or a Hugging Face config.
/// `None`, or text that is none of them, falls back to `detected`.
pub fn config_from_text(
    config_json: Option<&str>,
    detected: &DetectedConfig,
    dtype: DType,
) -> Result<BlazrConfig> {
    let dtype = dtype_name(dtype)?;
    if let Some(content) = config_json {
        if let Ok(mut config) = serde_json::from_str::<BlazrConfig>(content) {
            config.inference.dtype = dtype.to_string();
            return Ok(config);
        }
        if let Ok(model_config) = serde_json::from_str::<UniversalConfig>(content) {
            return Ok(BlazrConfig::from_universal_with_dtype(model_config, dtype));
        }
        if let Ok(hf_config) = HuggingFaceConfig::from_json(content) {
            return Ok(BlazrConfig::from_universal_with_dtype(
                hf_config.to_universal(),
                dtype,
            ));
        }
    }
    Ok(config_from_detected_with_dtype(detected, dtype))
}

/// Create BlazrConfig from detected architecture, with `dtype` as the
/// compute dtype the caller resolved.
pub fn config_from_detected(detected: &DetectedConfig, dtype: DType) -> Result<BlazrConfig> {
    Ok(config_from_detected_with_dtype(
        detected,
        dtype_name(dtype)?,
    ))
}

/// Create BlazrConfig from detected architecture with specified dtype
fn config_from_detected_with_dtype(detected: &DetectedConfig, dtype: &str) -> BlazrConfig {
    use crate::model::Config;

    let legacy_config = Config::from_detected(detected.clone());
    let model = detected_to_universal(detected, &legacy_config);
    BlazrConfig::from_universal_with_dtype(model, dtype)
}

/// Convert DetectedConfig + legacy Config to UniversalConfig
fn detected_to_universal(
    detected: &DetectedConfig,
    legacy: &crate::model::Config,
) -> UniversalConfig {
    use crate::model::detect::LayerType;

    let model_type = determine_model_type(&detected.layer_types);

    let rope_scaling = legacy
        .rope_scaling
        .as_ref()
        .map(|rs| boostr::model::RopeScalingConfig {
            scaling_type: rs.rope_type.clone().unwrap_or_else(|| "llama3".to_string()),
            factor: rs.factor as f32,
            original_max_position_embeddings: Some(rs.original_max_position_embeddings),
            low_freq_factor: Some(rs.low_freq_factor as f32),
            high_freq_factor: Some(rs.high_freq_factor as f32),
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
            short_factor: rs.short_factor.clone(),
            long_factor: rs.long_factor.clone(),
        });

    let attention = if detected.num_attention_heads.is_some() || detected.num_kv_heads.is_some() {
        Some(AttentionConfig {
            num_heads: detected.num_attention_heads.unwrap_or(32),
            num_kv_heads: detected.num_kv_heads,
            head_dim: detected.head_dim,
            rope_theta: legacy.rope_theta as f32,
            rope_scaling,
            kv_latent_dim: detected.kv_latent_dim,
            q_latent_dim: detected.q_latent_dim,
            d_rope: detected.d_rope,
            sliding_window: None,
            use_alibi: false,
        })
    } else {
        None
    };

    let ssm = if detected.mamba2_num_heads.is_some() {
        Some(SsmConfig {
            variant: if detected.mamba3_enabled.unwrap_or(false) {
                "mamba3"
            } else {
                "mamba2"
            }
            .to_string(),
            num_heads: detected.mamba2_num_heads.unwrap_or(64),
            head_dim: detected.mamba2_head_dim.unwrap_or(64),
            state_size: detected.mamba2_state_size.unwrap_or(64),
            chunk_size: legacy.mamba2_chunk_size,
            n_groups: legacy.mamba2_n_groups,
            conv_kernel: detected.mamba2_conv_kernel.unwrap_or(4),
            expand: detected.mamba2_expand.unwrap_or(2),
            complex_rope: detected.mamba3_complex_rope,
            mimo_rank: detected.mamba3_mimo_rank,
            use_conv: detected.mamba3_use_conv,
        })
    } else {
        None
    };

    let moe = if detected.num_experts.is_some() {
        Some(MoeConfig {
            num_experts: detected.num_experts.unwrap_or(8),
            experts_per_tok: legacy.experts_per_tok,
            shared_expert: Some(detected.shared_expert_enabled),
            intermediate_size: detected.intermediate_size,
            load_balance_alpha: 0.01,
            z_loss_alpha: 1e-3,
        })
    } else {
        None
    };

    let hybrid_layers = if has_mixed_layer_types(&detected.layer_types) {
        let ssm_layers: Vec<usize> = detected
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, lt)| matches!(lt, LayerType::Mamba2 | LayerType::Mamba3))
            .map(|(i, _)| i)
            .collect();

        let attention_layers: Vec<usize> = detected
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, lt)| !matches!(lt, LayerType::Mamba2 | LayerType::Mamba3))
            .map(|(i, _)| i)
            .collect();

        Some(HybridConfig {
            ssm_layers,
            attention_layers,
        })
    } else {
        None
    };

    UniversalConfig {
        model_type,
        vocab_size: detected.vocab_size,
        hidden_size: detected.hidden_size,
        num_layers: detected.num_layers,
        max_seq_len: legacy.max_seq_len,
        intermediate_size: detected.intermediate_size,
        rms_norm_eps: legacy.rms_norm_eps,
        attention,
        ssm,
        moe,
        hybrid_layers,
        tie_word_embeddings: false,
        grow_vocab: false,
        vision: None,
        audio: None,
        gdn: None,
        qwen35_attention: None,
        hadamard: None,
        quant_formats: Vec::new(),
    }
}

/// Determine model type from layer types
fn determine_model_type(layer_types: &[crate::model::detect::LayerType]) -> String {
    use crate::model::detect::LayerType;

    if layer_types.is_empty() {
        return "llama".to_string();
    }

    let has_mamba = layer_types
        .iter()
        .any(|lt| matches!(lt, LayerType::Mamba2 | LayerType::Mamba3));
    let has_attention = layer_types
        .iter()
        .any(|lt| !matches!(lt, LayerType::Mamba2 | LayerType::Mamba3));

    if has_mamba && has_attention {
        "hybrid".to_string()
    } else if has_mamba {
        if layer_types.iter().any(|lt| matches!(lt, LayerType::Mamba3)) {
            "mamba3".to_string()
        } else {
            "mamba2".to_string()
        }
    } else if layer_types
        .iter()
        .any(|lt| matches!(lt, LayerType::MlaWithMoe | LayerType::MlaWithMlp))
    {
        "hybrid".to_string()
    } else {
        "llama".to_string()
    }
}

/// Check if layer types are mixed (requires hybrid config)
fn has_mixed_layer_types(layer_types: &[crate::model::detect::LayerType]) -> bool {
    use crate::model::detect::LayerType;

    if layer_types.is_empty() {
        return false;
    }

    let first_is_mamba = matches!(layer_types[0], LayerType::Mamba2 | LayerType::Mamba3);

    layer_types.iter().any(|lt| {
        let is_mamba = matches!(lt, LayerType::Mamba2 | LayerType::Mamba3);
        is_mamba != first_is_mamba
    })
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use crate::config::DtypeSource;
    use boostr::CpuRuntime;

    /// A model directory whose `config.json` names `torch_dtype`.
    fn model_dir(torch_dtype: Option<&str>) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let field = torch_dtype
            .map(|d| format!(",\n  \"torch_dtype\": \"{d}\""))
            .unwrap_or_default();
        let json = format!(
            "{{\n  \"architectures\": [\"LlamaForCausalLM\"],\n  \"vocab_size\": 32,\n  \
             \"hidden_size\": 8,\n  \"num_hidden_layers\": 1,\n  \
             \"num_attention_heads\": 2{field}\n}}"
        );
        std::fs::write(dir.path().join("config.json"), json).unwrap();
        dir
    }

    #[test]
    fn torch_dtype_is_the_model_step() {
        let dir = model_dir(Some("float32"));
        assert_eq!(model_torch_dtype(dir.path()), Some(DType::F32));
        assert_eq!(model_torch_dtype(model_dir(None).path()), None);
        assert_eq!(
            checkpoint_compute_dtype::<CpuRuntime>(Some(dir.path()), None),
            DType::F32
        );
        // No torch_dtype: the backend default.
        assert_eq!(
            checkpoint_compute_dtype::<CpuRuntime>(Some(model_dir(None).path()), None),
            DType::F32
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn request_beats_torch_dtype() {
        let dir = model_dir(Some("float32"));
        let request = DtypeRequest {
            dtype: DType::BF16,
            source: DtypeSource::File,
        };
        assert_eq!(
            checkpoint_compute_dtype::<CpuRuntime>(Some(dir.path()), Some(&request)),
            DType::BF16
        );
        let bf16 = model_dir(Some("bfloat16"));
        assert_eq!(
            checkpoint_compute_dtype::<CpuRuntime>(Some(bf16.path()), None),
            DType::BF16
        );
    }

    /// The resolved dtype is the config's compute dtype, and the KV cache
    /// dtype follows it.
    #[test]
    fn config_records_the_compute_dtype_for_the_kv_cache() {
        let dir = model_dir(Some("bfloat16"));
        let detected = DetectedConfig::default();
        let config = load_or_create_config(dir.path(), &detected, DType::F32).unwrap();
        assert_eq!(config.dtype(), "f32");
        assert_eq!(config.compute_dtype().unwrap(), DType::F32);
        assert_eq!(config.kv_cache_dtype().unwrap(), DType::F32);
        assert!(load_or_create_config(dir.path(), &detected, DType::I32).is_err());
    }

    #[test]
    fn test_huggingface_config_rope_scaling() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 128256,
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 131072,
            "rope_theta": 500000.0,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192
            }
        }"#;

        let hf_config = HuggingFaceConfig::from_json(json).expect("Failed to parse config");
        let universal = hf_config.to_universal();
        let attn = universal.attention.expect("No attention config");

        assert!(attn.rope_scaling.is_some(), "rope_scaling should be Some");
        let rs = attn.rope_scaling.unwrap();
        assert_eq!(rs.scaling_type, "llama3");
        assert_eq!(rs.factor, 32.0);
        assert_eq!(rs.low_freq_factor, Some(1.0));
        assert_eq!(rs.high_freq_factor, Some(4.0));
    }
}
