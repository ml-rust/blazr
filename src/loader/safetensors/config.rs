//! Configuration loading and creation from detected architecture

use std::path::Path;

use anyhow::Result;

use boostr::model::{
    AttentionConfig, HuggingFaceConfig, HybridConfig, MoeConfig, SsmConfig, UniversalConfig,
};

use crate::config::BlazrConfig;
use crate::model::detect::DetectedConfig;

/// Detect dtype from config.json's torch_dtype field
pub fn detect_dtype_from_config(dir: &Path) -> &'static str {
    let config_path = dir.join("config.json");
    if let Ok(content) = std::fs::read_to_string(&config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            match json.get("torch_dtype").and_then(|v| v.as_str()) {
                Some("bfloat16") => return "bf16",
                Some("float16") => return "f16",
                Some("float32") => return "f32",
                _ => {}
            }
        }
    }
    "f16" // Default
}

/// Load config or create from detected architecture
pub fn load_or_create_config(dir: &Path, detected: &DetectedConfig) -> Result<BlazrConfig> {
    let config_path = dir.join("config.json");
    let dtype = detect_dtype_from_config(dir);

    if config_path.exists() {
        match BlazrConfig::from_json(&config_path) {
            Ok(mut config) => {
                config.inference.dtype = dtype.to_string();
                return Ok(config);
            }
            Err(_) => {
                if let Ok(content) = std::fs::read_to_string(&config_path) {
                    if let Ok(model_config) = serde_json::from_str::<UniversalConfig>(&content) {
                        return Ok(BlazrConfig::from_universal_with_dtype(model_config, dtype));
                    }
                    if let Ok(hf_config) = HuggingFaceConfig::from_json(&content) {
                        return Ok(BlazrConfig::from_universal_with_dtype(
                            hf_config.to_universal(),
                            dtype,
                        ));
                    }
                }
            }
        }
    }

    Ok(config_from_detected_with_dtype(detected, dtype))
}

/// Create BlazrConfig from detected architecture
pub fn config_from_detected(detected: &DetectedConfig) -> BlazrConfig {
    config_from_detected_with_dtype(detected, "f16")
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
