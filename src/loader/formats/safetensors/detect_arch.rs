//! Architecture detection from SafeTensors tensor names and shapes

use anyhow::Result;

use boostr::format::SafeTensorsLoader;

use crate::model::detect::DetectedConfig;

/// Detect architecture from SafeTensorsLoader
pub fn detect_architecture_from_loader(loader: &SafeTensorsLoader) -> Result<DetectedConfig> {
    let tensor_names = loader.tensor_names();

    let mut config = crate::model::detect::detect_architecture_from_names(&tensor_names)?;

    let format = config.format;
    let prefix = match format {
        crate::model::detect::ModelFormat::HuggingFace => "model.",
        crate::model::detect::ModelFormat::Oxidizr => "",
    };

    // Detect hidden_size and vocab_size from embedding layer
    let embed_key = format!("{}embed_tokens.weight", prefix);
    if let Ok(info) = loader.tensor_info(&embed_key) {
        if info.shape.len() == 2 {
            config.vocab_size = info.shape[0];
            config.hidden_size = info.shape[1];
        }
    }

    // Detect intermediate_size from MLP if present
    let mlp_gate_key = format!("{}layers.0.mlp.gate_proj.weight", prefix);
    if let Ok(info) = loader.tensor_info(&mlp_gate_key) {
        if info.shape.len() == 2 {
            config.intermediate_size = Some(info.shape[0]);
        }
    }

    // Detect number of attention heads from q_proj if present
    let q_proj_key = format!("{}layers.0.self_attn.q_proj.weight", prefix);
    if let Ok(info) = loader.tensor_info(&q_proj_key) {
        if info.shape.len() == 2 && config.hidden_size > 0 {
            let head_dim = 128; // Default head dimension for most models
            config.num_attention_heads = Some(info.shape[0] / head_dim);
            config.head_dim = Some(head_dim);
        }
    }

    // Detect KV heads from k_proj
    let k_proj_key = format!("{}layers.0.self_attn.k_proj.weight", prefix);
    if let Ok(info) = loader.tensor_info(&k_proj_key) {
        if info.shape.len() == 2 {
            let head_dim = config.head_dim.unwrap_or(128);
            config.num_kv_heads = Some(info.shape[0] / head_dim);
        }
    }

    Ok(config)
}

/// Detect AWQ quantization from tensor names and optional quant_config.json
pub fn detect_awq(loader: &SafeTensorsLoader, model_dir: Option<&std::path::Path>) -> bool {
    // Check quant_config.json first
    if let Some(dir) = model_dir {
        let quant_config_path = dir.join("quant_config.json");
        if quant_config_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&quant_config_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(method) = json.get("quant_method").and_then(|v| v.as_str()) {
                        return method.eq_ignore_ascii_case("awq");
                    }
                }
            }
        }

        // Also check config.json for quantization_config
        let config_path = dir.join("config.json");
        if config_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&config_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(qc) = json.get("quantization_config") {
                        if let Some(method) = qc.get("quant_method").and_then(|v| v.as_str()) {
                            return method.eq_ignore_ascii_case("awq");
                        }
                    }
                }
            }
        }
    }

    // Fallback: check tensor names for AWQ patterns (qweight/qzeros/scales triplets)
    let names = loader.tensor_names();
    names.iter().any(|n| n.ends_with(".qweight"))
}

/// Detect GPTQ quantization from tensor names and optional quantize_config.json
pub fn detect_gptq(loader: &SafeTensorsLoader, model_dir: Option<&std::path::Path>) -> bool {
    // Check quantize_config.json first (GPTQ standard)
    if let Some(dir) = model_dir {
        for config_name in &["quantize_config.json", "quant_config.json"] {
            let path = dir.join(config_name);
            if path.exists() {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(method) = json.get("quant_method").and_then(|v| v.as_str()) {
                            if method.eq_ignore_ascii_case("gptq") {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        // Also check config.json for quantization_config
        let config_path = dir.join("config.json");
        if config_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&config_path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(qc) = json.get("quantization_config") {
                        if let Some(method) = qc.get("quant_method").and_then(|v| v.as_str()) {
                            return method.eq_ignore_ascii_case("gptq");
                        }
                    }
                }
            }
        }
    }

    // Fallback: check for g_idx tensors (GPTQ-specific, AWQ doesn't have these)
    let names = loader.tensor_names();
    names.iter().any(|n| n.ends_with(".g_idx"))
}

/// Read GPTQ group_size from quantize_config.json or config.json
pub fn read_gptq_group_size(model_dir: &std::path::Path) -> Result<usize> {
    for config_name in &["quantize_config.json", "quant_config.json"] {
        let path = model_dir.join(config_name);
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(gs) = json.get("group_size").and_then(|v| v.as_u64()) {
                        return Ok(gs as usize);
                    }
                }
            }
        }
    }

    // Try config.json quantization_config
    let config_path = model_dir.join("config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(qc) = json.get("quantization_config") {
                    if let Some(gs) = qc.get("group_size").and_then(|v| v.as_u64()) {
                        return Ok(gs as usize);
                    }
                }
            }
        }
    }

    // Default group_size for GPTQ
    Ok(128)
}

/// Read AWQ group_size from quant_config.json or config.json
pub fn read_awq_group_size(model_dir: &std::path::Path) -> Result<usize> {
    // Try quant_config.json first
    let quant_config_path = model_dir.join("quant_config.json");
    if quant_config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&quant_config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(gs) = json.get("group_size").and_then(|v| v.as_u64()) {
                    return Ok(gs as usize);
                }
            }
        }
    }

    // Try config.json quantization_config
    let config_path = model_dir.join("config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(qc) = json.get("quantization_config") {
                    if let Some(gs) = qc.get("group_size").and_then(|v| v.as_u64()) {
                        return Ok(gs as usize);
                    }
                }
            }
        }
    }

    // Default group_size for AWQ
    Ok(128)
}

#[cfg(test)]
mod tests {
    use crate::model::detect::{detect_architecture_from_names, LayerType, ModelFormat};

    fn hf_transformer_names(num_layers: usize) -> Vec<String> {
        let mut names = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
            "lm_head.weight".into(),
        ];
        for i in 0..num_layers {
            let p = format!("model.layers.{}.", i);
            names.push(format!("{}self_attn.q_proj.weight", p));
            names.push(format!("{}self_attn.k_proj.weight", p));
            names.push(format!("{}self_attn.v_proj.weight", p));
            names.push(format!("{}self_attn.o_proj.weight", p));
            names.push(format!("{}mlp.gate_proj.weight", p));
            names.push(format!("{}mlp.up_proj.weight", p));
            names.push(format!("{}mlp.down_proj.weight", p));
            names.push(format!("{}input_layernorm.weight", p));
            names.push(format!("{}post_attention_layernorm.weight", p));
        }
        names
    }

    fn hf_mla_moe_names(num_layers: usize) -> Vec<String> {
        let mut names = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
            "lm_head.weight".into(),
        ];
        for i in 0..num_layers {
            let p = format!("model.layers.{}.", i);
            names.push(format!("{}self_attn.w_dkv.weight", p));
            names.push(format!("{}self_attn.w_q.weight", p));
            names.push(format!("{}self_attn.w_o.weight", p));
            names.push(format!("{}moe.gate.weight", p));
            names.push(format!("{}moe.experts.0.up_proj.weight", p));
            names.push(format!("{}moe.experts.0.down_proj.weight", p));
            names.push(format!("{}input_layernorm.weight", p));
        }
        names
    }

    #[test]
    fn test_hf_llama_32_layers() {
        let names = hf_transformer_names(32);
        let config = detect_architecture_from_names(&names).unwrap();
        assert_eq!(config.format, ModelFormat::HuggingFace);
        assert_eq!(config.num_layers, 32);
        assert!(!config.tie_word_embeddings);
        assert!(config
            .layer_types
            .iter()
            .all(|&t| t == LayerType::StandardTransformer));
    }

    #[test]
    fn test_hf_small_model_2_layers() {
        let names = hf_transformer_names(2);
        let config = detect_architecture_from_names(&names).unwrap();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.format, ModelFormat::HuggingFace);
    }

    #[test]
    fn test_hf_tied_embeddings() {
        let mut names = hf_transformer_names(4);
        names.retain(|n| n != "lm_head.weight");
        let config = detect_architecture_from_names(&names).unwrap();
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_hf_deepseek_mla_moe() {
        let names = hf_mla_moe_names(8);
        let config = detect_architecture_from_names(&names).unwrap();
        assert_eq!(config.format, ModelFormat::HuggingFace);
        assert_eq!(config.num_layers, 8);
        assert!(config
            .layer_types
            .iter()
            .all(|&t| t == LayerType::MlaWithMoe));
    }

    #[test]
    fn test_hf_hybrid_transformer_and_mamba() {
        let mut names = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
            "lm_head.weight".into(),
        ];
        for suffix in &[
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "mlp.gate_proj.weight",
        ] {
            names.push(format!("model.layers.0.{}", suffix));
        }
        for suffix in &["mamba2.mixer.A_log", "mamba2.mixer.conv1d.weight"] {
            names.push(format!("model.layers.1.{}", suffix));
        }
        let config = detect_architecture_from_names(&names).unwrap();
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.layer_types[0], LayerType::StandardTransformer);
        assert_eq!(config.layer_types[1], LayerType::Mamba2);
    }

    #[test]
    fn test_hf_no_layers_errors() {
        let names: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "model.norm.weight".into(),
        ];
        assert!(detect_architecture_from_names(&names).is_err());
    }
}
