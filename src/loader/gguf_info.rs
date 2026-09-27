//! GGUF header summary for `blazr info` and `blazr list`.
//!
//! Reads metadata only. No tensor is loaded.

use std::path::Path;

use anyhow::{Result, anyhow};

use boostr::format::Gguf;

/// Get metadata from a GGUF file without loading the full model
pub fn get_gguf_info<P: AsRef<Path>>(path: P) -> Result<GgufInfo> {
    let gguf = Gguf::open(path.as_ref()).map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;

    let metadata = gguf.metadata();

    let arch = metadata.architecture().unwrap_or("llama");
    let is_ssm = matches!(arch, "mamba" | "mamba2" | "mamba3");
    let is_gdn = arch == "qwen35";
    let num_layers = metadata.block_count().map(|v| v as usize);

    // qwen35: every `full_attention_interval`-th layer is attention, the rest
    // GDN. Same split as boostr's `HybridConfig::from_full_attention_interval`.
    let attention_layers = if is_gdn {
        match (
            num_layers,
            metadata.get_u32(&format!("{arch}.full_attention_interval")),
        ) {
            (Some(layers), Some(interval)) if interval > 0 => Some(layers / interval as usize),
            (Some(_), Some(_)) => Some(0),
            _ => None,
        }
    } else {
        None
    };
    let gdn_layers = match (num_layers, attention_layers) {
        (Some(layers), Some(attention)) => Some(layers - attention),
        _ => None,
    };

    Ok(GgufInfo {
        architecture: arch.to_string(),
        vocab_size: metadata.get_u32("general.vocab_size").map(|v| v as usize),
        hidden_size: metadata.embedding_length().map(|v| v as usize),
        num_layers,
        num_heads: metadata
            .get_u32(&format!("{arch}.attention.head_count"))
            .map(|v| v as usize),
        num_kv_heads: metadata
            .get_u32(&format!("{arch}.attention.head_count_kv"))
            .map(|v| v as usize),
        context_length: metadata.context_length().map(|v| v as usize),
        quantization_type: detect_quantization_type(&gguf),
        file_size_bytes: std::fs::metadata(path.as_ref()).map(|m| m.len()).ok(),
        is_mla: metadata
            .get_u32(&format!("{arch}.attention.kv_lora_rank"))
            .is_some(),
        is_moe: metadata.get_u32(&format!("{arch}.expert_count")).is_some(),
        is_ssm,
        ssm_state_size: if is_ssm || is_gdn {
            metadata
                .get_u32(&format!("{arch}.ssm.state_size"))
                .map(|v| v as usize)
        } else {
            None
        },
        is_gdn,
        attention_layers,
        gdn_layers,
    })
}

/// Information about a GGUF file
#[derive(Debug, Clone)]
pub struct GgufInfo {
    pub architecture: String,
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub num_heads: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub context_length: Option<usize>,
    pub quantization_type: String,
    pub file_size_bytes: Option<u64>,
    pub is_mla: bool,
    pub is_moe: bool,
    pub is_ssm: bool,
    pub ssm_state_size: Option<usize>,
    /// Gated DeltaNet hybrid (`qwen35`).
    pub is_gdn: bool,
    /// Layers that read the KV cache. `Some` only when `is_gdn`.
    pub attention_layers: Option<usize>,
    /// Layers that read the GDN state. `Some` only when `is_gdn`.
    pub gdn_layers: Option<usize>,
}

/// Detect the primary quantization type from tensor types
fn detect_quantization_type(gguf: &Gguf) -> String {
    use std::collections::HashMap;

    let mut type_counts: HashMap<String, usize> = HashMap::new();

    for name in gguf.tensor_names() {
        if let Ok(info) = gguf.tensor_info(name) {
            let type_name = format!("{:?}", info.ggml_type);
            *type_counts.entry(type_name).or_insert(0) += 1;
        }
    }

    type_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(t, _)| t)
        .unwrap_or_else(|| "Unknown".to_string())
}
