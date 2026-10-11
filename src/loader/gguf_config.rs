//! `BlazrConfig` from a GGUF header. `qwen35` goes to boostr's key table,
//! every other architecture through the generic blocks here, unknown ones error.

use anyhow::{anyhow, Result};

use boostr::format::Gguf;
use boostr::model::UniversalConfig;

use crate::config::BlazrConfig;

use super::gguf_rope::rope_scaling_from_gguf;

/// `general.architecture` values [`config_from_gguf_metadata`] accepts.
const SUPPORTED_GGUF_ARCHS: &[&str] = &[
    "llama",
    "llama2",
    "llama3",
    "mistral",
    "deepseek",
    "deepseek2",
    "mamba",
    "mamba2",
    "mamba3",
    "falcon",
    "qwen2",
    "phi3",
    "gemma",
    "gemma2",
    "starcoder2",
    "qwen35",
];

/// Create BlazrConfig from GGUF metadata.
///
/// # Errors
///
/// `general.architecture` missing or outside [`SUPPORTED_GGUF_ARCHS`]. An
/// unknown architecture never falls back to llama.
pub(crate) fn config_from_gguf_metadata(gguf: &mut Gguf) -> Result<BlazrConfig> {
    use boostr::model::{AttentionConfig, MoeConfig, SsmConfig};

    // Get architecture first as it's needed for other keys
    let arch_name = gguf
        .metadata()
        .architecture()
        .ok_or_else(|| anyhow!("GGUF missing general.architecture"))?
        .to_string();
    let rope_scaling = rope_scaling_from_gguf(gguf, &arch_name)?;
    let metadata = gguf.metadata();
    let arch = arch_name.as_str();

    // qwen35 (Bonsai 2): GDN + gated attention hybrid. boostr owns the key
    // table, so the generic attention/ssm blocks below never see it.
    if arch == "qwen35" {
        let model_config = boostr::model::qwen35_config_from_gguf(metadata)
            .map_err(|e| anyhow!("Failed to build qwen35 config from GGUF metadata: {}", e))?;
        return Ok(BlazrConfig::from_universal_with_dtype(model_config, "f32"));
    }
    if arch == "qwen35moe" {
        return Err(anyhow!(
            "GGUF architecture 'qwen35moe' is not supported: boostr parses only 'qwen35'"
        ));
    }

    // Extract core parameters from GGUF metadata
    let vocab_size: usize = if let Some(vs) = metadata.get_u32("general.vocab_size") {
        vs as usize
    } else if let Some(tokens) = metadata.get_array("tokenizer.ggml.tokens") {
        tokens.len()
    } else {
        match arch {
            "llama" | "llama2" | "llama3" => 128256,
            "mistral" => 32000,
            _ => 32000,
        }
    };

    let hidden_size: usize = metadata
        .embedding_length()
        .ok_or_else(|| anyhow!("GGUF missing {}.embedding_length", arch))?
        .try_into()?;

    let num_layers: usize = metadata
        .block_count()
        .ok_or_else(|| anyhow!("GGUF missing {}.block_count", arch))?
        .try_into()?;

    let max_seq_len = metadata
        .context_length()
        .map(|v| v as usize)
        .unwrap_or(4096);

    let model_type = match arch {
        "llama" | "llama2" | "llama3" => "llama",
        "mistral" => "mistral",
        "deepseek" | "deepseek2" => "deepseek",
        "mamba" | "mamba2" => "mamba2",
        "mamba3" => "mamba3",
        "falcon" => "falcon",
        "qwen2" => "qwen2",
        "phi3" => "phi3",
        "gemma" | "gemma2" => arch,
        "starcoder2" => "starcoder2",
        other => {
            return Err(anyhow!(
                "Unsupported GGUF architecture '{}'. Supported: {}",
                other,
                SUPPORTED_GGUF_ARCHS.join(", ")
            ));
        }
    };

    let is_ssm = model_type == "mamba2" || model_type == "mamba3";

    let intermediate_size = metadata
        .get_u32(&format!("{arch}.feed_forward_length"))
        .map(|v| v as usize);

    let rms_norm_eps = metadata
        .get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .map(|v| v as f64)
        .unwrap_or(1e-5);

    // --- Attention config (not present for pure SSM models) ---
    let attention = if is_ssm {
        None
    } else {
        let num_heads = metadata
            .get_u32(&format!("{arch}.attention.head_count"))
            .map(|v| v as usize)
            .unwrap_or(32);
        let num_kv_heads = metadata
            .get_u32(&format!("{arch}.attention.head_count_kv"))
            .map(|v| v as usize);
        let head_dim = metadata
            .get_u32(&format!("{arch}.attention.key_length"))
            .map(|v| v as usize)
            .or_else(|| hidden_size.checked_div(num_heads));
        let rope_theta = metadata
            .get_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(10000.0);

        // DeepSeek MLA detection: kv_lora_rank indicates compressed KV
        let kv_latent_dim = metadata
            .get_u32(&format!("{arch}.attention.kv_lora_rank"))
            .map(|v| v as usize);
        let q_latent_dim = metadata
            .get_u32(&format!("{arch}.attention.q_lora_rank"))
            .map(|v| v as usize);
        let d_rope = metadata
            .get_u32(&format!("{arch}.attention.rope_dimension_count"))
            .map(|v| v as usize);

        // ALiBi detection (Falcon v1)
        let use_alibi = model_type == "falcon"
            && metadata
                .get_u32(&format!("{arch}.attention.use_alibi"))
                .is_some_and(|v| v != 0);

        Some(AttentionConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rope_scaling,
            kv_latent_dim,
            q_latent_dim,
            d_rope,
            sliding_window: None,
            use_alibi,
        })
    };

    // --- SSM config (Mamba2/3) ---
    let ssm = if is_ssm {
        let state_size = metadata
            .get_u32(&format!("{arch}.ssm.state_size"))
            .map(|v| v as usize)
            .unwrap_or(64);
        let conv_kernel = metadata
            .get_u32(&format!("{arch}.ssm.conv_kernel"))
            .map(|v| v as usize)
            .unwrap_or(4);
        let inner_size = metadata
            .get_u32(&format!("{arch}.ssm.inner_size"))
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 2);
        // Derive num_heads and head_dim from inner_size
        // Convention: head_dim = 64 (Mamba2 default), num_heads = inner_size / head_dim
        let ssm_head_dim = metadata
            .get_u32(&format!("{arch}.ssm.head_dim"))
            .map(|v| v as usize)
            .unwrap_or(64);
        let ssm_num_heads = inner_size / ssm_head_dim;
        let expand = inner_size.checked_div(hidden_size).unwrap_or(2);
        let n_groups = metadata
            .get_u32(&format!("{arch}.ssm.group_count"))
            .map(|v| v as usize)
            .unwrap_or(1);

        Some(SsmConfig {
            variant: model_type.to_string(),
            num_heads: ssm_num_heads,
            head_dim: ssm_head_dim,
            state_size,
            chunk_size: 256,
            n_groups,
            conv_kernel,
            expand,
            complex_rope: if model_type == "mamba3" {
                Some(true)
            } else {
                None
            },
            mimo_rank: None,
            use_conv: None,
        })
    } else {
        None
    };

    // --- MoE config (DeepSeek, Mixtral, etc.) ---
    let moe = metadata
        .get_u32(&format!("{arch}.expert_count"))
        .map(|num_experts| {
            let experts_per_tok = metadata
                .get_u32(&format!("{arch}.expert_used_count"))
                .map(|v| v as usize)
                .unwrap_or(2);
            MoeConfig {
                num_experts: num_experts as usize,
                experts_per_tok,
                shared_expert: None,
                intermediate_size: None,
                load_balance_alpha: 0.01,
                z_loss_alpha: 1e-3,
            }
        });

    let model_config = UniversalConfig {
        model_type: model_type.to_string(),
        vocab_size,
        hidden_size,
        num_layers,
        max_seq_len,
        intermediate_size,
        rms_norm_eps,
        attention,
        ssm,
        moe,
        hybrid_layers: None,
        // llama.cpp reads `token_embd.weight` as the output head when a file
        // has no `output.weight`.
        tie_word_embeddings: gguf.tensor_info("output.weight").is_err(),
        grow_vocab: false,
        vision: None,
        audio: None,
        gdn: None,
        qwen35_attention: None,
        hadamard: None,
        quant_formats: Vec::new(),
    };

    Ok(BlazrConfig::from_universal_with_dtype(model_config, "f32"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One GGUF metadata value, encoded as the header stores it.
    enum Kv<'a> {
        U32(u32),
        F32(f32),
        Str(&'a str),
        I32Array(&'a [i32]),
    }

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    /// GGUF v3 header with zero tensors and the given key/value pairs.
    fn gguf_bytes(pairs: &[(&str, Kv<'_>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
        for (key, value) in pairs {
            put_str(&mut out, key);
            match value {
                Kv::U32(v) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Kv::F32(v) => {
                    out.extend_from_slice(&6u32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Kv::Str(v) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut out, v);
                }
                Kv::I32Array(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&5u32.to_le_bytes());
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for item in *items {
                        out.extend_from_slice(&item.to_le_bytes());
                    }
                }
            }
        }
        let aligned = out.len().div_ceil(32) * 32;
        out.resize(aligned, 0);
        out
    }

    const SECTIONS: &[i32] = &[11, 11, 10, 0];

    /// Bonsai-2-27B header keys that boostr's `qwen35_config_from_gguf` reads.
    fn bonsai_pairs() -> Vec<(&'static str, Kv<'static>)> {
        vec![
            ("general.architecture", Kv::Str("qwen35")),
            ("qwen35.block_count", Kv::U32(64)),
            ("qwen35.context_length", Kv::U32(262_144)),
            ("qwen35.embedding_length", Kv::U32(5120)),
            ("qwen35.feed_forward_length", Kv::U32(17_408)),
            ("qwen35.attention.head_count", Kv::U32(24)),
            ("qwen35.attention.head_count_kv", Kv::U32(4)),
            ("qwen35.attention.key_length", Kv::U32(256)),
            ("qwen35.attention.value_length", Kv::U32(256)),
            ("qwen35.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
            ("qwen35.rope.dimension_count", Kv::U32(64)),
            ("qwen35.rope.dimension_sections", Kv::I32Array(SECTIONS)),
            ("qwen35.rope.freq_base", Kv::F32(1e7)),
            ("qwen35.ssm.conv_kernel", Kv::U32(4)),
            ("qwen35.ssm.state_size", Kv::U32(128)),
            ("qwen35.ssm.group_count", Kv::U32(16)),
            ("qwen35.ssm.time_step_rank", Kv::U32(48)),
            ("qwen35.ssm.inner_size", Kv::U32(6144)),
            ("qwen35.full_attention_interval", Kv::U32(4)),
            ("qwen35.vocab_size", Kv::U32(248_320)),
        ]
    }

    fn open(pairs: &[(&str, Kv<'_>)]) -> Gguf {
        Gguf::from_bytes(gguf_bytes(pairs)).expect("hand-built GGUF header parses")
    }

    #[test]
    fn qwen35_header_routes_to_boostr_parser() {
        let config = config_from_gguf_metadata(&mut open(&bonsai_pairs())).expect("qwen35 config");
        let model = &config.model;
        assert_eq!(model.model_type, "qwen35");
        assert_eq!(model.num_layers, 64);
        assert_eq!(model.hidden_size, 5120);
        assert_eq!(model.vocab_size, 248_320);
        assert!(model.attention.is_none());
        assert!(model.ssm.is_none());
        assert!(model.gdn.is_some());
        assert!(model.qwen35_attention.is_some());
        let layers = model.hybrid_layers.as_ref().expect("hybrid_layers");
        assert_eq!(layers.attention_layers.len(), 16);
        assert_eq!(layers.ssm_layers.len(), 48);
    }

    #[test]
    fn unknown_architecture_errors_instead_of_llama_fallback() {
        let pairs = [
            ("general.architecture", Kv::Str("not_a_real_arch")),
            ("not_a_real_arch.embedding_length", Kv::U32(64)),
            ("not_a_real_arch.block_count", Kv::U32(2)),
        ];
        let err = config_from_gguf_metadata(&mut open(&pairs)).expect_err("unknown arch");
        let msg = err.to_string();
        assert!(msg.contains("not_a_real_arch"), "{msg}");
        assert!(msg.contains("qwen35"), "{msg}");
        assert!(msg.contains("llama"), "{msg}");
    }

    #[test]
    fn qwen35moe_and_missing_architecture_error() {
        let pairs = [("general.architecture", Kv::Str("qwen35moe"))];
        let err = config_from_gguf_metadata(&mut open(&pairs)).expect_err("qwen35moe");
        assert!(err.to_string().contains("qwen35moe"), "{err}");

        let pairs = [("llama.embedding_length", Kv::U32(64))];
        let err = config_from_gguf_metadata(&mut open(&pairs)).expect_err("missing arch");
        assert!(err.to_string().contains("general.architecture"), "{err}");
    }

    #[test]
    fn known_architecture_still_parses() {
        let pairs = [
            ("general.architecture", Kv::Str("llama")),
            ("llama.embedding_length", Kv::U32(64)),
            ("llama.block_count", Kv::U32(2)),
            ("llama.attention.head_count", Kv::U32(4)),
        ];
        let config = config_from_gguf_metadata(&mut open(&pairs)).expect("llama config");
        assert_eq!(config.model.model_type, "llama");
        assert_eq!(config.model.num_layers, 2);
    }
}
