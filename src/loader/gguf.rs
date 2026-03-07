//! GGUF model loading
//!
//! Wraps boostr's GGUF loading functionality.

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::Gguf;
use boostr::model::{LoadedModel, UniversalConfig};
use boostr::ops::TensorOps;
use boostr::{DType, Runtime, VarBuilder, VarMap};

use crate::config::BlazrConfig;

/// Load a model from GGUF format
///
/// GGUF files contain both model weights and metadata, so no separate
/// config file is needed. The metadata is used to construct the configuration.
pub fn load_gguf<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    let path = path.as_ref();

    let gguf = Gguf::open(path).map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;
    let config = config_from_gguf_metadata(&gguf)?;

    // Load all tensors (names auto-mapped from GGUF to HF convention)
    let mut var_map = VarMap::<R>::from_gguf(path, device)
        .map_err(|e| anyhow!("Failed to load GGUF tensors: {}", e))?;

    tracing::info!("Loaded {} tensors from GGUF", var_map.len());

    let mut vb = VarBuilder::new(&mut var_map, device);

    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config))
}

/// Load a model from GGUF with explicit configuration
pub fn load_gguf_with_config<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    config: &UniversalConfig,
    device: &R::Device,
) -> Result<LoadedModel<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    let path = path.as_ref();

    let mut var_map = VarMap::<R>::from_gguf(path, device)
        .map_err(|e| anyhow!("Failed to load GGUF tensors: {}", e))?;

    let mut vb = VarBuilder::new(&mut var_map, device);

    let model =
        LoadedModel::load(config, &mut vb).map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok(model)
}

/// Load a model from GGUF format with embedded tokenizer
///
/// This returns the GGUF-embedded tokenizer which uses the exact vocabulary
/// from the GGUF file (SentencePiece format for Llama/Mistral models).
pub fn load_gguf_with_tokenizer<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig, crate::tokenizer::GgufTokenizer)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    let path = path.as_ref();

    let gguf = Gguf::open_with_mmap(path, false)
        .map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;

    let config = config_from_gguf_metadata(&gguf)?;
    let tokenizer = crate::tokenizer::GgufTokenizer::from_gguf(&gguf)?;

    let mut var_map = VarMap::<R>::from_gguf(path, device)
        .map_err(|e| anyhow!("Failed to load GGUF tensors: {}", e))?;

    tracing::info!("Loaded {} tensors from GGUF", var_map.len());

    let mut vb = VarBuilder::new(&mut var_map, device);

    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config, tokenizer))
}

/// Create BlazrConfig from GGUF metadata
fn config_from_gguf_metadata(gguf: &Gguf) -> Result<BlazrConfig> {
    use boostr::model::{AttentionConfig, MoeConfig, SsmConfig};

    let metadata = gguf.metadata();

    // Get architecture first as it's needed for other keys
    let arch = metadata.architecture().unwrap_or("llama");

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
        _ => "llama",
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
            .or_else(|| {
                if num_heads > 0 {
                    Some(hidden_size / num_heads)
                } else {
                    None
                }
            });
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
                .map_or(false, |v| v != 0);

        Some(AttentionConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rope_scaling: None,
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
        let expand = if hidden_size > 0 {
            inner_size / hidden_size
        } else {
            2
        };
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
        tie_word_embeddings: false,
    };

    Ok(BlazrConfig::from_universal_with_dtype(model_config, "f32"))
}

/// Get metadata from a GGUF file without loading the full model
pub fn get_gguf_info<P: AsRef<Path>>(path: P) -> Result<GgufInfo> {
    let gguf = Gguf::open(path.as_ref()).map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;

    let metadata = gguf.metadata();

    let arch = metadata.architecture().unwrap_or("llama");
    let is_ssm = matches!(arch, "mamba" | "mamba2" | "mamba3");

    Ok(GgufInfo {
        architecture: arch.to_string(),
        vocab_size: metadata.get_u32("general.vocab_size").map(|v| v as usize),
        hidden_size: metadata.embedding_length().map(|v| v as usize),
        num_layers: metadata.block_count().map(|v| v as usize),
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
        ssm_state_size: if is_ssm {
            metadata
                .get_u32(&format!("{arch}.ssm.state_size"))
                .map(|v| v as usize)
        } else {
            None
        },
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
