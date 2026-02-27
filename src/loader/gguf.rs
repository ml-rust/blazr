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
    R::Client: TensorOps<R>,
{
    let path = path.as_ref();

    let gguf = Gguf::open(path).map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;
    let config = config_from_gguf_metadata(&gguf)?;

    // Load all tensors (names auto-mapped from GGUF to HF convention)
    let var_map = VarMap::<R>::from_gguf(path, device)
        .map_err(|e| anyhow!("Failed to load GGUF tensors: {}", e))?;

    tracing::info!("Loaded {} tensors from GGUF", var_map.len());

    let var_map_ref: &'static mut VarMap<R> = Box::leak(Box::new(var_map));
    let mut vb = VarBuilder::new(var_map_ref, device);

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
    R::Client: TensorOps<R>,
{
    let path = path.as_ref();

    let var_map = VarMap::<R>::from_gguf(path, device)
        .map_err(|e| anyhow!("Failed to load GGUF tensors: {}", e))?;

    let var_map_ref: &'static mut VarMap<R> = Box::leak(Box::new(var_map));
    let mut vb = VarBuilder::new(var_map_ref, device);

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
    R::Client: TensorOps<R>,
{
    let path = path.as_ref();

    let gguf = Gguf::open_with_mmap(path, false)
        .map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;

    let config = config_from_gguf_metadata(&gguf)?;
    let tokenizer = crate::tokenizer::GgufTokenizer::from_gguf(&gguf)?;

    let var_map = VarMap::<R>::from_gguf(path, device)
        .map_err(|e| anyhow!("Failed to load GGUF tensors: {}", e))?;

    tracing::info!("Loaded {} tensors from GGUF", var_map.len());

    let var_map_ref: &'static mut VarMap<R> = Box::leak(Box::new(var_map));
    let mut vb = VarBuilder::new(var_map_ref, device);

    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config, tokenizer))
}

/// Create BlazrConfig from GGUF metadata
fn config_from_gguf_metadata(gguf: &Gguf) -> Result<BlazrConfig> {
    use boostr::model::{AttentionConfig, SsmConfig};

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

    let hidden_size = metadata
        .embedding_length()
        .ok_or_else(|| anyhow!("GGUF missing {}.embedding_length", arch))?
        .try_into()?;

    let num_layers = metadata
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
        "mamba" | "mamba2" => "mamba2",
        "mamba3" => "mamba3",
        _ => "llama",
    };

    let num_heads = metadata
        .get_u32(&format!("{}.attention.head_count", arch))
        .map(|v| v as usize)
        .unwrap_or(32);
    let num_kv_heads = metadata
        .get_u32(&format!("{}.attention.head_count_kv", arch))
        .map(|v| v as usize);
    let head_dim = if num_heads > 0 {
        Some(hidden_size / num_heads)
    } else {
        None
    };
    let rope_theta = metadata
        .get_f32(&format!("{}.rope.freq_base", arch))
        .unwrap_or(10000.0);

    let attention = AttentionConfig {
        num_heads,
        num_kv_heads,
        head_dim,
        rope_theta: rope_theta as f32,
        rope_scaling: None,
        kv_latent_dim: None,
        q_latent_dim: None,
        d_rope: None,
        sliding_window: None,
    };

    let ssm = if model_type == "mamba2" || model_type == "mamba3" {
        Some(SsmConfig {
            variant: model_type.to_string(),
            num_heads,
            head_dim: 64,
            state_size: 64,
            chunk_size: 256,
            n_groups: 1,
            conv_kernel: 4,
            expand: 2,
            complex_rope: None,
            mimo_rank: None,
            use_conv: None,
        })
    } else {
        None
    };

    let intermediate_size = metadata
        .get_u32(&format!("{}.feed_forward_length", arch))
        .map(|v| v as usize);

    let rms_norm_eps = metadata
        .get_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch))
        .map(|v| v as f64)
        .unwrap_or(1e-5);

    let model_config = UniversalConfig {
        model_type: model_type.to_string(),
        vocab_size,
        hidden_size,
        num_layers,
        max_seq_len,
        intermediate_size,
        rms_norm_eps,
        attention: Some(attention),
        ssm,
        moe: None,
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

    Ok(GgufInfo {
        architecture: arch.to_string(),
        vocab_size: metadata.get_u32("general.vocab_size").map(|v| v as usize),
        hidden_size: metadata.embedding_length().map(|v| v as usize),
        num_layers: metadata.block_count().map(|v| v as usize),
        num_heads: metadata
            .get_u32(&format!("{}.attention.head_count", arch))
            .map(|v| v as usize),
        num_kv_heads: metadata
            .get_u32(&format!("{}.attention.head_count_kv", arch))
            .map(|v| v as usize),
        context_length: metadata.context_length().map(|v| v as usize),
        quantization_type: detect_quantization_type(&gguf),
        file_size_bytes: std::fs::metadata(path.as_ref()).map(|m| m.len()).ok(),
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
