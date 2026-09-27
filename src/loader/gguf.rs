//! GGUF model loading
//!
//! Wraps boostr's GGUF loading functionality.

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::Gguf;
use boostr::model::{LoadedModel, UniversalConfig};
use boostr::ops::TensorOps;
use boostr::{DType, Runtime, VarBuilder, VarMap};
use splintr::AnyTokenizer;

use crate::config::BlazrConfig;

use super::gguf_config::config_from_gguf_metadata;

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
) -> Result<(LoadedModel<R>, BlazrConfig, AnyTokenizer)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    let path = path.as_ref();

    let gguf = Gguf::open_with_mmap(path, false)
        .map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;

    let config = config_from_gguf_metadata(&gguf)?;
    let vocab = boostr::format::extract_gguf_vocab(gguf.metadata())
        .map_err(|e| anyhow!("Failed to extract GGUF vocabulary: {}", e))?;
    let tokenizer = splintr::from_gguf_vocab(vocab)
        .map_err(|e| anyhow!("Failed to build tokenizer from GGUF vocabulary: {}", e))?;

    let mut var_map = VarMap::<R>::from_gguf(path, device)
        .map_err(|e| anyhow!("Failed to load GGUF tensors: {}", e))?;

    tracing::info!("Loaded {} tensors from GGUF", var_map.len());

    let mut vb = VarBuilder::new(&mut var_map, device);

    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config, tokenizer))
}
