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

use crate::config::{fixed_format_dtype, BlazrConfig, DtypeRequest};
use crate::model::chat_template::ModelTemplate;

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

    let mut gguf = Gguf::open(path).map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;
    let config = config_from_gguf_metadata(&mut gguf)?;

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
    load_gguf_with_tokenizer_as(path, device, None)
}

/// [`load_gguf_with_tokenizer`] for a user dtype `request`.
///
/// A GGUF model computes in f32. A request for another dtype logs a
/// warning that names its flag or key, and the model runs f32.
pub fn load_gguf_with_tokenizer_as<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    request: Option<&DtypeRequest>,
) -> Result<(LoadedModel<R>, BlazrConfig, AnyTokenizer)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    fixed_format_dtype("GGUF", DType::F32, request);
    let path = path.as_ref();

    // Memory-mapped where the platform has it: each tensor's file pages
    // leave resident memory once the tensor is copied out.
    let mut gguf =
        Gguf::open_with_mmap(path, true).map_err(|e| anyhow!("Failed to open GGUF file: {}", e))?;
    load_gguf_reader(&mut gguf, device)
}

/// [`load_gguf_with_tokenizer_as`] over the bytes of a whole GGUF file, for
/// a host with no file system such as the browser.
///
/// # Errors
///
/// An error when the bytes are not a GGUF file or the model does not load.
pub fn load_gguf_bytes_with_tokenizer_as<R: Runtime<DType = DType>>(
    bytes: Vec<u8>,
    device: &R::Device,
    request: Option<&DtypeRequest>,
) -> Result<GgufBytesModel<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    fixed_format_dtype("GGUF", DType::F32, request);
    let mut gguf =
        Gguf::from_bytes(bytes).map_err(|e| anyhow!("the bytes are not a GGUF file: {e}"))?;
    let chat_template = super::chat_template_from_metadata(gguf.metadata());
    let (model, config, tokenizer) = load_gguf_reader(&mut gguf, device)?;
    Ok(GgufBytesModel {
        model,
        config,
        tokenizer,
        chat_template,
    })
}

/// A model that [`load_gguf_bytes_with_tokenizer_as`] loaded.
pub struct GgufBytesModel<R: Runtime> {
    pub model: LoadedModel<R>,
    pub config: BlazrConfig,
    pub tokenizer: AnyTokenizer,
    /// The file's own chat template, when it carries one.
    pub chat_template: Option<ModelTemplate>,
}

/// Config, embedded tokenizer and weights of an open GGUF.
fn load_gguf_reader<R: Runtime<DType = DType>>(
    gguf: &mut Gguf,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig, AnyTokenizer)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    let config = config_from_gguf_metadata(gguf)?;
    let vocab = boostr::format::extract_gguf_vocab(gguf.metadata())
        .map_err(|e| anyhow!("Failed to extract GGUF vocabulary: {}", e))?;
    let tokenizer = splintr::from_gguf_vocab(vocab)
        .map_err(|e| anyhow!("Failed to build tokenizer from GGUF vocabulary: {}", e))?;

    let mut var_map = VarMap::<R>::from_gguf_reader(gguf, device)
        .map_err(|e| anyhow!("Failed to load GGUF tensors: {}", e))?;

    tracing::info!("Loaded {} tensors from GGUF", var_map.len());

    let mut vb = VarBuilder::new(&mut var_map, device);

    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config, tokenizer))
}
