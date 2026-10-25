//! A model's config and tokenizer, read with no weights.

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::Gguf;
use boostr::{DType, Runtime};
use splintr::AnyTokenizer;

use super::gguf::gguf_tokenizer;
use super::gguf_config::config_from_gguf_metadata;
use super::safetensors::read_safetensors_config;
use crate::config::BlazrConfig;
use crate::loader::{detect_model_source, ModelFormat};
use crate::tokenizer::checkpoint_tokenizer;

/// The config and tokenizer of the GGUF file or SafeTensors checkpoint at
/// `path`, as a load on runtime `R` resolves them. No tensor is read.
///
/// A GGUF carries its tokenizer in its metadata. A checkpoint directory is
/// matched to its tokenizer by vocabulary size, as a load matches it.
///
/// # Errors
///
/// A path that holds no model, a config that does not parse, or a tokenizer
/// that does not load.
pub fn load_model_meta<R: Runtime<DType = DType>>(
    path: &Path,
) -> Result<(BlazrConfig, AnyTokenizer)> {
    match detect_model_source(path)?.format {
        ModelFormat::Gguf => {
            let mut gguf =
                Gguf::open(path).map_err(|e| anyhow!("Failed to open GGUF file: {e}"))?;
            let config = config_from_gguf_metadata(&mut gguf)?;
            let tokenizer = gguf_tokenizer(&gguf)?;
            Ok((config, tokenizer))
        }
        ModelFormat::SafeTensors => {
            let config = read_safetensors_config::<R>(path)?;
            let tokenizer = checkpoint_tokenizer(path, config.vocab_size())?;
            Ok((config, tokenizer))
        }
    }
}
