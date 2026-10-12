//! A SafeTensors model from bytes in memory, for a host with no file system.

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::model::LoadedModel;
use boostr::ops::TensorOps;
use boostr::{DType, Runtime, VarBuilder};
use splintr::AnyTokenizer;

use crate::config::{compute_dtype, BlazrConfig, DtypeRequest};
use crate::tokenizer::{apply_bos_files, from_vocab_size, CheckpointTokenizerFiles};

use super::config::{config_from_text, torch_dtype_of};
use super::detect_arch::{detect_architecture_from_loader, detect_awq, detect_gptq};
use super::regular::load_tensors_from_loader;

/// One SafeTensors checkpoint held in memory.
#[derive(Debug, Clone, Default)]
pub struct SafeTensorsBytes {
    /// The bytes of one `.safetensors` file. Sharded checkpoints are not
    /// supported here.
    pub weights: Vec<u8>,
    /// The text of `config.json`. Without it the shape comes from the
    /// tensor names and shapes alone.
    pub config_json: Option<String>,
    /// The text of `tokenizer.json`. When set, it is the tokenizer, and it
    /// sets the BOS rule.
    pub tokenizer_json: Option<String>,
    /// The text of `tokenizer_config.json`. It sets only the BOS rule.
    pub tokenizer_config_json: Option<String>,
}

fn parse_json(text: Option<&str>, name: &str) -> Result<Option<serde_json::Value>> {
    text.map(|text| {
        serde_json::from_str(text).map_err(|e| anyhow!("{name} is not valid JSON: {e}"))
    })
    .transpose()
}

/// Load a regular SafeTensors checkpoint from memory.
///
/// The tokenizer is `tokenizer_json` when set, else the bundled vocabulary
/// the config's `vocab_size` matches. The JSON files set its BOS rule. GPTQ and AWQ checkpoints are
/// refused.
///
/// # Errors
///
/// An error when the bytes are not a SafeTensors file, a JSON text does not
/// parse, no bundled vocabulary matches, or the model does not load.
pub fn load_safetensors_bytes<R: Runtime<DType = DType>>(
    input: SafeTensorsBytes,
    device: &R::Device,
    request: Option<&DtypeRequest>,
) -> Result<(LoadedModel<R>, BlazrConfig, AnyTokenizer)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let SafeTensorsBytes {
        weights,
        config_json,
        tokenizer_json,
        tokenizer_config_json,
    } = input;
    let files = CheckpointTokenizerFiles {
        tokenizer_config: parse_json(tokenizer_config_json.as_deref(), "tokenizer_config.json")?,
        tokenizer_json: parse_json(tokenizer_json.as_deref(), "tokenizer.json")?,
        config: parse_json(config_json.as_deref(), "config.json")?,
    };

    let mut loader = SafeTensorsLoader::from_bytes(weights)
        .map_err(|e| anyhow!("the bytes are not a SafeTensors file: {e}"))?;
    if detect_gptq(&loader, None) || detect_awq(&loader, None) {
        return Err(anyhow!(
            "GPTQ and AWQ checkpoints do not load from bytes. Pass a regular SafeTensors \
             checkpoint or a GGUF file"
        ));
    }
    let dtype = compute_dtype::<R>(request, config_json.as_deref().and_then(torch_dtype_of));
    let config = config_from_text(
        config_json.as_deref(),
        || detect_architecture_from_loader(&loader),
        dtype,
    )?;

    let mut var_map = load_tensors_from_loader::<R>(&mut loader, device, dtype)?;
    boostr::format::restructure_hf_checkpoint(&mut var_map, &config.model, device)
        .map_err(|e| anyhow!("{e}"))?;
    let mut vb = VarBuilder::new(&mut var_map, device);
    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {e}"))?;

    let base = match tokenizer_json.as_deref() {
        Some(text) => splintr::from_json_bytes(text.as_bytes())
            .map_err(|e| anyhow!("tokenizer.json does not load as a tokenizer: {e}"))?,
        None => from_vocab_size(config.vocab_size())?,
    };
    let tokenizer = apply_bos_files(base, &files)?;
    Ok((model, config, tokenizer))
}
