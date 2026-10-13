//! The load every path from a model path to an executor shares: the
//! scheduler, the embedder API and the in-process commands. Each step
//! reads the disk or uploads to the device.

use std::path::Path;

use anyhow::Result;
use boostr::model::{LoadedModel, ModelClient};
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use splintr::AnyTokenizer;

use crate::config::{BlazrConfig, DtypeRequest};
use crate::engine::Executor;
use crate::loader::{self, ModelFormat, OffloadingOptions};
use crate::model::chat_template::ChatTemplate;
use crate::tokenizer::checkpoint_tokenizer;

/// A loaded model, its config and the tokenizer it decodes with.
pub(crate) type ModelParts<R> = (LoadedModel<R>, BlazrConfig, AnyTokenizer);

/// Read the GGUF file or SafeTensors checkpoint at `path` onto `device`,
/// with its tokenizer, in compute dtype `dtype`.
///
/// A GGUF carries its own tokenizer. A checkpoint directory is matched to
/// one by vocabulary size. `offload` splits a checkpoint's layers between
/// `device` and the CPU. A GGUF loads whole and ignores it.
///
/// # Errors
///
/// A path that holds no model, weights that do not load, or a tokenizer
/// that does not load.
pub(crate) fn load_model_parts<R>(
    path: &Path,
    device: &R::Device,
    dtype: Option<&DtypeRequest>,
    offload: Option<OffloadingOptions>,
) -> Result<ModelParts<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let source = loader::detect_model_source(path)?;
    let (model, config) = match (source.format, offload) {
        (ModelFormat::Gguf, _) => {
            let parts = loader::load_gguf_with_tokenizer_as::<R, _>(path, device, dtype)?;
            tracing::info!("Using GGUF-embedded tokenizer");
            return Ok(parts);
        }
        (ModelFormat::SafeTensors, None) => loader::load_model_as::<R, _>(path, device, dtype)?,
        (ModelFormat::SafeTensors, Some(options)) => {
            let (model, config, info) =
                loader::load_model_with_offloading::<R, _>(path, device, options, dtype)?;
            tracing::info!(
                "Model loaded: {} layers on GPU, {} on CPU ({:.2} GB on GPU)",
                info.gpu_layers,
                info.cpu_layers,
                info.gpu_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            (model, config)
        }
    };
    let tokenizer = checkpoint_tokenizer(path, config.vocab_size())?;
    Ok((model, config, tokenizer))
}

/// Start an executor over `parts` on `device` with a `num_ctx` KV cache.
///
/// The chat template is the one `chat_template` names, else the model's
/// own. The vision tower is the one `config.inference.mmproj` names, if
/// any. A caller writes its settings into the config before this call.
///
/// # Errors
///
/// A chat template that does not load, an executor that does not start,
/// or a vision tower that does not load.
pub(crate) fn start_executor<R>(
    path: &Path,
    parts: ModelParts<R>,
    device: R::Device,
    num_ctx: usize,
    chat_template: Option<&str>,
) -> Result<Executor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>,
{
    let (model, config, tokenizer) = parts;
    let template = ChatTemplate::detect(path, config.model_type(), chat_template)?;
    Executor::new(model, config, tokenizer, device, num_ctx)?
        .with_chat_template(template)
        .with_configured_vision_tower()
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::{CpuDevice, CpuRuntime};

    #[test]
    fn a_path_with_no_model_is_an_error() {
        let dir = tempfile::tempdir().expect("temp dir");
        let missing = dir.path().join("absent");
        for offload in [None, Some(OffloadingOptions::default())] {
            let loaded = load_model_parts::<CpuRuntime>(&missing, &CpuDevice::new(), None, offload);
            let message = loaded.err().map(|e| e.to_string()).unwrap_or_default();
            assert!(message.contains("does not exist"), "{message}");
        }
    }
}
