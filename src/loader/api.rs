use std::path::Path;

use anyhow::Result;

use boostr::model::{LoadedModel, UniversalConfig};
use boostr::ops::TensorOps;
use boostr::{DType, Runtime};

use crate::config::BlazrConfig;

use super::detect::{detect_model_source, ModelFormat};
use super::gguf;
use super::safetensors::{self, OffloadingInfo, OffloadingOptions};

/// Load a model from any supported format
///
/// This function auto-detects the format and loads the model appropriately.
/// For AWQ quantized SafeTensors models, automatically detects and uses
/// native INT4 inference.
pub fn load_model<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let path = path.as_ref();
    let source = detect_model_source(path)?;

    match source.format {
        ModelFormat::SafeTensors => safetensors::load_safetensors(path, device),
        ModelFormat::Gguf => gguf::load_gguf(path, device),
    }
}

/// Load a model with tensor parallelism (weight sharding across ranks).
///
/// Requires a NCCL communicator. Only supported for SafeTensors Llama-family models.
pub fn load_model_tp<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    comm: std::sync::Arc<dyn boostr::runtime::Communicator>,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let path = path.as_ref();
    let source = detect_model_source(path)?;

    match source.format {
        ModelFormat::SafeTensors => safetensors::load_safetensors_tp(path, device, comm),
        ModelFormat::Gguf => Err(anyhow::anyhow!(
            "Tensor parallelism is not supported for GGUF format"
        )),
    }
}

/// Load a model with explicit configuration
///
/// Use this when you have a specific configuration and don't want auto-detection.
pub fn load_model_with_config<R: Runtime<DType = DType>, P: AsRef<Path>>(
    weights_path: P,
    config: &UniversalConfig,
    device: &R::Device,
) -> Result<LoadedModel<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let weights_path = weights_path.as_ref();
    let source = detect_model_source(weights_path)?;

    match source.format {
        ModelFormat::SafeTensors => {
            safetensors::load_safetensors_with_config(weights_path, config, device)
        }
        ModelFormat::Gguf => gguf::load_gguf_with_config(weights_path, config, device),
    }
}

/// Load a model with GPU+CPU offloading
///
/// Automatically determines how many layers fit in available VRAM and loads
/// the rest to CPU. Currently only supported for SafeTensors format.
pub fn load_model_with_offloading<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    options: OffloadingOptions,
) -> Result<(LoadedModel<R>, BlazrConfig, OffloadingInfo)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let path = path.as_ref();
    let source = detect_model_source(path)?;

    match source.format {
        ModelFormat::SafeTensors => {
            safetensors::load_safetensors_with_offloading(path, device, options)
        }
        ModelFormat::Gguf => {
            let (model, config) = gguf::load_gguf(path, device)?;
            Ok((
                model,
                config,
                OffloadingInfo {
                    gpu_layers: 0,
                    cpu_layers: 0,
                    gpu_bytes: 0,
                    cpu_bytes: 0,
                    total_layers: 0,
                },
            ))
        }
    }
}
