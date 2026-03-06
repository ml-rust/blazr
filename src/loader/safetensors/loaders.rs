use std::path::Path;

use anyhow::Result;

use boostr::model::LoadedModel;
use boostr::ops::TensorOps;
use boostr::{DType, Runtime};

use crate::config::BlazrConfig;

use super::offloading::{OffloadingInfo, OffloadingOptions};
use super::regular;

/// Load a model from SafeTensors format (no offloading)
pub fn load_safetensors<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    regular::load_safetensors_regular(path, device)
}

/// Load a model from SafeTensors with explicit configuration
pub fn load_safetensors_with_config<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    config: &boostr::model::UniversalConfig,
    device: &R::Device,
) -> Result<LoadedModel<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    regular::load_safetensors_with_config(path, config, device)
}

/// Load a model with GPU+CPU offloading
#[cfg(feature = "cuda")]
pub fn load_safetensors_with_offloading<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    gpu_device: &R::Device,
    options: OffloadingOptions,
) -> Result<(LoadedModel<R>, BlazrConfig, OffloadingInfo)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    super::offloading::load_safetensors_with_offloading(path, gpu_device, options)
}

/// Load a model with tensor parallelism (weight sharding across ranks)
pub fn load_safetensors_tp<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    comm: std::sync::Arc<dyn boostr::runtime::Communicator>,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    regular::load_safetensors_tp(path, device, comm)
}

/// Non-CUDA fallback for offloading (just loads normally)
#[cfg(not(feature = "cuda"))]
pub fn load_safetensors_with_offloading<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    _options: OffloadingOptions,
) -> Result<(LoadedModel<R>, BlazrConfig, OffloadingInfo)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let (model, config) = load_safetensors(path, device)?;
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
