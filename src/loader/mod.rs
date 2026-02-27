//! Model loading utilities
//!
//! This module provides unified model loading from various formats:
//! - SafeTensors (HuggingFace standard)
//! - GGUF (llama.cpp format, for quantized models)
//!
//! All heavy lifting is done by boostr's format module.

mod detect;
mod gguf;
mod safetensors;

pub use detect::{detect_model_source, ModelFormat, ModelSource};
pub use gguf::{get_gguf_info, load_gguf, load_gguf_with_tokenizer, GgufInfo};
pub use safetensors::{
    load_safetensors, load_safetensors_with_offloading, OffloadingInfo, OffloadingOptions,
};

use std::path::Path;

use anyhow::Result;

use boostr::model::{LoadedModel, UniversalConfig};
use boostr::ops::TensorOps;
use boostr::{DType, Runtime};

use crate::config::BlazrConfig;

/// Load a model from any supported format
///
/// This function auto-detects the format and loads the model appropriately.
pub fn load_model<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R>,
{
    let path = path.as_ref();
    let source = detect_model_source(path)?;

    match source.format {
        ModelFormat::SafeTensors => load_safetensors(path, device),
        ModelFormat::Gguf => load_gguf(path, device),
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
    R::Client: TensorOps<R>,
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
///
/// # Arguments
/// * `path` - Path to model directory or weights file
/// * `device` - GPU device to load to
/// * `options` - Offloading configuration
///
/// # Returns
/// Tuple of (model, config, offloading_info)
pub fn load_model_with_offloading<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    options: OffloadingOptions,
) -> Result<(LoadedModel<R>, BlazrConfig, OffloadingInfo)>
where
    R::Client: TensorOps<R>,
{
    let path = path.as_ref();
    let source = detect_model_source(path)?;

    match source.format {
        ModelFormat::SafeTensors => load_safetensors_with_offloading(path, device, options),
        ModelFormat::Gguf => {
            // GGUF already handles quantization - just load normally
            let (model, config) = load_gguf(path, device)?;
            // Return with dummy offloading info since GGUF is already compressed
            Ok((
                model,
                config,
                OffloadingInfo {
                    gpu_layers: 0, // Not applicable
                    cpu_layers: 0,
                    gpu_bytes: 0,
                    cpu_bytes: 0,
                    total_layers: 0,
                },
            ))
        }
    }
}
