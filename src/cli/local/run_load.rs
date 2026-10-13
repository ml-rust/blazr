//! The model load of `run` and `generate`. It reads the disk and uploads
//! to the device, so `run` calls it off the async executor.

use std::path::Path;

use anyhow::Result;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
#[cfg(feature = "cuda")]
use boostr::CudaRuntime;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, TypeConversionOps, UnaryOps,
};

use crate::cli::RunSettings;
use crate::config::InferenceConfig;
use crate::engine::load::parts::{load_model_parts, start_executor};
use crate::engine::lora::{derive_run_model, run_adapter};
use crate::engine::Executor;
use crate::loader::OffloadingOptions;

/// Load `model` on CUDA GPU `cuda_index` with the `--lora` adapter, then
/// warm it up. Returns the executor with whether paged attention and
/// graph mode are on.
///
/// `--gpu-layers` below 0 loads a checkpoint whole on the GPU. 0 splits
/// its layers between the GPU and the CPU by free VRAM, and N puts N
/// layers on the GPU. A GGUF loads whole.
///
/// # Errors
///
/// A model, adapter, tokenizer, chat template or vision tower that does
/// not load, or CUDA modules that do not preload.
#[cfg(feature = "cuda")]
pub(super) fn load_cuda(
    model: &str,
    cuda_index: usize,
    runtime: &RunSettings,
    chat_template: Option<&str>,
) -> Result<(Executor<CudaRuntime>, bool, bool)> {
    let gpu_layers = runtime.gpu_layers;
    let offload = match usize::try_from(gpu_layers) {
        Err(_) => None,
        Ok(0) => Some(OffloadingOptions::default()),
        Ok(layers) => Some(OffloadingOptions::default().gpu_layers(layers)),
    };
    tracing::info!("Loading model: {}", model);
    let mut decode_modes = (false, false);
    let executor = load_for_run::<CudaRuntime>(
        model,
        boostr::CudaDevice::new(cuda_index),
        runtime,
        chat_template,
        offload,
        |inference, device| {
            // The user file's inference keys, then the flags.
            runtime.apply(inference)?;
            decode_modes = (inference.paged_attention, inference.graphs);
            if inference.paged_attention {
                tracing::info!(
                    "Paged attention enabled (block_size={})",
                    inference.block_size
                );
            }
            if inference.graphs {
                tracing::info!("CUDA graph mode enabled");
            }
            crate::runtime::preload_cuda_modules(device)
        },
    )?;
    let (paged_attention, graphs) = decode_modes;
    Ok((executor, paged_attention, graphs))
}

/// Load `model` on `device`, a runtime with no paged or graph decode, with
/// the `--lora` adapter, then warm it up.
///
/// # Errors
///
/// A model, adapter, tokenizer, chat template or vision tower that does
/// not load.
pub(super) fn load_plain<R>(
    model: &str,
    device: R::Device,
    runtime: &RunSettings,
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
        + ModelClient<R>
        + RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    tracing::info!("Loading model: {} on {}", model, R::name());
    // This path has no paged or graph decode, so only the load keys apply.
    load_for_run::<R>(
        model,
        device,
        runtime,
        chat_template,
        None,
        |inference, _| runtime.load.apply(inference),
    )
}

/// Load `model` on `device`, derive the `--lora` variant, run `settle`
/// over the loaded config, then start the executor and warm it up.
///
/// # Errors
///
/// A model, adapter, tokenizer, chat template or vision tower that does
/// not load, or the error of `settle`.
fn load_for_run<R>(
    model: &str,
    device: R::Device,
    runtime: &RunSettings,
    chat_template: Option<&str>,
    offload: Option<OffloadingOptions>,
    settle: impl FnOnce(&mut InferenceConfig, &R::Device) -> Result<()>,
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
        + ModelClient<R>
        + RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    let path = Path::new(model);
    let (loaded_model, mut config, tokenizer) =
        load_model_parts::<R>(path, &device, runtime.load.dtype.as_ref(), offload)?;
    let loaded_model =
        derive_run_model(loaded_model, run_adapter(&runtime.lora)?.as_ref(), &device)?;
    settle(&mut config.inference, &device)?;
    let executor = start_executor(
        path,
        (loaded_model, config, tokenizer),
        device,
        runtime.load.num_ctx,
        chat_template,
    )?;
    // Warm up kernels: compiles GPU pipelines before the first token.
    executor.warmup()?;
    tracing::info!("Model ready");
    Ok(executor)
}
