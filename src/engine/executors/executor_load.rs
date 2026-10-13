//! One-call model loading for embedders: path in, ready [`Executor`] out.
//!
//! The CLI, the C ABI and a host with no file system share these loaders.
//! [`load_executor`] reads a GGUF file or a SafeTensors checkpoint
//! directory. [`load_executor_from_bytes`] takes the bytes of a GGUF file or
//! of a single SafeTensors file plus its `config.json`.

use std::path::Path;

use anyhow::{anyhow, Result};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};

pub use crate::engine::load::bytes::{load_executor_from_bytes, ModelBytes};

use super::executor::Executor;
use crate::config::{DtypeRequest, DEFAULT_NUM_CTX};
use crate::engine::load::parts::{load_model_parts, start_executor};

/// How [`load_executor`] loads a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    /// The compute dtype. `None` takes the checkpoint's dtype.
    pub dtype: Option<DtypeRequest>,
    /// The KV cache context length.
    pub num_ctx: usize,
    /// Run one warmup forward pass before returning.
    pub warmup: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            dtype: None,
            num_ctx: DEFAULT_NUM_CTX,
            warmup: true,
        }
    }
}

/// Load the GGUF file or SafeTensors checkpoint at `path` on `device`.
///
/// A GGUF file carries its own tokenizer. A SafeTensors checkpoint takes the
/// tokenizer files in its directory. The chat template is the model's own.
///
/// # Errors
///
/// An error when the path holds no model, the weights do not load, or the
/// warmup pass fails.
pub fn load_executor<R>(
    path: &Path,
    device: R::Device,
    options: &LoadOptions,
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
        + boostr::RoPEPackedOps<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    if !path.exists() {
        return Err(anyhow!(
            "model path {} does not exist. Pass a GGUF file or a SafeTensors directory",
            path.display()
        ));
    }
    let parts = load_model_parts::<R>(path, &device, options.dtype.as_ref(), None)?;
    let executor = start_executor(path, parts, device, options.num_ctx, None)?;
    if options.warmup {
        executor.warmup()?;
    }
    Ok(executor)
}
