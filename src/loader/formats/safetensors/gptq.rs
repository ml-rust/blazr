//! GPTQ SafeTensors loading
//!
//! GPTQ models store quantized weights as 5 tensors per linear layer:
//! - `qweight` [K/8, N] int32 (8 INT4 values packed per u32, sequential 4-bit)
//! - `qzeros` [groups, N/8] int32 (packed zero points, kept packed for kernel)
//! - `scales` [groups, N] float16
//! - `g_idx` [K] int32 (column-to-group mapping)
//! - `bias` [N] float16 (optional)
//!
//! This loader reads the raw SafeTensors, keeps qzeros packed (the kernel
//! reads them directly), and stores them as `DecomposedQuantTensor` in the VarMap.

use std::ops::Range;
use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::model::LoadedModel;
use boostr::ops::TensorOps;
use boostr::quant::decomposed::{DecomposedQuantMethod, DecomposedQuantTensor};
use boostr::Tensor;
use boostr::{DType, Runtime};

use crate::config::{BlazrConfig, DtypeRequest};

use super::decomposed::{cast_f16_bytes_to_f32, load_decomposed_scoped, DecomposedFormat};
use super::detect_arch::read_gptq_group_size;

/// Load a GPTQ quantized model from SafeTensors
#[cfg(feature = "cuda")]
pub fn load_safetensors_gptq<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    request: Option<&DtypeRequest>,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    load_gptq_scoped(path.as_ref(), device, request, None)
}

/// [`load_safetensors_gptq`] of the whole model, or of layers `layers` as
/// a pipeline stage holds them. A range load reads only the tensors of its
/// span: see `LayerSpan::reads`.
pub(super) fn load_gptq_scoped<R: Runtime<DType = DType>>(
    path: &Path,
    device: &R::Device,
    request: Option<&DtypeRequest>,
    layers: Option<Range<usize>>,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    load_decomposed_scoped::<Gptq, R>(path, device, request, layers)
}

/// The GPTQ format: qweight, scales, qzeros, g_idx and an optional bias per
/// quantized layer.
struct Gptq;

impl DecomposedFormat for Gptq {
    const NAME: &'static str = "GPTQ";
    const PART_SUFFIXES: &'static [&'static str] = &[".qzeros", ".scales", ".g_idx"];
    const LAYER_BIAS: bool = true;

    fn group_size(model_dir: &Path) -> Result<usize> {
        read_gptq_group_size(model_dir)
    }

    fn load_layer<R: Runtime<DType = DType>>(
        loader: &mut SafeTensorsLoader,
        base_name: &str,
        group_size: usize,
        device: &R::Device,
    ) -> Result<(DecomposedQuantTensor<R>, Option<Tensor<R>>)>
    where
        R::Client: TensorOps<R>,
    {
        load_gptq_group::<R>(loader, base_name, group_size, device)
    }
}

/// Load a single GPTQ quantized group into a DecomposedQuantTensor + optional bias
fn load_gptq_group<R: Runtime<DType = DType>>(
    loader: &mut SafeTensorsLoader,
    base_name: &str,
    group_size: usize,
    device: &R::Device,
) -> Result<(DecomposedQuantTensor<R>, Option<Tensor<R>>)>
where
    R::Client: TensorOps<R>,
{
    let qweight_name = format!("{}.qweight", base_name);
    let scales_name = format!("{}.scales", base_name);
    let qzeros_name = format!("{}.qzeros", base_name);
    let g_idx_name = format!("{}.g_idx", base_name);
    let bias_name = format!("{}.bias", base_name);

    // Get tensor info for shapes
    let qw_info = loader
        .tensor_info(&qweight_name)
        .map_err(|e| anyhow!("Missing {}: {}", qweight_name, e))?
        .clone();
    let sc_info = loader
        .tensor_info(&scales_name)
        .map_err(|e| anyhow!("Missing {}: {}", scales_name, e))?
        .clone();

    // qweight: [K/8, N] int32 → load as raw bytes, reinterpret as U32
    let qw_bytes = loader
        .read_tensor_bytes(&qweight_name)
        .map_err(|e| anyhow!("Failed to read {}: {}", qweight_name, e))?;
    let qw_storage = boostr::tensor::Storage::<R>::from_bytes(&qw_bytes, DType::U32, device)
        .map_err(|e| anyhow!("Failed to create qweight storage: {}", e))?;
    let qweight = Tensor::<R>::from_storage_contiguous(qw_storage, &qw_info.shape);

    let k_div_8 = qw_info.shape[0];
    let n = qw_info.shape[1];
    let k = k_div_8 * 8;

    // scales: [groups, N] float16 → load as F16, then cast to F32
    let sc_bytes = loader
        .read_tensor_bytes(&scales_name)
        .map_err(|e| anyhow!("Failed to read {}: {}", scales_name, e))?;
    let scales = cast_f16_bytes_to_f32::<R>(&sc_bytes, &sc_info.shape, device)?;

    // qzeros: [groups, N/8] int32 → keep packed as U32 (kernel reads packed)
    let qz_bytes = loader
        .read_tensor_bytes(&qzeros_name)
        .map_err(|e| anyhow!("Failed to read {}: {}", qzeros_name, e))?;
    let qz_info = loader
        .tensor_info(&qzeros_name)
        .map_err(|e| anyhow!("Missing {}: {}", qzeros_name, e))?
        .clone();
    let qz_storage = boostr::tensor::Storage::<R>::from_bytes(&qz_bytes, DType::U32, device)
        .map_err(|e| anyhow!("Failed to create qzeros storage: {}", e))?;
    let qzeros = Tensor::<R>::from_storage_contiguous(qz_storage, &qz_info.shape);

    // g_idx: [K] int32 → reinterpret as I32
    let g_idx = if let Ok(gi_bytes) = loader.read_tensor_bytes(&g_idx_name) {
        let gi_storage = boostr::tensor::Storage::<R>::from_bytes(&gi_bytes, DType::I32, device)
            .map_err(|e| anyhow!("Failed to create g_idx storage: {}", e))?;
        Some(Tensor::<R>::from_storage_contiguous(gi_storage, &[k]))
    } else {
        None
    };

    // bias: [N] float16 → load as F16, cast to F32
    let bias = if let Ok(bias_bytes) = loader.read_tensor_bytes(&bias_name) {
        let bias_info = loader
            .tensor_info(&bias_name)
            .map_err(|e| anyhow!("Missing {}: {}", bias_name, e))?
            .clone();
        let bias_tensor = cast_f16_bytes_to_f32::<R>(&bias_bytes, &bias_info.shape, device)?;
        Some(bias_tensor)
    } else {
        None
    };

    // Logical shape: [N, K] (out_features × in_features)
    let logical_shape = vec![n, k];

    let dqt = DecomposedQuantTensor::new(
        qweight,
        scales,
        qzeros,
        g_idx,
        DecomposedQuantMethod::Gptq { group_size },
        logical_shape,
    );

    Ok((dqt, bias))
}
