//! AWQ SafeTensors loading
//!
//! AWQ models store quantized weights as 3 tensors per linear layer:
//! - `qweight` [K, N/8] int32 (8 INT4 values packed per u32, AWQ bit order)
//! - `qzeros` [K/gs, N/8] int32 (packed zero points, AWQ bit order)
//! - `scales` [K/gs, N] float16
//!
//! This loader reads the raw SafeTensors, unpacks qzeros at load time,
//! and stores them as `DecomposedQuantTensor` in the VarMap.

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
use super::detect_arch::read_awq_group_size;

/// AWQ bit-shift order for unpacking 8 INT4 values from a u32.
/// AWQ packs with order [0,2,4,6,1,3,5,7], so to extract column j
/// the shift is: [0, 16, 4, 20, 8, 24, 12, 28][j]
const AWQ_SHIFTS: [u32; 8] = [0, 16, 4, 20, 8, 24, 12, 28];

/// Load an AWQ quantized model from SafeTensors
#[cfg(feature = "cuda")]
pub fn load_safetensors_awq<R: Runtime<DType = DType>, P: AsRef<Path>>(
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
    load_awq_scoped(path.as_ref(), device, request, None)
}

/// [`load_safetensors_awq`] of the whole model, or of layers `layers` as
/// a pipeline stage holds them. A range load reads only the tensors of its
/// span: see `LayerSpan::reads`.
pub(super) fn load_awq_scoped<R: Runtime<DType = DType>>(
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
    load_decomposed_scoped::<Awq, R>(path, device, request, layers)
}

/// The AWQ format: a qweight + scales + qzeros triplet per quantized layer.
struct Awq;

impl DecomposedFormat for Awq {
    const NAME: &'static str = "AWQ";
    const PART_SUFFIXES: &'static [&'static str] = &[".qzeros", ".scales"];
    const LAYER_BIAS: bool = false;

    fn group_size(model_dir: &Path) -> Result<usize> {
        read_awq_group_size(model_dir)
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
        Ok((
            load_awq_triplet::<R>(loader, base_name, group_size, device)?,
            None,
        ))
    }
}

/// Load a single AWQ quantized triplet (qweight + scales + qzeros) into a DecomposedQuantTensor
fn load_awq_triplet<R: Runtime<DType = DType>>(
    loader: &mut SafeTensorsLoader,
    base_name: &str,
    group_size: usize,
    device: &R::Device,
) -> Result<DecomposedQuantTensor<R>>
where
    R::Client: TensorOps<R>,
{
    let qweight_name = format!("{}.qweight", base_name);
    let scales_name = format!("{}.scales", base_name);
    let qzeros_name = format!("{}.qzeros", base_name);

    // Get tensor info for shapes
    let qw_info = loader
        .tensor_info(&qweight_name)
        .map_err(|e| anyhow!("Missing {}: {}", qweight_name, e))?
        .clone();
    let sc_info = loader
        .tensor_info(&scales_name)
        .map_err(|e| anyhow!("Missing {}: {}", scales_name, e))?
        .clone();
    let qz_info = loader
        .tensor_info(&qzeros_name)
        .map_err(|e| anyhow!("Missing {}: {}", qzeros_name, e))?
        .clone();

    // qweight: [K, N/8] int32 → load as raw bytes, reinterpret as U32
    let qw_bytes = loader
        .read_tensor_bytes(&qweight_name)
        .map_err(|e| anyhow!("Failed to read {}: {}", qweight_name, e))?;
    let qw_storage = boostr::tensor::Storage::<R>::from_bytes(&qw_bytes, DType::U32, device)
        .map_err(|e| anyhow!("Failed to create qweight storage: {}", e))?;
    let qweight = Tensor::<R>::from_storage_contiguous(qw_storage, &qw_info.shape);

    let k = qw_info.shape[0];
    let n_div_8 = qw_info.shape[1];
    let n = n_div_8 * 8;

    // scales: [K/gs, N] float16 → load as F16, then cast to F32
    let sc_bytes = loader
        .read_tensor_bytes(&scales_name)
        .map_err(|e| anyhow!("Failed to read {}: {}", scales_name, e))?;
    let scales = cast_f16_bytes_to_f32::<R>(&sc_bytes, &sc_info.shape, device)?;

    // qzeros: [K/gs, N/8] int32 → unpack to [K/gs, N] f32
    let qz_bytes = loader
        .read_tensor_bytes(&qzeros_name)
        .map_err(|e| anyhow!("Failed to read {}: {}", qzeros_name, e))?;
    let num_groups = qz_info.shape[0];
    let qzeros = unpack_awq_zeros(&qz_bytes, num_groups, n, device)?;

    // Logical shape: [N, K] (out_features × in_features, like a standard linear weight)
    let logical_shape = vec![n, k];

    Ok(DecomposedQuantTensor::new(
        qweight,
        scales,
        qzeros,
        None,
        DecomposedQuantMethod::Awq { group_size },
        logical_shape,
    ))
}

/// Unpack AWQ-packed qzeros from [num_groups, N/8] int32 → [num_groups, N] f32
///
/// AWQ packs 8 INT4 zero values per u32 with bit-shift order: [0,16,4,20,8,24,12,28]
fn unpack_awq_zeros<R: Runtime<DType = DType>>(
    bytes: &[u8],
    num_groups: usize,
    n: usize,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let packed: &[u32] = bytemuck::cast_slice(bytes);
    let n_div_8 = n / 8;
    let mut unpacked = vec![0.0f32; num_groups * n];

    for g in 0..num_groups {
        for j in 0..n_div_8 {
            let packed_val = packed[g * n_div_8 + j];
            for (k, &shift) in AWQ_SHIFTS.iter().enumerate() {
                let zero = ((packed_val >> shift) & 0xF) as f32;
                unpacked[g * n + j * 8 + k] = zero;
            }
        }
    }

    Ok(Tensor::<R>::from_slice(
        &unpacked,
        &[num_groups, n],
        device,
    )?)
}
