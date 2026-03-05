//! AWQ SafeTensors loading
//!
//! AWQ models store quantized weights as 3 tensors per linear layer:
//! - `qweight` [K, N/8] int32 (8 INT4 values packed per u32, AWQ bit order)
//! - `qzeros` [K/gs, N/8] int32 (packed zero points, AWQ bit order)
//! - `scales` [K/gs, N] float16
//!
//! This loader reads the raw SafeTensors, unpacks qzeros at load time,
//! and stores them as `DecomposedQuantTensor` in the VarMap.

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::model::LoadedModel;
use boostr::ops::TensorOps;
use boostr::quant::decomposed::{DecomposedQuantMethod, DecomposedQuantTensor};
use boostr::Tensor;
use boostr::VarBuilder;
use boostr::VarMap;
use boostr::{DType, Runtime};

use crate::config::BlazrConfig;

use super::config::load_or_create_config;
use super::detect_arch::{detect_architecture_from_loader, read_awq_group_size};

/// AWQ bit-shift order for unpacking 8 INT4 values from a u32.
/// AWQ packs with order [0,2,4,6,1,3,5,7], so to extract column j
/// the shift is: [0, 16, 4, 20, 8, 24, 12, 28][j]
const AWQ_SHIFTS: [u32; 8] = [0, 16, 4, 20, 8, 24, 12, 28];

/// Load an AWQ quantized model from SafeTensors
pub fn load_safetensors_awq<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    let path = path.as_ref();

    let config_dir = if path.is_file() {
        path.parent()
    } else {
        Some(path)
    };

    let model_dir = config_dir.ok_or_else(|| anyhow!("Cannot determine model directory"))?;

    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    let group_size = read_awq_group_size(model_dir)?;

    tracing::info!(
        "Loading AWQ INT4 model (group_size={}), size: {:.2} GB",
        group_size,
        loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Detect architecture
    let detected = detect_architecture_from_loader(&loader)?;

    // Load config — AWQ models always run in F16 (scales are F16, reference uses F16)
    let mut config = load_or_create_config(model_dir, &detected)?;
    config.inference.dtype = "f16".to_string();

    // Separate tensor names into quantized triplets and regular tensors
    let tensor_names = loader.tensor_names();
    let (quant_bases, regular_names) = classify_awq_tensors(&tensor_names);

    let mut var_map = VarMap::<R>::new();

    // Load regular tensors (embeddings, norms, lm_head)
    let regular_count = regular_names.len();
    for (idx, name) in regular_names.iter().enumerate() {
        if idx % 20 == 0 {
            tracing::debug!(
                "Loading regular tensor {}/{}: {}",
                idx + 1,
                regular_count,
                name
            );
        }
        let tensor = loader
            .load_tensor::<R>(name, device)
            .map_err(|e| anyhow!("Failed to load tensor '{}': {}", name, e))?;
        // AWQ models are calibrated for F16 inference. If embeddings/norms are stored
        // as BF16, cast to F16 to match the expected precision (BF16 has 3 fewer
        // mantissa bits which causes accumulated errors across 32+ layers).
        // AWQ models run in F16 — cast BF16 tensors to F16
        let tensor = if tensor.dtype() == DType::BF16 {
            let client = R::default_client(device);
            boostr::TypeConversionOps::cast(&client, &tensor, DType::F16)
                .map_err(|e| anyhow!("Failed to cast BF16→F16: {}", e))?
        } else {
            tensor
        };
        var_map.insert(name.clone(), tensor);
    }
    tracing::info!("Loaded {} regular tensors", regular_count);

    // Load quantized triplets as DecomposedQuantTensor
    let quant_count = quant_bases.len();
    for (idx, base_name) in quant_bases.iter().enumerate() {
        if idx % 10 == 0 {
            tracing::debug!(
                "Loading AWQ layer {}/{}: {}",
                idx + 1,
                quant_count,
                base_name
            );
        }

        let dqt = load_awq_triplet::<R>(&mut loader, base_name, group_size, device)?;

        // Store as "base_name.weight" so VarBuilder finds it with standard naming
        let weight_name = format!("{}.weight", base_name);
        var_map.insert_decomposed_quant(weight_name, dqt);
    }
    tracing::info!(
        "Loaded {} AWQ quantized layers (group_size={})",
        quant_count,
        group_size
    );

    // Build model
    let mut vb = VarBuilder::new(&mut var_map, device);

    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config))
}

/// Classify tensor names into quantized layer bases and regular tensor names.
///
/// AWQ quantized layers have `.qweight`, `.qzeros`, `.scales` suffixes.
/// Returns (unique_base_names, regular_tensor_names).
fn classify_awq_tensors(names: &[String]) -> (Vec<String>, Vec<String>) {
    let mut quant_bases = std::collections::BTreeSet::new();
    let mut regular = Vec::new();

    for name in names {
        if let Some(base) = name.strip_suffix(".qweight") {
            quant_bases.insert(base.to_string());
        } else if name.ends_with(".qzeros") || name.ends_with(".scales") {
            // Part of a quant triplet, skip
        } else {
            regular.push(name.clone());
        }
    }

    (quant_bases.into_iter().collect(), regular)
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

/// Cast F16 raw bytes to a F32 tensor
fn cast_f16_bytes_to_f32<R: Runtime<DType = DType>>(
    bytes: &[u8],
    shape: &[usize],
    device: &R::Device,
) -> Result<Tensor<R>> {
    let f16_data: &[half::f16] = bytemuck::cast_slice(bytes);
    let f32_data: Vec<f32> = f16_data.iter().map(|v| v.to_f32()).collect();
    Ok(Tensor::<R>::from_slice(&f32_data, shape, device))
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

    Ok(Tensor::<R>::from_slice(&unpacked, &[num_groups, n], device))
}
