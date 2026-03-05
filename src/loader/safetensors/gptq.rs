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
use super::detect_arch::{detect_architecture_from_loader, read_gptq_group_size};

/// Load a GPTQ quantized model from SafeTensors
pub fn load_safetensors_gptq<R: Runtime<DType = DType>, P: AsRef<Path>>(
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

    let group_size = read_gptq_group_size(model_dir)?;

    tracing::info!(
        "Loading GPTQ INT4 model (group_size={}), size: {:.2} GB",
        group_size,
        loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Detect architecture
    let detected = detect_architecture_from_loader(&loader)?;

    // Load config — GPTQ models run in F16 (scales are F16)
    let mut config = load_or_create_config(model_dir, &detected)?;
    config.inference.dtype = "f16".to_string();

    // Separate tensor names into quantized groups and regular tensors
    let tensor_names = loader.tensor_names();
    let (quant_bases, regular_names) = classify_gptq_tensors(&tensor_names);

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
        // GPTQ models run in F16 — cast BF16 tensors to F16
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

    // Load quantized groups as DecomposedQuantTensor
    let quant_count = quant_bases.len();
    for (idx, base_name) in quant_bases.iter().enumerate() {
        if idx % 10 == 0 {
            tracing::debug!(
                "Loading GPTQ layer {}/{}: {}",
                idx + 1,
                quant_count,
                base_name
            );
        }

        let (dqt, bias) = load_gptq_group::<R>(&mut loader, base_name, group_size, device)?;

        // Store as "base_name.weight" so VarBuilder finds it with standard naming
        let weight_name = format!("{}.weight", base_name);
        var_map.insert_decomposed_quant(weight_name, dqt);

        // Store bias if present
        if let Some(bias_tensor) = bias {
            let bias_name = format!("{}.bias", base_name);
            var_map.insert(bias_name, bias_tensor);
        }
    }
    tracing::info!(
        "Loaded {} GPTQ quantized layers (group_size={})",
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
/// GPTQ quantized layers have `.qweight`, `.qzeros`, `.scales`, `.g_idx`, `.bias` suffixes.
/// Returns (unique_base_names, regular_tensor_names).
fn classify_gptq_tensors(names: &[String]) -> (Vec<String>, Vec<String>) {
    let mut quant_bases = std::collections::BTreeSet::new();
    let mut regular = Vec::new();

    for name in names {
        if let Some(base) = name.strip_suffix(".qweight") {
            quant_bases.insert(base.to_string());
        } else if name.ends_with(".qzeros") || name.ends_with(".scales") || name.ends_with(".g_idx")
        {
            // Part of a quant group, skip
        } else if name.ends_with(".bias") {
            // Check if this bias belongs to a quantized layer (will be handled there)
            let base = name.strip_suffix(".bias").unwrap();
            let qw_name = format!("{}.qweight", base);
            if names.iter().any(|n| n == &qw_name) {
                // Part of a quant group, skip
            } else {
                regular.push(name.clone());
            }
        } else {
            regular.push(name.clone());
        }
    }

    (quant_bases.into_iter().collect(), regular)
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
