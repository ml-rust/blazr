//! Regular (non-quantized) SafeTensors loading

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::model::{LoadedModel, UniversalConfig};
use boostr::ops::TensorOps;
use boostr::VarBuilder;
use boostr::VarMap;
use boostr::{DType, Runtime};

use crate::config::BlazrConfig;

use super::config::{config_from_detected, load_or_create_config};
use super::detect_arch::{detect_architecture_from_loader, detect_awq, detect_gptq};

/// Load a regular (non-AWQ) SafeTensors model
pub fn load_safetensors_regular<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let path = path.as_ref();

    // Determine config directory
    let config_dir = if path.is_file() {
        path.parent()
    } else {
        Some(path)
    };

    // Use boostr's unified SafeTensorsLoader
    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    // Check if this is a GPTQ model
    if detect_gptq(&loader, config_dir) {
        return super::gptq::load_safetensors_gptq(path, device);
    }

    // Check if this is an AWQ model
    if detect_awq(&loader, config_dir) {
        return super::awq::load_safetensors_awq(path, device);
    }

    // Log loading info
    if loader.is_sharded() {
        tracing::info!(
            "Loading sharded model with {} shards, total size: {:.2} GB",
            loader.num_shards(),
            loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        tracing::info!(
            "Loading single-file model, size: {:.2} GB",
            loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    // Detect architecture from tensor names and shapes
    let detected = detect_architecture_from_loader(&loader)?;

    // Try to load config.json if available
    let config = if let Some(dir) = config_dir {
        load_or_create_config(dir, &detected)?
    } else {
        config_from_detected(&detected)
    };

    // Load all tensors into a VarMap
    let mut var_map = load_tensors_from_loader::<R>(&mut loader, device)?;

    // VarMap only needs to live through model loading — LoadedModel::load takes
    // tensors out via take_tensor, so VarMap is empty after load and can be dropped.
    let mut vb = VarBuilder::new(&mut var_map, device);

    // Load model using boostr's LoadedModel
    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config))
}

/// Load all tensors from a SafeTensorsLoader into a VarMap
pub fn load_tensors_from_loader<R: Runtime<DType = DType>>(
    loader: &mut SafeTensorsLoader,
    device: &R::Device,
) -> Result<VarMap<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    let mut var_map = VarMap::<R>::new();

    let tensor_names = loader.tensor_names();
    let total = tensor_names.len();

    tracing::info!("Loading {} tensors to device...", total);

    for (idx, name) in tensor_names.into_iter().enumerate() {
        if idx % 50 == 0 {
            tracing::debug!("Loading tensor {}/{}: {}", idx + 1, total, name);
        }

        let tensor = loader
            .load_tensor::<R>(&name, device)
            .map_err(|e| anyhow!("Failed to load tensor '{}': {}", name, e))?;
        var_map.insert(name, tensor);
    }

    tracing::info!("Loaded {} tensors successfully", var_map.len());

    Ok(var_map)
}

/// Load a tensor-parallel model from SafeTensors.
/// Loads all tensors then uses `LoadedModel::load_tp` to shard weights per-rank.
pub fn load_safetensors_tp<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    comm: std::sync::Arc<dyn boostr::runtime::Communicator>,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let path = path.as_ref();
    let config_dir = if path.is_file() {
        path.parent()
    } else {
        Some(path)
    };

    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    let detected = detect_architecture_from_loader(&loader)?;
    let config = if let Some(dir) = config_dir {
        load_or_create_config(dir, &detected)?
    } else {
        config_from_detected(&detected)
    };

    tracing::info!(
        "Loading tensor-parallel model (rank={}, world_size={})",
        comm.rank(),
        comm.world_size()
    );

    let mut var_map = load_tensors_from_loader::<R>(&mut loader, device)?;
    let mut vb = VarBuilder::new(&mut var_map, device);

    let model = LoadedModel::load_tp(&config.model, &mut vb, comm)
        .map_err(|e| anyhow!("Failed to load TP model: {}", e))?;

    Ok((model, config))
}

/// Load a model from SafeTensors with explicit configuration
pub fn load_safetensors_with_config<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    config: &UniversalConfig,
    device: &R::Device,
) -> Result<LoadedModel<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let path = path.as_ref();

    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    if loader.is_sharded() {
        tracing::info!(
            "Loading sharded model with {} shards, total size: {:.2} GB",
            loader.num_shards(),
            loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    let mut var_map = load_tensors_from_loader::<R>(&mut loader, device)?;
    let mut vb = VarBuilder::new(&mut var_map, device);

    let model =
        LoadedModel::load(config, &mut vb).map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok(model)
}
