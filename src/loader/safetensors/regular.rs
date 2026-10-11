//! Regular (non-quantized) SafeTensors loading

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::model::{LoadedModel, UniversalConfig};
use boostr::ops::TensorOps;
use boostr::VarBuilder;
use boostr::VarMap;
use boostr::{DType, Runtime};

use crate::config::{BlazrConfig, DtypeRequest};

use super::config::{checkpoint_compute_dtype, config_from_detected, load_or_create_config};
use super::detect_arch::{detect_architecture_from_loader, detect_awq, detect_gptq};

/// Load a regular (non-AWQ) SafeTensors model.
///
/// `request` is the user's compute dtype, see `config::compute_dtype`.
/// Every float weight is cast to the resolved dtype.
pub fn load_safetensors_regular<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    request: Option<&DtypeRequest>,
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
        return super::gptq::load_safetensors_gptq(path, device, request);
    }

    // Check if this is an AWQ model
    if detect_awq(&loader, config_dir) {
        return super::awq::load_safetensors_awq(path, device, request);
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

    // Resolve the compute dtype before any weight loads.
    let dtype = checkpoint_compute_dtype::<R>(config_dir, request);
    let config = if let Some(dir) = config_dir {
        load_or_create_config(dir, &detected, dtype)?
    } else {
        config_from_detected(&detected, dtype)?
    };

    // Load all tensors into a VarMap, every float weight at `dtype`.
    let mut var_map = load_tensors_from_loader::<R>(&mut loader, device, dtype)?;

    // VarMap only needs to live through model loading — LoadedModel::load takes
    // tensors out via take_tensor, so VarMap is empty after load and can be dropped.
    let mut vb = VarBuilder::new(&mut var_map, device);

    // Load model using boostr's LoadedModel
    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config))
}

/// Load all tensors from a SafeTensorsLoader into a VarMap.
///
/// Every F32, F16 and BF16 tensor is cast to `dtype`, the compute dtype.
/// Integer tensors keep their dtype.
pub fn load_tensors_from_loader<R: Runtime<DType = DType>>(
    loader: &mut SafeTensorsLoader,
    device: &R::Device,
    dtype: DType,
) -> Result<VarMap<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    let mut var_map = VarMap::<R>::new();
    let client = R::default_client(device);

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
        let tensor = cast_float(&client, tensor, dtype)
            .map_err(|e| anyhow!("tensor '{name}' did not cast to {dtype:?}: {e}"))?;
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
    request: Option<&DtypeRequest>,
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
    let dtype = checkpoint_compute_dtype::<R>(config_dir, request);
    let config = if let Some(dir) = config_dir {
        load_or_create_config(dir, &detected, dtype)?
    } else {
        config_from_detected(&detected, dtype)?
    };

    // Check the head split before any weight loads.
    if let Some(attention) = &config.model.attention {
        let heads = crate::engine::tensor_parallel::TpHeads {
            num_heads: attention.num_heads,
            num_kv_heads: attention.num_kv_heads.unwrap_or(attention.num_heads),
        };
        crate::engine::tensor_parallel::validate_tp_config(comm.world_size(), Some(heads))
            .map_err(|e| anyhow!("{e}"))?;
    }

    tracing::info!(
        "Loading tensor-parallel model (rank={}, world_size={})",
        comm.rank(),
        comm.world_size()
    );

    let mut var_map = load_tensors_from_loader::<R>(&mut loader, device, dtype)?;
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

    let config_dir = if path.is_file() {
        path.parent()
    } else {
        Some(path)
    };
    let dtype = checkpoint_compute_dtype::<R>(config_dir, None);
    let mut var_map = load_tensors_from_loader::<R>(&mut loader, device, dtype)?;
    let mut vb = VarBuilder::new(&mut var_map, device);

    let model =
        LoadedModel::load(config, &mut vb).map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok(model)
}

/// `tensor` at `dtype` when it holds F32, F16 or BF16. Any other tensor
/// passes unchanged.
pub(super) fn cast_float<R: Runtime<DType = DType>>(
    client: &R::Client,
    tensor: boostr::Tensor<R>,
    dtype: DType,
) -> Result<boostr::Tensor<R>>
where
    R::Client: TensorOps<R>,
{
    let float = matches!(tensor.dtype(), DType::F32 | DType::F16 | DType::BF16);
    if !float || tensor.dtype() == dtype {
        return Ok(tensor);
    }
    boostr::TypeConversionOps::cast(client, &tensor, dtype).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::{CpuDevice, CpuRuntime, Tensor};

    #[cfg(feature = "f16")]
    #[test]
    fn float_weights_take_the_compute_dtype() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let values = [1.5f32, -2.25, 0.125, 3.0];
        let bf16: Vec<half::bf16> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let weight = Tensor::<CpuRuntime>::from_slice(&bf16, &[2, 2], &device).unwrap();
        let up = cast_float(&client, weight, DType::F32).unwrap();
        assert_eq!(up.dtype(), DType::F32);
        assert_eq!(up.to_vec::<f32>(), values);

        let f32_weight = Tensor::<CpuRuntime>::from_slice(&values, &[4], &device).unwrap();
        assert_eq!(
            cast_float(&client, f32_weight, DType::BF16)
                .unwrap()
                .dtype(),
            DType::BF16
        );
    }

    #[test]
    fn integer_tensors_keep_their_dtype() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let ids = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3], &[3], &device).unwrap();
        assert_eq!(
            cast_float(&client, ids, DType::F32).unwrap().dtype(),
            DType::I32
        );
    }
}
