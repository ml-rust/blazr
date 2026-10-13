//! Regular (non-quantized) SafeTensors loading

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::model::{LoadedModel, UniversalConfig};
use boostr::ops::TensorOps;
use boostr::VarBuilder;
use boostr::{DType, Runtime};

use crate::config::{BlazrConfig, DtypeRequest};

use super::checkpoint_map::{load_checkpoint_map, TensorLoadReport};
use super::config::{checkpoint_compute_dtype, config_from_detected, load_or_create_config};
use super::detect_arch::{detect_architecture_from_loader, detect_awq, detect_gptq};

/// Load a regular (non-AWQ) SafeTensors model.
///
/// `request` is the user's compute dtype, see `config::compute_dtype`.
/// Every float weight takes the resolved dtype.
pub fn load_safetensors_regular<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    request: Option<&DtypeRequest>,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let (model, config, _) = load_safetensors_with_report(path, device, request)?;
    Ok((model, config))
}

/// [`load_safetensors_regular`], plus what the load read from the files.
///
/// The report is `None` for a GPTQ or AWQ checkpoint, which loads through
/// its own reader.
pub fn load_safetensors_with_report<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
    request: Option<&DtypeRequest>,
) -> Result<(LoadedModel<R>, BlazrConfig, Option<TensorLoadReport>)>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R> + boostr::quant::QuantMatmulOps<R>,
{
    let path = path.as_ref();
    let config_dir = config_dir_of(path);

    // Use boostr's unified SafeTensorsLoader
    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    if detect_gptq(&loader, config_dir) {
        let (model, config) = super::gptq::load_safetensors_gptq(path, device, request)?;
        return Ok((model, config, None));
    }
    if detect_awq(&loader, config_dir) {
        let (model, config) = super::awq::load_safetensors_awq(path, device, request)?;
        return Ok((model, config, None));
    }

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

    // Resolve the compute dtype before any weight loads. Tensor names are
    // read only when no config.json describes the model.
    let dtype = checkpoint_compute_dtype::<R>(config_dir, request);
    let config = resolve_config(&loader, config_dir, dtype)?;

    let (mut var_map, report) =
        load_checkpoint_map::<R>(&mut loader, &config.model, device, dtype)?;

    // LoadedModel::load takes every tensor out of the map, so the map is
    // empty afterwards.
    let mut vb = VarBuilder::new(&mut var_map, device);
    let model = LoadedModel::load(&config.model, &mut vb)
        .map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok((model, config, Some(report)))
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
    let config_dir = config_dir_of(path);

    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    let dtype = checkpoint_compute_dtype::<R>(config_dir, request);
    let config = resolve_config(&loader, config_dir, dtype)?;

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

    let (mut var_map, _) = load_checkpoint_map::<R>(&mut loader, &config.model, device, dtype)?;
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

    let dtype = checkpoint_compute_dtype::<R>(config_dir_of(path), None);
    let (mut var_map, _) = load_checkpoint_map::<R>(&mut loader, config, device, dtype)?;
    let mut vb = VarBuilder::new(&mut var_map, device);

    let model =
        LoadedModel::load(config, &mut vb).map_err(|e| anyhow!("Failed to load model: {}", e))?;

    Ok(model)
}

/// The directory that holds `path`'s `config.json`: `path` itself, or the
/// parent of a file.
pub(super) fn config_dir_of(path: &Path) -> Option<&Path> {
    if path.is_file() {
        path.parent()
    } else {
        Some(path)
    }
}

/// The config of the checkpoint `loader` reads, from `config.json` in
/// `config_dir` when present, else from its tensor names.
pub(super) fn resolve_config(
    loader: &SafeTensorsLoader,
    config_dir: Option<&Path>,
    dtype: DType,
) -> Result<BlazrConfig> {
    match config_dir {
        Some(dir) => load_or_create_config(dir, || detect_architecture_from_loader(loader), dtype),
        None => config_from_detected(&detect_architecture_from_loader(loader)?, dtype),
    }
}
