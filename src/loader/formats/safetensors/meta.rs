//! The config of a SafeTensors checkpoint, read with no weights.

use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::{DType, Runtime};

use crate::config::BlazrConfig;

use super::config::checkpoint_compute_dtype;
use super::regular::{config_dir_of, resolve_config};

/// The config of the checkpoint at `path` on runtime `R`, as a load
/// resolves it, without reading a tensor.
///
/// The SafeTensors headers are read only when no `config.json` describes
/// the model.
///
/// # Errors
///
/// A path that holds no SafeTensors checkpoint, or a config that does not parse.
pub fn read_safetensors_config<R: Runtime<DType = DType>>(path: &Path) -> Result<BlazrConfig> {
    let config_dir = config_dir_of(path);
    let loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {e}"))?;
    let dtype = checkpoint_compute_dtype::<R>(config_dir, None);
    resolve_config(&loader, config_dir, dtype)
}
