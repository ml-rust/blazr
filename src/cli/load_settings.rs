//! Resolved settings for the commands that load one model in process:
//! `run`, `generate`, `chat` and `bench`.

use anyhow::{anyhow, Result};

use super::RuntimeArgs;
use crate::config::resolve::{resolve_dtype, resolve_num_ctx, resolve_run_device};
use crate::config::{
    DeviceChoice, DtypeRequest, FileConfig, InferenceConfig, InferenceOverlay, VisionOverrides,
};

/// What every model-loading command applies after the load.
#[derive(Debug, Clone, Default)]
pub struct LoadSettings {
    /// Initial KV cache size: `--num-ctx`, then `num_ctx`, then 2048.
    pub num_ctx: usize,
    /// The user config file's `inference` keys.
    pub inference: InferenceOverlay,
    /// `--mmproj` and the image token flags.
    pub vision: VisionOverrides,
    /// `--dtype`, then `inference.dtype`. The loader reads it before any
    /// weight loads. `None` takes the model's `torch_dtype`, else f32.
    pub dtype: Option<DtypeRequest>,
}

impl LoadSettings {
    /// Resolve for a command without device flags.
    pub fn resolve(
        num_ctx: Option<usize>,
        vision: VisionOverrides,
        dtype: Option<boostr::DType>,
        file: &FileConfig,
    ) -> Self {
        Self {
            num_ctx: resolve_num_ctx(num_ctx, file),
            inference: file.inference.clone(),
            vision,
            dtype: resolve_dtype(dtype, file),
        }
    }

    /// Write the file's inference keys, then the vision flags, over a
    /// loaded model's config.
    pub fn apply(&self, inference: &mut InferenceConfig) -> Result<()> {
        self.inference.apply(inference)?;
        self.vision.apply(inference);
        Ok(())
    }
}

/// The GPU index for `chat` and `bench`, from the `device` key.
pub fn build_gpu_index(file: &FileConfig) -> Result<usize> {
    DeviceChoice::build_gpu_index(file.device).map_err(|e| anyhow!(e))
}

/// Settings for `run` and `generate`.
#[derive(Debug, Clone)]
pub struct RunSettings {
    /// `--cpu`, then `device`, then GPU 0 of the build, else the CPU.
    pub device: DeviceChoice,
    /// `--gpu-layers`. No key sets it.
    pub gpu_layers: i32,
    /// `--paged-attention`. Absent leaves `inference.paged_attention`.
    pub paged_attention: bool,
    /// `--graphs`. Absent leaves `inference.graphs`.
    pub graphs: bool,
    pub load: LoadSettings,
}

impl RuntimeArgs {
    /// Resolve the runtime flags over the user config file.
    pub fn resolve(self, file: &FileConfig) -> RunSettings {
        RunSettings {
            device: resolve_run_device(self.cpu, file),
            gpu_layers: self.gpu_layers,
            paged_attention: self.paged_attention,
            graphs: self.graphs,
            load: LoadSettings::resolve(
                self.num_ctx,
                self.vision.into_overrides(),
                self.dtype.dtype,
                file,
            ),
        }
    }
}

impl RunSettings {
    /// Apply the file keys, then each given flag, to a loaded model's
    /// config. A switch can only turn its key on.
    pub fn apply(&self, inference: &mut InferenceConfig) -> Result<()> {
        self.load.apply(inference)?;
        inference.paged_attention |= self.paged_attention;
        inference.graphs |= self.graphs;
        Ok(())
    }
}
