//! The one runtime-selection point.
//!
//! The build fixes the GPU runtime of `serve`, `chat`, `bench` and the
//! cluster commands. Priority when several features are on: `cuda`, then
//! `wgpu`, then the CPU. `run` and `generate` also take `--cpu` or
//! `device: cpu`, which runs the CPU runtime on any build.
//!
//! `wgpu` covers Metal (Apple), Vulkan (Android, Linux, AMD, Intel,
//! NVIDIA), DX12 (Windows) and WebGPU (browser).

use anyhow::Result;

/// The runtime of every command whose runtime the build fixes.
#[cfg(feature = "cuda")]
pub type ServerRuntime = boostr::CudaRuntime;
/// The runtime of every command whose runtime the build fixes.
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type ServerRuntime = boostr::runtime::wgpu::WgpuRuntime;
/// The runtime of every command whose runtime the build fixes.
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type ServerRuntime = boostr::CpuRuntime;

/// The device type of [`ServerRuntime`].
pub type ServerDevice = <ServerRuntime as boostr::Runtime>::Device;

/// The GPU backend this build runs on, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA CUDA.
    Cuda,
    /// WebGPU over Metal, Vulkan, DX12 or the browser.
    Wgpu,
}

impl GpuBackend {
    /// The backend [`ServerRuntime`] uses, or `None` in a CPU build.
    pub const fn of_build() -> Option<Self> {
        if cfg!(feature = "cuda") {
            Some(Self::Cuda)
        } else if cfg!(feature = "wgpu") {
            Some(Self::Wgpu)
        } else {
            None
        }
    }

    /// The name the `device` key and the build features use.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
        }
    }
}

/// The name of the runtime [`ServerRuntime`] is: `cuda`, `wgpu` or `cpu`.
pub const fn backend_name() -> &'static str {
    match GpuBackend::of_build() {
        Some(gpu) => gpu.name(),
        None => "cpu",
    }
}

/// Open device `index` of [`ServerRuntime`].
///
/// A CPU build ignores `index`. A `wgpu` build opens the adapter now.
///
/// # Errors
///
/// A `wgpu` build returns an error when no adapter answers at `index`.
pub fn open_device(index: usize) -> Result<ServerDevice> {
    #[cfg(feature = "cuda")]
    {
        Ok(boostr::CudaDevice::new(index))
    }
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    {
        open_wgpu_device(index)
    }
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    {
        let _ = index;
        Ok(boostr::CpuDevice::new())
    }
}

/// Open WebGPU adapter `index` and return its device.
///
/// The device carries the adapter name for log lines.
///
/// # Errors
///
/// Returns an error naming the adapter index, the driver error and the next
/// action when no adapter answers.
#[cfg(feature = "wgpu")]
pub fn open_wgpu_device(index: usize) -> Result<boostr::runtime::wgpu::WgpuDevice> {
    use boostr::runtime::wgpu::{WgpuClient, WgpuDevice};
    use boostr::RuntimeClient;

    let client = WgpuClient::new(WgpuDevice::new(index)).map_err(|e| {
        anyhow::anyhow!(
            "wgpu adapter {index} did not open: {e}. Install a Vulkan, Metal or DX12 driver, \
             set device: wgpu:N for another adapter, or pass --cpu"
        )
    })?;
    let device = client.device().clone();
    tracing::info!("wgpu adapter {index}: {}", device.adapter_name());
    Ok(device)
}

/// Compile every CUDA inference module for `device` now, so the first
/// forward pass compiles none.
///
/// # Errors
///
/// A CUDA client that does not open on `device`, or a module that does
/// not load.
#[cfg(feature = "cuda")]
pub fn preload_cuda_modules(device: &boostr::CudaDevice) -> Result<()> {
    let client = boostr::CudaClient::new(device.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create CUDA client for preload: {e}"))?;
    boostr::preload_inference_modules(&client)
        .map_err(|e| anyhow::anyhow!("Failed to preload CUDA modules: {e}"))?;
    tracing::debug!("Pre-loaded all CUDA PTX modules");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_follows_feature_priority() {
        let expected = if cfg!(feature = "cuda") {
            "cuda"
        } else if cfg!(feature = "wgpu") {
            "wgpu"
        } else {
            "cpu"
        };
        assert_eq!(backend_name(), expected);
        assert_eq!(<ServerRuntime as boostr::Runtime>::name(), expected);
    }
}
