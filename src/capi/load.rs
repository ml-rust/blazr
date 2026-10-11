//! `blazr_model_load` and `blazr_model_unload`.

use std::ffi::c_char;
use std::path::Path;
use std::sync::atomic::AtomicBool;

use super::error::{guard, BlazrStatus, FfiError};
use super::handle::{self, AnyExecutor, BlazrModel, ModelInner};
use super::string::{read_optional_str, read_str, write_out};
use crate::config::{parse_dtype_for, DeviceChoice, DtypeRequest, DtypeSource, DEFAULT_NUM_CTX};
use crate::engine::{load_executor, LoadOptions};

/// How `blazr_model_load` loads a model. Start from
/// `blazr_load_options_default` and set the fields you need.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlazrLoadOptions {
    /// `cpu`, `cuda`, `cuda:N`, `wgpu` or `wgpu:N`, as UTF-8. Null picks
    /// GPU 0 of the library build, else the CPU.
    pub device: *const c_char,
    /// `f32`, `bf16` or `f16`. Null takes the checkpoint's dtype.
    pub dtype: *const c_char,
    /// KV cache context length in tokens. Zero takes the default, 2048.
    pub num_ctx: u32,
}

/// The default load options: build-default device, checkpoint dtype,
/// 2048-token context.
#[no_mangle]
pub extern "C" fn blazr_load_options_default() -> BlazrLoadOptions {
    BlazrLoadOptions {
        device: std::ptr::null(),
        dtype: std::ptr::null(),
        num_ctx: 0,
    }
}

/// Load the GGUF file or SafeTensors directory at `path`.
///
/// `options` can be null for the defaults. On success `*out_model` holds a
/// handle for `blazr_model_unload`. On error `*out_model` is null.
///
/// # Safety
///
/// `path` is a NUL-terminated string. `options` is null or points to a
/// `BlazrLoadOptions`. `out_model` points to writable storage.
#[no_mangle]
pub unsafe extern "C" fn blazr_model_load(
    path: *const c_char,
    options: *const BlazrLoadOptions,
    out_model: *mut *mut BlazrModel,
) -> BlazrStatus {
    guard(|| {
        // SAFETY: the caller guarantees `out_model` is null or writable.
        unsafe { write_out(out_model, std::ptr::null_mut(), "out_model") }?;
        // SAFETY: forwarded from the caller.
        let path = unsafe { read_str(path, "path") }?;
        let options = if options.is_null() {
            blazr_load_options_default()
        } else {
            // SAFETY: checked non-null, the caller guarantees it is valid.
            unsafe { *options }
        };
        // SAFETY: the option strings are null or NUL-terminated.
        let device = unsafe { read_optional_str(options.device, "options.device") }?;
        // SAFETY: as above.
        let dtype = unsafe { read_optional_str(options.dtype, "options.dtype") }?;

        let device = match device {
            Some(text) => DeviceChoice::parse(text).map_err(FfiError::invalid)?,
            None => DeviceChoice::build_default(),
        };
        let dtype = dtype
            .map(|text| {
                parse_dtype_for(text, "options.dtype").map(|dtype| DtypeRequest {
                    dtype,
                    source: DtypeSource::Flag,
                })
            })
            .transpose()
            .map_err(|e| FfiError::invalid(format!("{e:#}")))?;
        let load = LoadOptions {
            dtype,
            num_ctx: match options.num_ctx {
                0 => DEFAULT_NUM_CTX,
                n => n as usize,
            },
            warmup: true,
        };

        let executor = open(Path::new(path), device, &load)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .map_err(|e| {
                FfiError::new(
                    BlazrStatus::Load,
                    format!("the async runtime for generation did not start: {e}"),
                )
            })?;
        let model = handle::register(ModelInner {
            executor,
            runtime,
            cancel: AtomicBool::new(false),
        });
        // SAFETY: checked writable above.
        unsafe { write_out(out_model, model, "out_model") }?;
        Ok(BlazrStatus::Ok)
    })
}

fn load_error(e: anyhow::Error) -> FfiError {
    FfiError::new(BlazrStatus::Load, format!("{e:#}"))
}

fn unsupported_device(device: DeviceChoice) -> FfiError {
    FfiError::new(
        BlazrStatus::Unsupported,
        format!(
            "device {device} needs a library built with --features {}. Pick another device",
            device.backend_name()
        ),
    )
}

fn open(path: &Path, device: DeviceChoice, load: &LoadOptions) -> Result<AnyExecutor, FfiError> {
    match device {
        DeviceChoice::Cpu => {
            load_executor::<boostr::CpuRuntime>(path, boostr::CpuDevice::new(), load)
                .map(AnyExecutor::Cpu)
                .map_err(load_error)
        }
        #[cfg(feature = "wgpu")]
        DeviceChoice::Wgpu(index) => {
            let device = crate::runtime::open_wgpu_device(index).map_err(load_error)?;
            load_executor::<boostr::runtime::wgpu::WgpuRuntime>(path, device, load)
                .map(AnyExecutor::Wgpu)
                .map_err(load_error)
        }
        #[cfg(feature = "cuda")]
        DeviceChoice::Cuda(index) => {
            load_executor::<boostr::CudaRuntime>(path, boostr::CudaDevice::new(index), load)
                .map(AnyExecutor::Cuda)
                .map_err(load_error)
        }
        #[allow(unreachable_patterns)]
        other => Err(unsupported_device(other)),
    }
}

/// Unload a model. Its memory frees once no call on another thread uses it.
///
/// A second unload of the same handle returns
/// `BLAZR_STATUS_INVALID_HANDLE` and does nothing else. blazr never reads
/// through `model`, so any pointer value is safe to pass.
#[no_mangle]
pub extern "C" fn blazr_model_unload(model: *mut BlazrModel) -> BlazrStatus {
    guard(|| {
        drop(handle::remove(model)?);
        Ok(BlazrStatus::Ok)
    })
}
