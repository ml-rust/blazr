//! Opaque model handles.
//!
//! A handle is an id, never a pointer to memory. A table maps live ids to
//! models. Unload removes the id, so a second unload, or any call with a
//! stale handle, returns `BLAZR_STATUS_INVALID_HANDLE`. Ids never repeat in
//! one process, so a stale handle never reaches a newer model.
//!
//! Each call holds its own reference to the model. An unload during a
//! generation on another thread frees the model when that generation ends.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, CpuRuntime, DType, NormalizationOps, Runtime, SamplingOps,
    ScalarOps, TypeConversionOps, UnaryOps,
};

use super::error::{BlazrStatus, FfiError};
use crate::engine::Executor;

/// An opaque loaded model. C code holds only a pointer to it.
pub struct BlazrModel {
    _private: [u8; 0],
}

/// Every client trait the engine needs from a runtime.
pub(crate) trait EngineClient<R: Runtime<DType = DType>>:
    TensorOps<R>
    + ScalarOps<R>
    + ConvOps<R>
    + NormalizationOps<R>
    + UnaryOps<R>
    + ActivationOps<R>
    + BinaryOps<R>
    + TypeConversionOps<R>
    + SamplingOps<R>
    + boostr::GrammarDfaOps<R>
    + ModelClient<R>
    + boostr::RoPEPackedOps<R>
    + boostr::quant::DequantOps<R>
    + boostr::quant::QuantMatmulOps<R>
{
}

impl<R, C> EngineClient<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::RoPEPackedOps<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
}

/// An executor on whichever runtime the load options chose.
pub(crate) enum AnyExecutor {
    Cpu(Executor<CpuRuntime>),
    #[cfg(feature = "wgpu")]
    Wgpu(Executor<boostr::runtime::wgpu::WgpuRuntime>),
    #[cfg(feature = "cuda")]
    Cuda(Executor<boostr::CudaRuntime>),
}

/// Run `$body` with `$exec` bound to the executor, whatever its runtime.
macro_rules! with_executor {
    ($any:expr, $exec:ident => $body:expr) => {
        match $any {
            $crate::capi::handle::AnyExecutor::Cpu($exec) => $body,
            #[cfg(feature = "wgpu")]
            $crate::capi::handle::AnyExecutor::Wgpu($exec) => $body,
            #[cfg(feature = "cuda")]
            $crate::capi::handle::AnyExecutor::Cuda($exec) => $body,
        }
    };
}
pub(crate) use with_executor;

/// A loaded model and what its calls share.
pub(crate) struct ModelInner {
    pub executor: AnyExecutor,
    /// Drives the engine's async generation stream on the calling thread.
    pub runtime: tokio::runtime::Runtime,
    /// Set by `blazr_cancel`. Each generation clears it when it starts.
    pub cancel: AtomicBool,
}

fn table() -> &'static Mutex<HashMap<usize, Arc<ModelInner>>> {
    static TABLE: OnceLock<Mutex<HashMap<usize, Arc<ModelInner>>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_table() -> std::sync::MutexGuard<'static, HashMap<usize, Arc<ModelInner>>> {
    // A panic never happens while the lock is held, but recover anyway.
    table()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

/// Add a model to the table and return its handle.
pub(crate) fn register(inner: ModelInner) -> *mut BlazrModel {
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    lock_table().insert(id, Arc::new(inner));
    std::ptr::without_provenance_mut(id)
}

fn invalid_handle(model: *const BlazrModel) -> FfiError {
    FfiError::new(
        BlazrStatus::InvalidHandle,
        format!(
            "model handle {model:p} is unknown or already unloaded. Load a model with \
             blazr_model_load"
        ),
    )
}

/// The model behind `model`, kept alive for this call.
pub(crate) fn lookup(model: *const BlazrModel) -> Result<Arc<ModelInner>, FfiError> {
    lock_table()
        .get(&(model as usize))
        .cloned()
        .ok_or_else(|| invalid_handle(model))
}

/// Remove `model` from the table. Its memory frees when the last call
/// that holds it returns.
pub(crate) fn remove(model: *const BlazrModel) -> Result<Arc<ModelInner>, FfiError> {
    lock_table()
        .remove(&(model as usize))
        .ok_or_else(|| invalid_handle(model))
}
