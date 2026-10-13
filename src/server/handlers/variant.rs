//! The executor a generation request runs on: its `model`, or the LoRA
//! variant that its `model` (`<model>:<adapter>`) or its `lora_adapter`
//! selects. Also the error response of each lookup failure.

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::Response;

use super::state::{AppState, ServerRuntime};
use crate::engine::{Executor, LoraError};
use crate::server::generation::error_response;

/// The status and error `type` of a LoRA failure.
fn status_of(error: &LoraError) -> (StatusCode, &'static str) {
    match error {
        LoraError::ModelNotFound(_) => (StatusCode::NOT_FOUND, "invalid_request_error"),
        LoraError::AdapterNotFound(_) => (StatusCode::NOT_FOUND, "lora_not_found"),
        LoraError::Invalid(_) | LoraError::Unsupported(_) => {
            (StatusCode::BAD_REQUEST, "invalid_request_error")
        }
        LoraError::Adapter(_) => (StatusCode::UNPROCESSABLE_ENTITY, "lora_load_error"),
        LoraError::BaseLoad(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
    }
}

/// The response of a LoRA failure: 404 for an unknown model or adapter,
/// 400 for a malformed selection or a base that takes no adapter, 422 for
/// an adapter that does not load or fit, 500 for a base that does not
/// load.
pub fn lora_error_response(error: &LoraError) -> Response {
    let (status, kind) = status_of(error);
    error_response(status, error.message(), kind)
}

/// The response of a failed executor lookup: a [`LoraError`] by its kind,
/// any other failure a 404.
pub fn lookup_error_response(error: &anyhow::Error) -> Response {
    match error.downcast_ref::<LoraError>() {
        Some(lora) => lora_error_response(lora),
        None => error_response(
            StatusCode::NOT_FOUND,
            &format!("Model not found: {error}"),
            "invalid_request_error",
        ),
    }
}

/// The executor for a request naming `model` and `lora_adapter`, loading
/// the model or building the variant when needed. A request without
/// `lora_adapter` takes the live `generation.lora_adapter`.
///
/// # Errors
///
/// The error response to send: see [`lora_error_response`] and
/// [`lookup_error_response`]. An unknown adapter is a 404, never the base
/// model.
pub async fn select_executor(
    state: &AppState,
    model: &str,
    lora_adapter: Option<&str>,
) -> Result<Arc<Executor<ServerRuntime>>, Box<Response>> {
    let config = state.config();
    let name = state
        .scheduler
        .select_model(
            model,
            lora_adapter,
            config.generation.lora_adapter.as_deref(),
        )
        .map_err(|e| Box::new(lora_error_response(&e)))?;
    state
        .scheduler
        .get_executor(&name)
        .await
        .map_err(|e| Box::new(lookup_error_response(&e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Scheduler, VariantInfo};

    fn test_state(dir: &std::path::Path) -> AppState {
        // A device handle only. No model loads, so no adapter opens.
        #[cfg(feature = "cuda")]
        let device = boostr::CudaDevice::new(0);
        #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
        let device = boostr::runtime::wgpu::WgpuDevice::new(0);
        #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
        let device = boostr::CpuDevice::new();
        let scheduler = Arc::new(Scheduler::<ServerRuntime>::new(dir.to_path_buf(), device));
        let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        AppState::new(scheduler, metrics_handle)
    }

    #[test]
    fn each_failure_has_its_status() {
        let cases = [
            (
                LoraError::ModelNotFound("m".into()),
                404,
                "invalid_request_error",
            ),
            (
                LoraError::AdapterNotFound("m".into()),
                404,
                "lora_not_found",
            ),
            (LoraError::Invalid("m".into()), 400, "invalid_request_error"),
            (
                LoraError::Unsupported("m".into()),
                400,
                "invalid_request_error",
            ),
            (LoraError::Adapter("m".into()), 422, "lora_load_error"),
            (LoraError::BaseLoad("m".into()), 500, "server_error"),
        ];
        for (error, status, kind) in cases {
            assert_eq!(
                status_of(&error),
                (StatusCode::from_u16(status).unwrap(), kind)
            );
            assert_eq!(lora_error_response(&error).status().as_u16(), status);
            let wrapped = anyhow::Error::from(error);
            assert_eq!(lookup_error_response(&wrapped).status().as_u16(), status);
        }
        let other = anyhow::anyhow!("Model not found: x");
        assert_eq!(
            lookup_error_response(&other).status(),
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn selection_failures_answer_before_any_load() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir(dir.path().join("m")).expect("mkdir");
        std::fs::write(dir.path().join("m/config.json"), "{}").expect("write");
        let state = test_state(dir.path());
        state.scheduler.lora.insert(VariantInfo {
            id: "m:kev".into(),
            base: "m".into(),
            adapter: "kev".into(),
            path: "/adapters/kev".into(),
            rank: 16,
            adapted_modules: 4,
        });

        let status = |r: Result<_, Box<Response>>| r.err().map(|r| r.status().as_u16());
        assert_eq!(
            status(select_executor(&state, "m", Some("nope")).await),
            Some(404)
        );
        assert_eq!(
            status(select_executor(&state, "m:nope", None).await),
            Some(404)
        );
        assert_eq!(
            status(select_executor(&state, "m:kev", Some("jev")).await),
            Some(400)
        );
        assert_eq!(
            status(select_executor(&state, "absent", None).await),
            Some(404)
        );
    }
}
