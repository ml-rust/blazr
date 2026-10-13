//! The executor a generation request runs on: its `model`, or the LoRA
//! variant that its `model` (`<model>:<adapter>`) or its `lora_adapter`
//! selects. Also the classification and the OpenAI error response of each
//! lookup failure.

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::Response;

use super::state::{AppState, ServerRuntime};
use crate::engine::{is_out_of_memory, Executor, LoraError, ModelNotFound};
use crate::server::generation::error_response_with_code;

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
    respond(&lora_failure(error))
}

/// The classification of a LoRA failure: see [`status_of`]. An unknown
/// model also carries the `model_not_found` code.
fn lora_failure(error: &LoraError) -> LookupFailure {
    let (status, kind) = status_of(error);
    LookupFailure {
        status,
        kind,
        code: matches!(error, LoraError::ModelNotFound(_)).then_some(MODEL_NOT_FOUND),
        message: error.message().to_string(),
    }
}

/// The OpenAI error response of `failure`.
fn respond(failure: &LookupFailure) -> Response {
    error_response_with_code(failure.status, &failure.message, failure.kind, failure.code)
}

/// How a failed executor lookup answers, before an endpoint puts it in
/// its own error envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupFailure {
    /// The HTTP status.
    pub status: StatusCode,
    /// The OpenAI error `type`.
    pub kind: &'static str,
    /// The OpenAI error `code`, when the failure has one.
    pub code: Option<&'static str>,
    /// The caller-facing message.
    pub message: String,
}

/// Classify a failed executor lookup: a [`LoraError`] by its kind, an
/// unknown model ([`ModelNotFound`]) a 404 `model_not_found`, a load the
/// device could not hold a 503, and any other load failure a 500 with its
/// message.
pub fn classify_lookup(error: &anyhow::Error) -> LookupFailure {
    if let Some(lora) = error.downcast_ref::<LoraError>() {
        return lora_failure(lora);
    }
    if let Some(missing) = error.downcast_ref::<ModelNotFound>() {
        return LookupFailure {
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            code: Some(MODEL_NOT_FOUND),
            message: missing.to_string(),
        };
    }
    let (status, code) = if is_out_of_memory(error) {
        (StatusCode::SERVICE_UNAVAILABLE, "model_load_out_of_memory")
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, "model_load_failed")
    };
    LookupFailure {
        status,
        kind: "server_error",
        code: Some(code),
        message: format!("{error:#}"),
    }
}

/// The OpenAI `code` of an unknown model.
const MODEL_NOT_FOUND: &str = "model_not_found";

/// The OpenAI error response of a failed executor lookup: see
/// [`classify_lookup`].
pub fn lookup_error_response(error: &anyhow::Error) -> Response {
    respond(&classify_lookup(error))
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
    }

    /// The JSON body of `response`.
    async fn body_of(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body reads");
        serde_json::from_slice(&bytes).expect("body is JSON")
    }

    #[test]
    fn lookups_are_classified_by_their_typed_error() {
        let missing = classify_lookup(&ModelNotFound::new("x").into());
        assert_eq!(
            missing,
            LookupFailure {
                status: StatusCode::NOT_FOUND,
                kind: "invalid_request_error",
                code: Some("model_not_found"),
                message: "Model not found: x".into(),
            }
        );

        let lora = classify_lookup(&LoraError::ModelNotFound("no m".into()).into());
        assert_eq!(
            (lora.status, lora.code),
            (StatusCode::NOT_FOUND, Some("model_not_found"))
        );

        let load = anyhow::anyhow!("weights do not read").context("model 'x'");
        assert_eq!(
            classify_lookup(&load),
            LookupFailure {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                kind: "server_error",
                code: Some("model_load_failed"),
                message: "model 'x': weights do not read".into(),
            }
        );

        let oom = anyhow::Error::from(boostr::NumrError::OutOfMemory { size: 1 << 30 })
            .context("model 'x'");
        let oom = classify_lookup(&oom);
        assert_eq!(
            (oom.status, oom.kind, oom.code),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                Some("model_load_out_of_memory")
            )
        );
    }

    #[tokio::test]
    async fn lookup_responses_carry_one_prefix_and_the_code() {
        let missing = lookup_error_response(&ModelNotFound::new("x").into());
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body_of(missing).await,
            serde_json::json!({"error": {
                "message": "Model not found: x",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }})
        );

        let load = lookup_error_response(&anyhow::anyhow!("weights do not read"));
        assert_eq!(load.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            body_of(load).await,
            serde_json::json!({"error": {
                "message": "weights do not read",
                "type": "server_error",
                "code": "model_load_failed",
            }})
        );
    }

    #[tokio::test]
    async fn an_unknown_model_is_a_404_and_a_failed_load_a_500() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir(dir.path().join("broken")).expect("mkdir");
        std::fs::write(dir.path().join("broken/config.json"), "{}").expect("write");
        let state = test_state(dir.path());

        let missing = select_executor(&state, "absent", None)
            .await
            .err()
            .expect("refused");
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
        let body = body_of(*missing).await;
        assert_eq!(body["error"]["message"], "Model not found: absent");
        assert_eq!(body["error"]["code"], "model_not_found");

        let broken = select_executor(&state, "broken", None)
            .await
            .err()
            .expect("an empty config does not load");
        assert_eq!(broken.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_of(*broken).await;
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "model_load_failed");
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(!message.is_empty(), "{body}");
        assert!(!message.contains("Model not found"), "{body}");
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
