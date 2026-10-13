//! LoRA adapter variant HTTP handlers.
//!
//! An adapter applies to a loaded base model as the variant
//! `<model>:<name>`: a model over the base's weights with the adapter
//! attached unmerged, with its own executor and KV cache. A request runs on
//! it by naming it as `model`, or by naming the adapter as `lora_adapter`.
//!
//! Endpoints:
//!   POST   /v1/lora                    Register an adapter as a variant and build it
//!   DELETE /v1/lora/{name}?model=<m>   Unregister the variant and drop its executor
//!   GET    /v1/lora[?model=<m>]        List the registered variants

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::generation::error_response;
use super::handlers::{lora_error_response, AppState};
use crate::engine::{lora::variant_id, AdapterSpec, LoraError, VariantInfo};

/// Request body for `POST /v1/lora`
#[derive(Deserialize)]
pub struct LoadLoraRequest {
    /// The base model, as requests name it. It loads when it is not loaded.
    pub model: String,
    /// The adapter name: requests reach the variant as `<model>:<name>` or
    /// with `lora_adapter: <name>`.
    pub name: String,
    /// The PEFT adapter directory, or its `adapter_model.safetensors` file.
    pub path: PathBuf,
}

/// One registered variant, as `GET /v1/lora` and `/v1/models` list it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LoraAdapterEntry {
    /// The variant id, `<model>:<name>`.
    pub id: String,
    /// The adapter name.
    pub name: String,
    /// The base model.
    pub base: String,
    /// The adapter directory or file.
    pub path: String,
    /// The adapter's `r`.
    pub rank: usize,
    /// How many base projections the adapter adapts.
    pub adapted_modules: usize,
    /// Whether the variant's executor is loaded now. A registered variant
    /// that is not loaded builds again on its next request.
    pub loaded: bool,
}

impl LoraAdapterEntry {
    /// The entry of `info`.
    pub fn new(info: VariantInfo, loaded: bool) -> Self {
        Self {
            id: info.id,
            name: info.adapter,
            base: info.base,
            path: info.path.display().to_string(),
            rank: info.rank,
            adapted_modules: info.adapted_modules,
            loaded,
        }
    }
}

/// Response from `GET /v1/lora`
#[derive(Serialize)]
pub struct ListLorasResponse {
    /// The `model` the listing was filtered to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub adapters: Vec<LoraAdapterEntry>,
}

/// The refusal for a variant of `model` when data-parallel replicas serve
/// it.
fn data_parallel_refusal(state: &AppState, model: &str) -> Option<LoraError> {
    let (replicated, replicas) = state.serving.replicated_model()?;
    (replicated == model).then(|| LoraError::data_parallel(model, replicas))
}

/// `POST /v1/lora`
///
/// Registers the PEFT adapter at `path` as the variant `<model>:<name>`,
/// loads `model` when needed, and builds the variant over its weights. A
/// variant of the same id is replaced. 404 for an unknown model, 400 for a
/// bad name, a model that takes no adapter or a data-parallel model, 422
/// for an adapter that does not load or fit.
pub async fn load_lora(
    State(state): State<Arc<AppState>>,
    Json(request): Json<LoadLoraRequest>,
) -> Response {
    let spec = match AdapterSpec::new(&request.name, request.path) {
        Ok(spec) => spec,
        Err(e) => return lora_error_response(&e),
    };
    if let Some(refusal) = data_parallel_refusal(&state, &request.model) {
        return lora_error_response(&refusal);
    }
    match state.scheduler.register_variant(&request.model, spec).await {
        Ok(info) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "model": info.base,
                "adapter": info.adapter,
                "id": info.id,
                "rank": info.rank,
                "adapted_modules": info.adapted_modules,
            })),
        )
            .into_response(),
        Err(e) => lora_error_response(&e),
    }
}

/// `DELETE /v1/lora/{name}?model=<model>`
///
/// Unregisters the variant `<model>:<name>` and drops its executor. 204 on
/// success, 404 with type `lora_not_found` when it is not registered.
pub async fn unload_lora(
    State(state): State<Arc<AppState>>,
    Path(adapter_name): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let Some(model_name) = params.get("model") else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Missing required query parameter: model",
            "invalid_request_error",
        );
    };
    let id = variant_id(model_name, &adapter_name);
    match state.scheduler.remove_variant(&id).await {
        Some(_) => StatusCode::NO_CONTENT.into_response(),
        None => error_response(
            StatusCode::NOT_FOUND,
            &format!("LoRA adapter '{adapter_name}' is not registered for model '{model_name}'"),
            "lora_not_found",
        ),
    }
}

/// `GET /v1/lora[?model=<model>]`
///
/// Lists the registered variants of `model`, or of every model. Loads
/// nothing.
pub async fn list_loras(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let model = params.get("model").cloned();
    let mut adapters = Vec::new();
    for info in state.scheduler.variants(model.as_deref()) {
        let loaded = state.scheduler.is_loaded(&info.id).await;
        adapters.push(LoraAdapterEntry::new(info, loaded));
    }
    (StatusCode::OK, Json(ListLorasResponse { model, adapters })).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Scheduler;
    use crate::server::handlers::ServerRuntime;

    fn test_state(dir: &std::path::Path) -> Arc<AppState> {
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
        Arc::new(AppState::new(scheduler, metrics_handle))
    }

    fn registered(base: &str, adapter: &str) -> VariantInfo {
        VariantInfo {
            id: variant_id(base, adapter),
            base: base.into(),
            adapter: adapter.into(),
            path: format!("/adapters/{adapter}").into(),
            rank: 16,
            adapted_modules: 4,
        }
    }

    async fn json(response: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        serde_json::from_slice(&bytes).expect("json")
    }

    fn query(pairs: &[(&str, &str)]) -> Query<HashMap<String, String>> {
        Query(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    #[tokio::test]
    async fn lists_and_removes_registered_variants() {
        let dir = tempfile::tempdir().expect("temp dir");
        let state = test_state(dir.path());
        state.scheduler.lora.insert(registered("m", "kev"));
        state.scheduler.lora.insert(registered("n", "jev"));

        let all = json(list_loras(State(Arc::clone(&state)), query(&[])).await).await;
        assert!(all.get("model").is_none());
        assert_eq!(all["adapters"].as_array().map(Vec::len), Some(2));
        let of_m =
            json(list_loras(State(Arc::clone(&state)), query(&[("model", "m")])).await).await;
        assert_eq!(of_m["model"], "m");
        assert_eq!(of_m["adapters"][0]["id"], "m:kev");
        assert_eq!(of_m["adapters"][0]["rank"], 16);
        assert_eq!(of_m["adapters"][0]["loaded"], false);

        let removed = unload_lora(
            State(Arc::clone(&state)),
            Path("kev".into()),
            query(&[("model", "m")]),
        )
        .await;
        assert_eq!(removed.status(), StatusCode::NO_CONTENT);
        let again = unload_lora(
            State(Arc::clone(&state)),
            Path("kev".into()),
            query(&[("model", "m")]),
        )
        .await;
        assert_eq!(again.status(), StatusCode::NOT_FOUND);
        let no_model = unload_lora(State(Arc::clone(&state)), Path("kev".into()), query(&[])).await;
        assert_eq!(no_model.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn a_bad_name_or_an_unknown_model_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        let state = test_state(dir.path());
        let request = |model: &str, name: &str| {
            Json(LoadLoraRequest {
                model: model.into(),
                name: name.into(),
                path: "/adapters/kev".into(),
            })
        };
        let bad = load_lora(State(Arc::clone(&state)), request("m", "a:b")).await;
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
        let unknown = load_lora(State(Arc::clone(&state)), request("absent", "kev")).await;
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
        state.scheduler.lora.insert(registered("m", "kev"));
        let stacked = load_lora(State(Arc::clone(&state)), request("m:kev", "jev")).await;
        assert_eq!(stacked.status(), StatusCode::BAD_REQUEST);
    }
}
