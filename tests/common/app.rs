//! An in-process ASR-only `AppState` and the API router over it.

use std::path::Path;
use std::sync::Arc;

use axum::Router;

use blazr::engine::Scheduler;
use blazr::server::handlers::{AppState, ServerRuntime};

/// An [`AppState`] with no text model (an unused scheduler model directory,
/// matching `server::handlers::auxiliary::tests::test_state`) and, when
/// `asr` is given, one real Whisper bundle loaded from its checkpoint
/// directory and registered under `asr.0`.
pub async fn asr_app_state(asr: Option<(&str, &Path)>) -> Arc<AppState> {
    let device = blazr::runtime::open_device(0)
        .unwrap_or_else(|e| panic!("opening the server runtime's device 0: {e}"));

    let scheduler = Arc::new(Scheduler::<ServerRuntime>::new(
        std::env::temp_dir().join("blazr-audio-test-models"),
        device.clone(),
    ));
    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    let state = AppState::new(scheduler, metrics_handle);

    if let Some((model_name, dir)) = asr {
        let bundle = blazr::loader::load_whisper_from_dir::<ServerRuntime, _>(dir, &device)
            .unwrap_or_else(|e| panic!("loading whisper checkpoint {}: {e}", dir.display()));
        state
            .register_asr_model(model_name.to_string(), Arc::new(bundle), dir.to_path_buf())
            .await;
    }

    Arc::new(state)
}

/// The full API router (no auth layer — these tests call it in-process,
/// never over a socket), wired to `state`.
pub fn audio_router(state: Arc<AppState>) -> Router {
    blazr::server::api_routes().with_state(state)
}
