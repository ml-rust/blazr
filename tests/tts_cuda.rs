//! Kokoro and VITS on CUDA device 0, loaded through the loaders `blazr
//! serve` and `blazr speak` use.
//!
//! Each engine synthesizes a short phrase on the loading thread. The same
//! bundle then answers `POST /v1/audio/speech` in-process, buffered and
//! streamed. The handler renders on a fresh blocking worker thread, so the
//! route checks that the worker binds the device before it runs the model.
//!
//! Fixtures, each overridable by its own var (see `tests/common/mod.rs`):
//! * `KOKORO_DIR` → `Kokoro-82M`. Its G2P models come from `g2p/` beside
//!   the weights, the hub cache, or a download.
//! * `MMS_TTS_ZLM_DIR` → `facebook/mms-tts-zlm`
//!
//! A test whose model or GPU is missing prints a skip notice and passes.
#![cfg(all(feature = "audio", feature = "cuda"))]

mod common;

use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use boostr::{CudaDevice, CudaRuntime};
use tower::ServiceExt;
use voxr::voxcpm::VoxCpm2SynthOptions;
use voxr::SynthesizeOptions;

use blazr::loader::hub::ModelSearch;
use blazr::loader::{load_kokoro_engine, load_tts_from_dir};
use common::{asr_app_state, audio_router, hub_online, model_fixture, skip_notice};

const KOKORO_VOICE: &str = "af_heart";
const KOKORO_PHRASE: &str = "The river runs north.";
const VITS_MODEL: &str = "facebook/mms-tts-zlm";
const VITS_VOICE: &str = "zlm";
const VITS_PHRASE: &str = "Sungai itu mengalir ke utara.";

/// Whether CUDA device 0 is usable. Prints a skip notice when it is not.
fn gpu() -> bool {
    let ok = boostr::runtime::cuda::is_cuda_available();
    if !ok {
        eprintln!("skipping: no CUDA device visible");
    }
    ok
}

/// The Kokoro checkpoint, or `None` with a notice when it is missing or its
/// G2P models can neither be found beside it nor downloaded.
fn kokoro() -> Option<std::path::PathBuf> {
    let Some(dir) = model_fixture("KOKORO_DIR", "Kokoro-82M") else {
        skip_notice("Kokoro-82M checkpoint", "KOKORO_DIR");
        return None;
    };
    if !dir.join("g2p").is_dir() && !hub_online() {
        eprintln!(
            "skipping: {} holds no g2p/ and the Hub is out of reach",
            dir.display()
        );
        return None;
    }
    Some(dir)
}

fn assert_audio(samples: &[f32], what: &str) {
    assert!(!samples.is_empty(), "{what}: no samples");
    assert!(
        samples.iter().all(|s| s.is_finite()),
        "{what}: a sample is not finite"
    );
}

/// Loads `dir` through the serve loader on CUDA device 0, synthesizes
/// `phrase` on this thread, then asks the route for it, buffered and
/// streamed.
async fn serve_on_cuda(dir: &Path, name: &str, voice: &str, phrase: &str) {
    let device = CudaDevice::new(0);
    let bundle = load_tts_from_dir(dir, VoxCpm2SynthOptions::default(), &device)
        .unwrap_or_else(|e| panic!("loading {} on CUDA: {e:#}", dir.display()));
    assert!(bundle.has_engine(), "{name}: no engine loaded");
    let samples = bundle
        .synthesize(phrase, voice, &SynthesizeOptions::default())
        .unwrap_or_else(|e| panic!("{name}: synthesizing on CUDA: {e}"));
    assert_audio(&samples, name);

    let state = asr_app_state(None).await;
    state
        .register_tts_model(name.to_string(), Arc::new(bundle), dir.to_path_buf())
        .await;
    for stream in [false, true] {
        let body = serde_json::json!({
            "model": name,
            "input": phrase,
            "voice": voice,
            "response_format": "pcm",
            "stream": stream,
        });
        let request = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap_or_else(|e| panic!("building the request: {e}"));
        let response = audio_router(Arc::clone(&state))
            .oneshot(request)
            .await
            .unwrap_or_else(|e| panic!("router call: {e}"));
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap_or_else(|e| panic!("reading the body: {e}"));
        assert_eq!(
            status,
            StatusCode::OK,
            "{name} (stream {stream}): {}",
            String::from_utf8_lossy(&bytes)
        );
        assert!(!bytes.is_empty(), "{name} (stream {stream}): empty body");
    }
}

#[tokio::test]
async fn kokoro_serves_on_cuda() {
    if !gpu() {
        return;
    }
    let Some(dir) = kokoro() else {
        return;
    };
    serve_on_cuda(&dir, "kokoro", KOKORO_VOICE, KOKORO_PHRASE).await;
}

#[tokio::test]
async fn vits_serves_on_cuda() {
    if !gpu() {
        return;
    }
    let Some(dir) = model_fixture("MMS_TTS_ZLM_DIR", VITS_MODEL) else {
        skip_notice("facebook/mms-tts-zlm checkpoint", "MMS_TTS_ZLM_DIR");
        return;
    };
    serve_on_cuda(&dir, "mms-tts-zlm", VITS_VOICE, VITS_PHRASE).await;
}

/// The loader `blazr speak` uses, on CUDA: a seeded take is finite and
/// repeats exactly on the same device.
#[test]
fn kokoro_speak_loader_runs_on_cuda() {
    if !gpu() {
        return;
    }
    let Some(dir) = kokoro() else {
        return;
    };
    let cwd =
        std::env::current_dir().unwrap_or_else(|e| panic!("reading the working directory: {e}"));
    let engine = load_kokoro_engine::<CudaRuntime>(
        &dir,
        &ModelSearch::from_env(&cwd),
        "test",
        &CudaDevice::new(0),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let take = |seed| {
        engine
            .synthesize_seeded(KOKORO_PHRASE, KOKORO_VOICE, 1.0, Some(seed))
            .unwrap_or_else(|e| panic!("synthesizing on CUDA: {e}"))
    };
    let first = take(7);
    assert_audio(&first, "kokoro");
    assert_eq!(first, take(7), "one seed gave two takes on one device");
}
