//! Shared fixtures for blazr's `tests/*.rs` integration crates: model-fixture
//! resolution (`$SPECIFIC_VAR` else `$BOOSTR_MODELS_DIR/<relative>`, mirroring
//! voxr's `tests/common/mod.rs`), an in-process ASR-only `AppState` +
//! router, and a hand-rolled multipart body encoder so a test can drive
//! `/v1/audio/transcriptions` without a live server.
//!
//! Each file in `tests/` compiles as its own crate and pulls this module in
//! wholesale, so a helper a given test file does not call reads as dead code
//! there. The helpers ARE used — just not all of them by every crate.
#![allow(dead_code)]

#[cfg(feature = "audio")]
use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "audio")]
use std::sync::Arc;

#[cfg(feature = "server")]
use axum::body::Body;
#[cfg(feature = "server")]
use axum::http::{Request, StatusCode};
#[cfg(feature = "server")]
use axum::Router;
#[cfg(feature = "server")]
use tower::ServiceExt;

#[cfg(feature = "audio")]
use blazr::engine::Scheduler;
#[cfg(feature = "audio")]
use blazr::server::handlers::{AppState, ServerRuntime};

/// Whether the Hugging Face hub is in reach: `HF_HUB_OFFLINE` unset, and
/// huggingface.co answers on port 443 within 5 s. A test that downloads
/// skips with a notice otherwise.
pub fn hub_online() -> bool {
    use std::net::{TcpStream, ToSocketAddrs};
    if std::env::var_os("HF_HUB_OFFLINE").is_some_and(|v| !v.is_empty() && v != "0") {
        return false;
    }
    ("huggingface.co", 443)
        .to_socket_addrs()
        .ok()
        .and_then(|mut a| a.next())
        .is_some_and(|addr| {
            TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(5)).is_ok()
        })
}

/// Resolve a model fixture: `$<specific_var>`, else `$BOOSTR_MODELS_DIR/<relative>`.
/// Pass `relative = ""` for a fixture with no fallback under the models root,
/// such as a single GGUF file. Returns `None` when neither is set or the
/// resolved path is absent on disk, so callers skip.
pub fn model_fixture(specific_var: &str, relative: &str) -> Option<PathBuf> {
    let path = match std::env::var(specific_var) {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ if relative.is_empty() => return None,
        _ => {
            let root = std::env::var("BOOSTR_MODELS_DIR")
                .ok()
                .filter(|r| !r.is_empty())?;
            PathBuf::from(root).join(relative)
        }
    };
    path.exists().then_some(path)
}

/// Symlink `link` to `target`, a file or a directory. Windows needs
/// Developer Mode or an elevated shell to create a symlink.
#[cfg(feature = "audio")]
pub fn symlink(target: &Path, link: &Path) {
    #[cfg(unix)]
    let made = std::os::unix::fs::symlink(target, link);
    #[cfg(windows)]
    let made = if target.is_dir() {
        std::os::windows::fs::symlink_dir(target, link)
    } else {
        std::os::windows::fs::symlink_file(target, link)
    };
    made.unwrap_or_else(|e| panic!("linking {} to {}: {e}", link.display(), target.display()));
}

/// Print the standard skip notice naming both the specific env var and
/// `BOOSTR_MODELS_DIR`, so the reader knows how to enable the test.
pub fn skip_notice(what: &str, specific_var: &str) {
    eprintln!("skipping: {what} unavailable; set {specific_var} or BOOSTR_MODELS_DIR");
}

/// `whisper-tiny` checkpoint directory: `$WHISPER_TINY_DIR`, else
/// `$BOOSTR_MODELS_DIR/whisper-tiny`, else the HF hub cache layout
/// `huggingface_hub.snapshot_download` produces (`~/.cache/huggingface/hub/
/// models--openai--whisper-tiny/snapshots/<rev>`) — the same fallback
/// voxr's `whisper_word_oracle` test uses for `whisper-small` when no
/// local copy is staged under `$BOOSTR_MODELS_DIR`.
pub fn whisper_tiny_dir() -> Option<PathBuf> {
    if let Some(dir) = model_fixture("WHISPER_TINY_DIR", "whisper-tiny") {
        return Some(dir);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--openai--whisper-tiny/snapshots");
    let mut entries = std::fs::read_dir(&snapshots).ok()?;
    entries.find_map(|e| e.ok()).map(|e| e.path())
}

/// A short real speech clip for the ASR route tests:
/// `$BLAZR_ASR_TEST_WAV`, else `$BOOSTR_MODELS_DIR/librispeech_en_sample.wav`.
pub fn asr_test_wav() -> Option<PathBuf> {
    model_fixture("BLAZR_ASR_TEST_WAV", "librispeech_en_sample.wav")
}

/// An [`AppState`] with no text model (an unused scheduler model directory,
/// matching `server::handlers::auxiliary::tests::test_state`) and, when
/// `asr` is given, one real Whisper bundle loaded from its checkpoint
/// directory and registered under `asr.0`.
#[cfg(feature = "audio")]
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
#[cfg(feature = "audio")]
pub fn audio_router(state: Arc<AppState>) -> Router {
    blazr::server::api_routes().with_state(state)
}

/// One multipart/form-data body: `fields` in call order (repeat a name for a
/// repeated field, e.g. two `("timestamp_granularities[]", "word")` /
/// `("timestamp_granularities[]", "segment")` entries), then one `file` part
/// as `(field_name, filename, content_type, bytes)`.
pub fn multipart_body(
    boundary: &str,
    fields: &[(&str, &str)],
    file: (&str, &str, &str, &[u8]),
) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    let (field_name, filename, content_type, bytes) = file;
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(bytes);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

/// POST an already-encoded multipart body to `uri` on `router` and return the
/// status plus the response parsed as JSON. Panics if the body is not JSON —
/// callers that expect `text`/`srt`/`vtt` read the router response directly
/// instead of going through this helper.
#[cfg(feature = "server")]
pub async fn post_multipart_json(
    router: Router,
    uri: &str,
    boundary: &str,
    body: Vec<u8>,
) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("building multipart request");
    let response = router.oneshot(request).await.expect("router call failed");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("reading response body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "response body is not JSON: {e}; body = {:?}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, json)
}
