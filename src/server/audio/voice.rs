//! Voice resolution for `/v1/audio/speech`: the omitted-vs-explicit rule,
//! and validating an explicit id against the bundle's catalog or the
//! on-disk `--voice-dir` resolver.

use axum::{http::StatusCode, response::Response};
use boostr_audio::voxcpm::ZERO_SHOT_VOICE_ID;

use crate::server::gen_types::error_response;
use crate::server::handlers::{AppState, TtsBundle};

/// Kokoro's long-standing default voice, used only when `voice` is omitted
/// and the bundle carries no `zero-shot` entry (i.e. not VoxCPM2).
pub(super) const DEFAULT_KOKORO_VOICE: &str = "alloy";

/// Resolve `requested` (the request's `voice` field) to a concrete voice id
/// and validate it. Returns the resolved id, or the 400 [`Response`] to send
/// back verbatim.
pub(super) fn resolve_voice(
    state: &AppState,
    bundle: &TtsBundle,
    requested: Option<&str>,
    model_name: &str,
) -> Result<String, Box<Response>> {
    // Resolve an omitted `voice`: zero-shot when the bundle carries it
    // (VoxCPM2), else the long-standing Kokoro default. An explicit id is
    // used exactly as given — never silently redirected — so a mismatch
    // (including a client's habitual `"alloy"` against a VoxCPM2 bundle)
    // surfaces as the 400 below rather than being papered over.
    let voice_id = requested.map(str::to_string).unwrap_or_else(|| {
        if bundle.voice(ZERO_SHOT_VOICE_ID).is_some() {
            ZERO_SHOT_VOICE_ID.to_string()
        } else {
            DEFAULT_KOKORO_VOICE.to_string()
        }
    });

    // Accepts either a bundled catalog ID (`af_alloy`) or a direct path to a
    // custom voice file (`.safetensors`, `.pt`, `.pth`). Path form is
    // detected by the presence of a separator or a known extension; bare IDs
    // fall through to the bundle's catalog.
    let voice_is_path = voice_id.contains('/')
        || voice_id.contains('\\')
        || voice_id.ends_with(".safetensors")
        || voice_id.ends_with(".pt")
        || voice_id.ends_with(".pth");

    if voice_is_path {
        // External voice file: validate that `VoiceResolver` can find it. The
        // actual loading happens once the neural synth path is wired; for now
        // we surface invalid paths as 400s at the HTTP layer so users get
        // fast feedback rather than waiting for a cryptic synth error.
        let mut resolver = boostr_audio::kokoro::VoiceResolver::new();
        if let Some(dir) = state.voice_dir.clone() {
            resolver = resolver.with_asset_dir(dir);
        }
        if let Err(err) = resolver.resolve_path(&voice_id) {
            return Err(Box::new(error_response(
                StatusCode::BAD_REQUEST,
                &format!("Voice path could not be resolved: {err}"),
                "invalid_request_error",
            )));
        }
    } else if bundle.voice(&voice_id).is_none() {
        // Bare ID — try the server's `--voice-dir` / `$BLAZR_VOICE_DIR` /
        // bundled lookup before giving up. This lets users drop a new voice
        // into the voice directory and reference it by ID without restarting
        // TTS model registration.
        let mut resolver = boostr_audio::kokoro::VoiceResolver::new();
        if let Some(dir) = state.voice_dir.clone() {
            resolver = resolver.with_asset_dir(dir);
        }
        if resolver.resolve_path(&voice_id).is_err() {
            let available: Vec<String> = bundle.voices().iter().map(|v| v.id.clone()).collect();
            return Err(Box::new(error_response(
                StatusCode::BAD_REQUEST,
                &format!(
                    "Invalid voice '{}'. Available voices for model '{}': {}. \
                     Custom voice files can also be referenced by absolute path \
                     or placed under --voice-dir.",
                    voice_id,
                    model_name,
                    available.join(", ")
                ),
                "invalid_request_error",
            )));
        }
    }

    Ok(voice_id)
}
