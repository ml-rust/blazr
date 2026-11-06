//! Voice resolution for `/v1/audio/speech`: the omitted-vs-explicit rule,
//! and validating an explicit id against the bundle's catalog, the
//! `--voice-dir` directory, and the model directory's `voices/`.

use std::path::{Path, PathBuf};

use axum::{http::StatusCode, response::Response};
use voxr::voxcpm::ZERO_SHOT_VOICE_ID;

use crate::server::gen_types::error_response;
use crate::server::handlers::{AppState, TtsBundle};

/// Kokoro's long-standing default voice, used only when `voice` is omitted
/// and the bundle carries no `zero-shot` entry (i.e. not VoxCPM2).
pub(super) const DEFAULT_KOKORO_VOICE: &str = "alloy";

/// Resolve `requested` (the request's `voice` field) to a concrete voice id
/// and validate it. `model_voices` is the model directory's `voices/`.
/// Returns the resolved id, or the 400 [`Response`] to send back verbatim.
pub(super) fn resolve_voice(
    state: &AppState,
    bundle: &TtsBundle,
    requested: Option<&str>,
    model_name: &str,
    model_voices: Option<&Path>,
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

    // A request names a voice by id. The engine reads any path it gets, so a
    // path from a client never reaches it.
    if is_path_form(&voice_id) {
        return Err(Box::new(error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Voice '{voice_id}' is a path. Send a voice id, and place custom voice files \
                 under --voice-dir."
            ),
            "invalid_request_error",
        )));
    }

    if let Some(file) = state
        .voice_dir
        .as_deref()
        .filter(|_| bundle.voice(ZERO_SHOT_VOICE_ID).is_none())
        .and_then(|dir| voice_file_in(dir, &voice_id))
    {
        // A Kokoro voice file in `--voice-dir` wins over the model's own
        // `voices/`. The engine reads it by path: its own resolver knows
        // only the model directory and `$BLAZR_VOICE_DIR`.
        return Ok(file.to_string_lossy().into_owned());
    } else if bundle.voice(&voice_id).is_none() {
        // Bare ID outside the catalog: the engine reads it from the model's
        // `voices/` or `$BLAZR_VOICE_DIR`.
        let in_model = model_voices
            .and_then(|dir| voice_file_in(dir, &voice_id))
            .is_some();
        if !in_model
            && voxr::kokoro::VoiceResolver::new()
                .resolve_path(&voice_id)
                .is_err()
        {
            let available: Vec<String> = bundle.voices().iter().map(|v| v.id.clone()).collect();
            return Err(Box::new(error_response(
                StatusCode::BAD_REQUEST,
                &format!(
                    "Invalid voice '{}'. Available voices for model '{}': {}. \
                     Place custom voice files under --voice-dir.",
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

/// True for what `voxr::kokoro::VoiceResolver` reads as a file path.
fn is_path_form(id: &str) -> bool {
    id.contains('/')
        || id.contains('\\')
        || id.ends_with(".safetensors")
        || id.ends_with(".pt")
        || id.ends_with(".pth")
}

/// The voice file `id` names in `dir`: `{id}.safetensors`, `.pt` or
/// `.pth`, in that order.
fn voice_file_in(dir: &Path, id: &str) -> Option<PathBuf> {
    ["safetensors", "pt", "pth"]
        .iter()
        .map(|ext| dir.join(format!("{id}.{ext}")))
        .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_voice_path_is_told_from_a_voice_id() {
        for path in [
            "/etc/passwd",
            "../voices/a",
            "a\\b",
            "custom.pt",
            "custom.pth",
            "custom.safetensors",
        ] {
            assert!(is_path_form(path), "{path}");
        }
        for id in ["af_heart", "alloy", "zero-shot"] {
            assert!(!is_path_form(id), "{id}");
        }
    }

    #[test]
    fn voice_file_in_prefers_safetensors_and_skips_missing_ids() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("am_test.pt"), b"").expect("write");
        std::fs::write(dir.path().join("am_test.safetensors"), b"").expect("write");
        assert_eq!(
            voice_file_in(dir.path(), "am_test"),
            Some(dir.path().join("am_test.safetensors"))
        );
        assert_eq!(voice_file_in(dir.path(), "am_none"), None);
    }
}
