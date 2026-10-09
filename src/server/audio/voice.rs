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
        // External voice file: a missing one is a 400 here, before the
        // engine loads it.
        if let Err(err) = voxr::kokoro::VoiceResolver::new().resolve_path(&voice_id) {
            return Err(Box::new(error_response(
                StatusCode::BAD_REQUEST,
                &format!("Voice path rejected: {err}"),
                "invalid_request_error",
            )));
        }
    } else if let Some(file) = state
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
