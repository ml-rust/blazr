//! `/v1/audio/speech` request body and its TTS response-format constants.

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    #[allow(dead_code)]
    pub model: String,
    pub input: String,
    /// `None` when the field is omitted — distinct from an explicit id.
    /// `speech()` resolves `None` per-bundle: zero-shot on VoxCPM2, `alloy`
    /// on Kokoro. `Some(id)` not in the bundle's catalog is a 400, always.
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    #[serde(default = "default_speed")]
    pub speed: f32,
}

pub(super) fn default_response_format() -> String {
    "wav".to_string()
}

pub(super) fn default_speed() -> f32 {
    1.0
}

pub(super) const VALID_TTS_FORMATS: &[&str] = &["wav", "pcm", "f32", "mp3", "opus", "aac", "flac"];
/// Formats we currently encode natively. Everything else returns 501 with a
/// message suggesting `wav` or `pcm`. MP3/Opus/AAC/FLAC land with the neural
/// synthesis work (checklist milestone 8).
pub(super) const NATIVE_TTS_FORMATS: &[&str] = &["wav", "pcm", "f32"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_request_defaults() {
        let json = r#"{"model": "tts-1", "input": "Hello world"}"#;
        let req: SpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "tts-1");
        assert_eq!(req.input, "Hello world");
        // Omitted `voice` is `None`, not a baked-in "alloy" — `speech()`
        // resolves it per-bundle at request time.
        assert_eq!(req.voice, None);
        assert_eq!(req.response_format, "wav");
        assert!((req.speed - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn speech_request_all_fields() {
        let json = r#"{
            "model": "tts-1-hd",
            "input": "Test",
            "voice": "nova",
            "response_format": "pcm",
            "speed": 1.5
        }"#;
        let req: SpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.voice, Some("nova".to_string()));
        assert_eq!(req.response_format, "pcm");
        assert!((req.speed - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn tts_native_formats_are_subset_of_valid() {
        for fmt in NATIVE_TTS_FORMATS {
            assert!(VALID_TTS_FORMATS.contains(fmt));
        }
    }

    #[test]
    fn tts_response_format_validation() {
        for fmt in VALID_TTS_FORMATS {
            assert!(VALID_TTS_FORMATS.contains(fmt));
        }
        assert!(!VALID_TTS_FORMATS.contains(&"invalid"));
    }
}
