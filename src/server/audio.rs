//! Audio endpoints: TTS (speech) and ASR (transcriptions)
//!
//! OpenAI-compatible `/v1/audio/speech` and `/v1/audio/transcriptions` endpoints.
//! These are stub implementations that return 501 until boostr adds full TTS/ASR
//! model support.

use std::sync::Arc;

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::Response,
    Json,
};
use serde::{Deserialize, Serialize};

use super::gen_types::error_response;
use super::handlers::AppState;

// ─── TTS types ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    #[allow(dead_code)]
    pub model: String,
    pub input: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    #[serde(default = "default_speed")]
    pub speed: f32,
}

fn default_voice() -> String {
    "alloy".to_string()
}

fn default_response_format() -> String {
    "wav".to_string()
}

fn default_speed() -> f32 {
    1.0
}

const VALID_VOICES: &[&str] = &["alloy", "echo", "fable", "onyx", "nova", "shimmer"];
const VALID_TTS_FORMATS: &[&str] = &["wav", "pcm", "mp3", "opus", "aac", "flac"];

// ─── ASR types ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct TranscriptionResponse {
    pub text: String,
}

#[derive(Debug, Serialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct VerboseTranscriptionResponse {
    pub text: String,
    pub language: String,
    pub duration: f64,
    pub segments: Vec<TranscriptionSegment>,
}

#[derive(Debug, Serialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub struct TranscriptionSegment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

const VALID_TRANSCRIPTION_FORMATS: &[&str] = &["json", "text", "verbose_json"];
const DEFAULT_TRANSCRIPTION_FORMAT: &str = "json";

// ─── Handlers ────────────────────────────────────────────────────────────────

/// `POST /v1/audio/speech` — Text-to-speech synthesis
pub async fn speech(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<SpeechRequest>,
) -> Response {
    // Validate input
    if request.input.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Input text must not be empty",
            "invalid_request_error",
        );
    }

    if !VALID_VOICES.contains(&request.voice.as_str()) {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Invalid voice '{}'. Must be one of: {}",
                request.voice,
                VALID_VOICES.join(", ")
            ),
            "invalid_request_error",
        );
    }

    if !VALID_TTS_FORMATS.contains(&request.response_format.as_str()) {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Invalid response_format '{}'. Must be one of: {}",
                request.response_format,
                VALID_TTS_FORMATS.join(", ")
            ),
            "invalid_request_error",
        );
    }

    if !(0.25..=4.0).contains(&request.speed) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Speed must be between 0.25 and 4.0",
            "invalid_request_error",
        );
    }

    // TTS model pipeline not yet available in boostr
    error_response(
        StatusCode::NOT_IMPLEMENTED,
        "TTS model not loaded. This endpoint requires a model with text-to-speech \
         capability (e.g., Qwen3-TTS).",
        "not_implemented",
    )
}

/// `POST /v1/audio/transcriptions` — Speech-to-text transcription
pub async fn transcriptions(
    State(_state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format: Option<String> = None;
    let mut temperature: Option<f32> = None;

    // Extract multipart fields
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = match field.name() {
            Some(n) => n.to_string(),
            None => continue,
        };

        match name.as_str() {
            "file" => {
                file_bytes = field.bytes().await.ok().map(|b| b.to_vec());
            }
            "model" => {
                model = field.text().await.ok();
            }
            "language" => {
                language = field.text().await.ok().filter(|s| !s.is_empty());
            }
            "response_format" => {
                response_format = field.text().await.ok().filter(|s| !s.is_empty());
            }
            "temperature" => {
                temperature = field.text().await.ok().and_then(|s| s.parse::<f32>().ok());
            }
            _ => {}
        }
    }

    // Validate file size (25MB limit, matching OpenAI)
    const MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;
    if file_bytes
        .as_ref()
        .map_or(false, |b| b.len() > MAX_AUDIO_BYTES)
    {
        return error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Audio file exceeds 25MB limit",
            "invalid_request_error",
        );
    }

    // Validate required fields
    let _model = match model {
        Some(m) if !m.is_empty() => m,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Field 'model' is required",
                "invalid_request_error",
            );
        }
    };

    let _file = match file_bytes {
        Some(ref b) if !b.is_empty() => b,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Field 'file' is required and must not be empty",
                "invalid_request_error",
            );
        }
    };

    let fmt = response_format
        .as_deref()
        .unwrap_or(DEFAULT_TRANSCRIPTION_FORMAT);
    if !VALID_TRANSCRIPTION_FORMATS.contains(&fmt) {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Invalid response_format '{}'. Must be one of: {}",
                fmt,
                VALID_TRANSCRIPTION_FORMATS.join(", ")
            ),
            "invalid_request_error",
        );
    }

    if let Some(t) = temperature {
        if !(0.0..=1.0).contains(&t) {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Temperature must be between 0.0 and 1.0",
                "invalid_request_error",
            );
        }
    }

    // Suppress unused variable warnings for future use
    let _ = (language, temperature, file_bytes);

    // ASR model pipeline not yet available in boostr
    error_response(
        StatusCode::NOT_IMPLEMENTED,
        "ASR model not loaded. This endpoint requires a speech-to-text model (e.g., Whisper).",
        "not_implemented",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_request_defaults() {
        let json = r#"{"model": "tts-1", "input": "Hello world"}"#;
        let req: SpeechRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "tts-1");
        assert_eq!(req.input, "Hello world");
        assert_eq!(req.voice, "alloy");
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
        assert_eq!(req.voice, "nova");
        assert_eq!(req.response_format, "pcm");
        assert!((req.speed - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn transcription_response_serialization() {
        let resp = TranscriptionResponse {
            text: "Hello world".to_string(),
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["text"], "Hello world");
    }

    #[test]
    fn verbose_transcription_response_serialization() {
        let resp = VerboseTranscriptionResponse {
            text: "Hello".to_string(),
            language: "en".to_string(),
            duration: 1.5,
            segments: vec![TranscriptionSegment {
                id: 0,
                start: 0.0,
                end: 1.5,
                text: "Hello".to_string(),
            }],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["language"], "en");
        assert_eq!(json["duration"], 1.5);
        assert_eq!(json["segments"][0]["id"], 0);
        assert_eq!(json["segments"][0]["start"], 0.0);
        assert_eq!(json["segments"][0]["end"], 1.5);
        assert_eq!(json["segments"][0]["text"], "Hello");
    }

    #[test]
    fn voice_validation() {
        for voice in VALID_VOICES {
            assert!(VALID_VOICES.contains(voice));
        }
        assert!(!VALID_VOICES.contains(&"invalid_voice"));
    }

    #[test]
    fn speed_range_validation() {
        assert!((0.25..=4.0).contains(&0.25));
        assert!((0.25..=4.0).contains(&4.0));
        assert!((0.25..=4.0).contains(&1.0));
        assert!(!(0.25..=4.0).contains(&0.24));
        assert!(!(0.25..=4.0).contains(&4.1));
    }

    #[test]
    fn response_format_validation() {
        for fmt in VALID_TTS_FORMATS {
            assert!(VALID_TTS_FORMATS.contains(fmt));
        }
        assert!(!VALID_TTS_FORMATS.contains(&"invalid"));

        for fmt in VALID_TRANSCRIPTION_FORMATS {
            assert!(VALID_TRANSCRIPTION_FORMATS.contains(fmt));
        }
        assert!(!VALID_TRANSCRIPTION_FORMATS.contains(&"invalid"));
    }
}
