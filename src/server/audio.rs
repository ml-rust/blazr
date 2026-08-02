//! Audio endpoints: TTS (speech) and ASR (transcriptions)
//!
//! OpenAI-compatible `/v1/audio/speech` and `/v1/audio/transcriptions` endpoints.
//! These are stub implementations that return 501 until boostr adds full TTS/ASR
//! model support.

use std::sync::Arc;

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use super::audio_decode::{decode_to_whisper_input, extension_hint};
use super::gen_types::error_response;
use super::handlers::AppState;
use boostr::model::audio::{
    encode_pcm16_raw, encode_wav_f32, encode_wav_pcm16, SynthesizeOptions, TtsError,
};

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

const VALID_TTS_FORMATS: &[&str] = &["wav", "pcm", "f32", "mp3", "opus", "aac", "flac"];
/// Formats we currently encode natively. Everything else returns 501 with a
/// message suggesting `wav` or `pcm`. MP3/Opus/AAC/FLAC land with the neural
/// synthesis work (checklist milestone 8).
const NATIVE_TTS_FORMATS: &[&str] = &["wav", "pcm", "f32"];

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

const VALID_TRANSCRIPTION_FORMATS: &[&str] = &["json", "text", "verbose_json", "vtt", "srt"];
const DEFAULT_TRANSCRIPTION_FORMAT: &str = "json";

/// Format a duration in seconds as an SRT timestamp `HH:MM:SS,mmm`.
fn format_srt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms / 60_000) % 60;
    let s = (total_ms / 1000) % 60;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Format a duration in seconds as a WebVTT timestamp `HH:MM:SS.mmm`.
fn format_vtt_time(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms / 60_000) % 60;
    let s = (total_ms / 1000) % 60;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

fn render_srt(segments: &[TranscriptionSegment]) -> String {
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            format_srt_time(seg.start),
            format_srt_time(seg.end)
        ));
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}

fn render_vtt(segments: &[TranscriptionSegment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in segments {
        out.push_str(&format!(
            "{} --> {}\n",
            format_vtt_time(seg.start),
            format_vtt_time(seg.end)
        ));
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}

// ─── Handlers ────────────────────────────────────────────────────────────────

/// `POST /v1/audio/speech` — Text-to-speech synthesis (Kokoro).
pub async fn speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SpeechRequest>,
) -> Response {
    // Basic input validation.
    if request.input.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Input text must not be empty",
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

    if !NATIVE_TTS_FORMATS.contains(&request.response_format.as_str()) {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            &format!(
                "response_format '{}' is not yet encoded natively. Use 'wav', 'pcm', or 'f32'.",
                request.response_format
            ),
            "not_implemented",
        );
    }

    if !(0.25..=4.0).contains(&request.speed) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Speed must be between 0.25 and 4.0",
            "invalid_request_error",
        );
    }

    // Look up the loaded TTS bundle.
    let bundle = match state.tts_model(&request.model).await {
        Some(b) => b,
        None => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!(
                    "TTS model '{}' is not loaded. Start the server with \
                     `--tts-model {}=PATH` to register a Kokoro checkpoint.",
                    request.model, request.model
                ),
                "invalid_request_error",
            );
        }
    };

    // Resolve the `voice` field — accepts either a bundled catalog ID
    // (`af_alloy`) or a direct path to a custom voice file (`.safetensors`,
    // `.pt`, `.pth`). Path form is detected by the presence of a separator or
    // a known extension; bare IDs fall through to the bundle's catalog.
    let voice_is_path = request.voice.contains('/')
        || request.voice.contains('\\')
        || request.voice.ends_with(".safetensors")
        || request.voice.ends_with(".pt")
        || request.voice.ends_with(".pth");

    if voice_is_path {
        // External voice file: validate that `VoiceResolver` can find it. The
        // actual loading happens once the neural synth path is wired; for now
        // we surface invalid paths as 400s at the HTTP layer so users get
        // fast feedback rather than waiting for a cryptic synth error.
        let mut resolver = boostr::model::audio::kokoro::VoiceResolver::new();
        if let Some(dir) = state.voice_dir.clone() {
            resolver = resolver.with_asset_dir(dir);
        }
        if let Err(err) = resolver.resolve_path(&request.voice) {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Voice path could not be resolved: {err}"),
                "invalid_request_error",
            );
        }
    } else if bundle.voice(&request.voice).is_none() {
        // Bare ID — try the server's `--voice-dir` / `$BLAZR_VOICE_DIR` /
        // bundled lookup before giving up. This lets users drop a new voice
        // into the voice directory and reference it by ID without restarting
        // TTS model registration.
        let mut resolver = boostr::model::audio::kokoro::VoiceResolver::new();
        if let Some(dir) = state.voice_dir.clone() {
            resolver = resolver.with_asset_dir(dir);
        }
        if resolver.resolve_path(&request.voice).is_err() {
            let available: Vec<String> = bundle.voices().iter().map(|v| v.id.clone()).collect();
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!(
                    "Invalid voice '{}'. Available voices for model '{}': {}. \
                     Custom voice files can also be referenced by absolute path \
                     or placed under --voice-dir.",
                    request.voice,
                    request.model,
                    available.join(", ")
                ),
                "invalid_request_error",
            );
        }
    }

    // Run synthesis off the reactor — neural inference is CPU-bound.
    let input = request.input.clone();
    let voice_id = request.voice.clone();
    let speed = request.speed;
    let synth_bundle = Arc::clone(&bundle);
    let samples = tokio::task::spawn_blocking(move || {
        synth_bundle.synthesize(&input, &voice_id, &SynthesizeOptions { speed })
    })
    .await;

    let samples = match samples {
        Ok(Ok(s)) => s,
        Ok(Err(TtsError::NotImplemented)) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "TTS neural synthesis path is under construction — the server has \
                 validated input, G2P, and voice lookup. See kokoro_tts_checklist.md.",
                "not_implemented",
            );
        }
        Ok(Err(TtsError::UnknownVoice(v))) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Unknown voice: {v}"),
                "invalid_request_error",
            );
        }
        Ok(Err(TtsError::G2p(e))) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("G2P failed: {e}"),
                "invalid_request_error",
            );
        }
        Ok(Err(TtsError::Load(e))) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("TTS bundle load error: {e}"),
                "server_error",
            );
        }
        Ok(Err(TtsError::Engine(e))) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("TTS synthesis failed: {e}"),
                "server_error",
            );
        }
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Synthesis task join error: {e}"),
                "server_error",
            );
        }
    };

    // Encode the produced waveform.
    let (bytes, content_type) = match request.response_format.as_str() {
        "pcm" => (encode_pcm16_raw(&samples), "audio/L16"),
        "f32" => (encode_wav_f32(&samples, bundle.sample_rate), "audio/wav"),
        _ => (encode_wav_pcm16(&samples, bundle.sample_rate), "audio/wav"),
    };

    let mut resp = (StatusCode::OK, bytes).into_response();
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        content_type.parse().unwrap(),
    );
    resp
}

/// Shared pipeline for `/v1/audio/transcriptions` and `/v1/audio/translations`.
/// `force_translate=true` overrides the `task` field.
async fn run_audio_pipeline(
    state: Arc<AppState>,
    mut multipart: Multipart,
    force_translate: bool,
) -> Response {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut file_ext: Option<String> = None;
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format: Option<String> = None;
    let mut temperature: Option<f32> = None;
    let mut translate = force_translate;

    // Extract multipart fields
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = match field.name() {
            Some(n) => n.to_string(),
            None => continue,
        };

        match name.as_str() {
            "file" => {
                if let Some(fname) = field.file_name() {
                    file_ext = extension_hint(fname).map(|s| s.to_string());
                }
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
            // Some clients pass `task=translate` on the transcriptions
            // endpoint — honor it when the route itself isn't forcing it.
            "task" if !force_translate => {
                translate = field
                    .text()
                    .await
                    .ok()
                    .is_some_and(|s| s.eq_ignore_ascii_case("translate"));
            }
            _ => {}
        }
    }

    // Validate file size (25MB limit, matching OpenAI)
    const MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;
    if file_bytes
        .as_ref()
        .is_some_and(|b| b.len() > MAX_AUDIO_BYTES)
    {
        return error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Audio file exceeds 25MB limit",
            "invalid_request_error",
        );
    }

    // Validate required fields
    let model_name = match model {
        Some(m) if !m.is_empty() => m,
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Field 'model' is required",
                "invalid_request_error",
            );
        }
    };

    let file = match file_bytes {
        Some(b) if !b.is_empty() => b,
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
        .unwrap_or(DEFAULT_TRANSCRIPTION_FORMAT)
        .to_string();
    if !VALID_TRANSCRIPTION_FORMATS.contains(&fmt.as_str()) {
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
    let _ = temperature; // greedy decode ignores temperature for now

    // Look up the loaded Whisper bundle.
    let bundle = match state.asr_model(&model_name).await {
        Some(b) => b,
        None => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!(
                    "ASR model '{}' is not loaded. Start the server with \
                     `--asr-model {}=PATH` to register a Whisper checkpoint.",
                    model_name, model_name
                ),
                "invalid_request_error",
            );
        }
    };

    // Decode audio off the executor thread — symphonia + rubato are CPU-bound.
    let ext_for_decode = file_ext.clone();
    let samples = match tokio::task::spawn_blocking(move || {
        decode_to_whisper_input(&file, ext_for_decode.as_deref())
    })
    .await
    {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                &format!("Audio decode failed: {e}"),
                "invalid_request_error",
            );
        }
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Decode task join error: {e}"),
                "server_error",
            );
        }
    };

    // Run the Whisper pipeline per 30-second chunk, concatenating segments.
    use super::audio_transcribe::{transcribe_chunks, ChunkTranscript};
    let language_for_pipeline = language.clone();
    let chunks: Vec<ChunkTranscript> = match transcribe_chunks(
        bundle.as_ref(),
        &samples,
        language_for_pipeline.as_deref(),
        translate,
        bundle.config.max_target_positions,
    ) {
        Ok(c) => c,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Transcription failed: {e}"),
                "server_error",
            );
        }
    };

    let full_text: String = chunks
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    let total_duration = samples.len() as f64 / 16_000.0;
    let segments: Vec<TranscriptionSegment> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| TranscriptionSegment {
            id: i,
            start: c.start,
            end: c.end,
            text: c.text.clone(),
        })
        .collect();

    match fmt.as_str() {
        "text" => (StatusCode::OK, full_text).into_response(),
        "srt" => (StatusCode::OK, render_srt(&segments)).into_response(),
        "vtt" => (StatusCode::OK, render_vtt(&segments)).into_response(),
        "verbose_json" => {
            let resp = VerboseTranscriptionResponse {
                text: full_text,
                language: language.unwrap_or_else(|| "en".to_string()),
                duration: total_duration,
                segments,
            };
            (StatusCode::OK, Json(resp)).into_response()
        }
        _ => {
            // "json" (default) and any other validated format falls here.
            let resp = TranscriptionResponse { text: full_text };
            (StatusCode::OK, Json(resp)).into_response()
        }
    }
}

/// `POST /v1/audio/transcriptions` — Speech-to-text transcription (Whisper).
pub async fn transcriptions(State(state): State<Arc<AppState>>, multipart: Multipart) -> Response {
    run_audio_pipeline(state, multipart, false).await
}

/// `POST /v1/audio/translations` — Speech-to-English translation (Whisper).
pub async fn translations(State(state): State<Arc<AppState>>, multipart: Multipart) -> Response {
    run_audio_pipeline(state, multipart, true).await
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
    fn tts_native_formats_are_subset_of_valid() {
        for fmt in NATIVE_TTS_FORMATS {
            assert!(VALID_TTS_FORMATS.contains(fmt));
        }
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

    #[test]
    fn srt_time_format() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(1.5), "00:00:01,500");
        assert_eq!(format_srt_time(61.123), "00:01:01,123");
        assert_eq!(format_srt_time(3661.001), "01:01:01,001");
    }

    #[test]
    fn vtt_time_format() {
        assert_eq!(format_vtt_time(0.0), "00:00:00.000");
        assert_eq!(format_vtt_time(1.5), "00:00:01.500");
        assert_eq!(format_vtt_time(3661.001), "01:01:01.001");
    }

    #[test]
    fn srt_render_segments() {
        let segs = vec![
            TranscriptionSegment {
                id: 0,
                start: 0.0,
                end: 1.5,
                text: "Hello".into(),
            },
            TranscriptionSegment {
                id: 1,
                start: 1.5,
                end: 3.2,
                text: "world".into(),
            },
        ];
        let out = render_srt(&segs);
        assert!(out.contains("1\n00:00:00,000 --> 00:00:01,500\nHello\n\n"));
        assert!(out.contains("2\n00:00:01,500 --> 00:00:03,200\nworld\n\n"));
    }

    #[test]
    fn vtt_render_segments() {
        let segs = vec![TranscriptionSegment {
            id: 0,
            start: 0.0,
            end: 2.5,
            text: "Hi".into(),
        }];
        let out = render_vtt(&segs);
        assert!(out.starts_with("WEBVTT\n\n"));
        assert!(out.contains("00:00:00.000 --> 00:00:02.500\nHi"));
    }

    #[test]
    fn validates_new_formats() {
        assert!(VALID_TRANSCRIPTION_FORMATS.contains(&"srt"));
        assert!(VALID_TRANSCRIPTION_FORMATS.contains(&"vtt"));
    }
}
