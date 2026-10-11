//! `/v1/audio/transcriptions` and `/v1/audio/translations` handlers, sharing
//! one pipeline: decode, run Whisper per 30s chunk, render the requested
//! format.

use std::sync::Arc;

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::server::audio_decode::{decode_to_whisper_input, extension_hint};
use crate::server::gen_types::error_response;
use crate::server::handlers::AppState;

use super::types::{
    render_srt, render_vtt, validate_timestamp_granularities, TranscriptionResponse,
    TranscriptionSegment, TranscriptionWord, VerboseTranscriptionResponse,
    DEFAULT_TRANSCRIPTION_FORMAT, VALID_TRANSCRIPTION_FORMATS,
};

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
    // `timestamp_granularities[]` arrives as one multipart field per value.
    let mut timestamp_granularities: Vec<String> = Vec::new();

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
            "timestamp_granularities[]" if !force_translate => {
                if let Some(v) = field.text().await.ok().filter(|s| !s.is_empty()) {
                    timestamp_granularities.push(v);
                }
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

    let granularities = match validate_timestamp_granularities(&timestamp_granularities, &fmt) {
        Ok(g) => g,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, &msg, "invalid_request_error"),
    };

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

    if granularities.want_words && bundle.generation.alignment_heads.is_none() {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "ASR model '{}' ships no `alignment_heads` in its generation_config.json; word \
                 timestamps are unavailable for this checkpoint",
                model_name
            ),
            "invalid_request_error",
        );
    }

    // Decode audio off the executor thread — the boostr decode/resample path is CPU-bound.
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

    // Run the Whisper pipeline per 30-second chunk, concatenating segments,
    // off the reactor: the model runs on a blocking worker thread.
    use crate::server::audio_transcribe::{transcribe_chunks, ChunkTranscript};
    let sample_count = samples.len();
    let language_for_pipeline = language.clone();
    let device = state.scheduler.device().clone();
    let want_words = granularities.want_words;
    let chunks: Vec<ChunkTranscript> = match tokio::task::spawn_blocking(move || {
        transcribe_chunks(
            bundle.as_ref(),
            &device,
            &samples,
            language_for_pipeline.as_deref(),
            translate,
            bundle.config.max_target_positions,
            want_words,
        )
    })
    .await
    {
        Ok(Ok(c)) => c,
        Ok(Err(e)) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Transcription failed: {e}"),
                "server_error",
            );
        }
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Transcription task join error: {e}"),
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
    let total_duration =
        sample_count as f64 / crate::server::audio_transcribe::WHISPER_SAMPLE_RATE as f64;
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
            // The language the first window was decoded under: the request's
            // own, or the detected one. An english-only checkpoint has none.
            let decoded_language = chunks
                .first()
                .and_then(|c| c.language)
                .map(str::to_string)
                .or(language)
                .unwrap_or_else(|| "en".to_string());
            let words: Vec<TranscriptionWord> = chunks
                .iter()
                .flat_map(|c| c.words.iter())
                .map(|w| TranscriptionWord {
                    word: w.text.clone(),
                    start: w.start,
                    end: w.end,
                })
                .collect();
            let resp = VerboseTranscriptionResponse {
                text: full_text,
                language: decoded_language,
                duration: total_duration,
                segments: granularities.want_segments.then_some(segments),
                words: granularities.want_words.then_some(words),
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
