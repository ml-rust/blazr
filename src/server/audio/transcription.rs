//! `/v1/audio/transcriptions` and `/v1/audio/translations` handlers, sharing
//! one pipeline: read the form, decode, read the audio through the
//! long-form path `blazr transcribe` runs, render the requested format.
//!
//! `/v1/audio/transcriptions` accepts multipart `timestamp_granularities[]`
//! (`word`, `segment`, or both), valid only with `response_format=verbose_json`
//! (400 otherwise). `word` adds a `words: [{word, start, end}]` array to the
//! response and is a 400 against a checkpoint with no `alignment_heads`.

use std::sync::Arc;

use axum::{
    extract::{Extension, Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::engine::WorkHold;
use crate::server::audio_decode::decode_to_whisper_input;
use crate::server::audio_transcribe::{
    transcribe_long, word_separator, LongTranscript, WhisperReader, WhisperRequest,
    WHISPER_SAMPLE_RATE,
};
use crate::server::body_limit::BodyLimit;
use crate::server::gen_types::error_response;
use crate::server::handlers::AppState;

use super::transcription_form::{read_form, TranscriptionForm};
use super::types::{
    render_srt, render_vtt, validate_timestamp_granularities, TranscriptionResponse,
    TranscriptionSegment, TranscriptionWord, VerboseTranscriptionResponse,
    DEFAULT_TRANSCRIPTION_FORMAT, VALID_TRANSCRIPTION_FORMATS,
};

/// Largest audio file, matching OpenAI.
const MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;

/// Shared pipeline for `/v1/audio/transcriptions` and `/v1/audio/translations`.
/// `force_translate=true` overrides the `task` field. `limit` is the server's
/// body limit, named when the form is over it.
async fn run_audio_pipeline(
    state: Arc<AppState>,
    limit: Option<BodyLimit>,
    mut multipart: Multipart,
    force_translate: bool,
) -> Response {
    let TranscriptionForm {
        file: file_bytes,
        file_ext,
        model,
        language,
        response_format,
        temperature,
        translate,
        timestamp_granularities,
    } = match read_form(&mut multipart, force_translate).await {
        Ok(form) => form,
        Err(e) => return e.into_response(limit),
    };

    if file_bytes
        .as_ref()
        .is_some_and(|b| b.len() > MAX_AUDIO_BYTES)
    {
        return error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            &format!(
                "Audio file exceeds the {MAX_AUDIO_BYTES}-byte (25 MB) limit. Send a shorter \
                 or more compressed file."
            ),
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

    // Read the audio through the long-form path off the reactor: the model
    // runs on a blocking worker thread.
    let sample_count = samples.len();
    let language_for_pipeline = language.clone();
    let device = state.scheduler.device().clone();
    // The blocking thread holds the request's slot until the model ends.
    let hold = WorkHold::current();
    let read: anyhow::Result<(LongTranscript, Option<&'static str>)> =
        match tokio::task::spawn_blocking(move || {
            let _hold = hold;
            let language = language_for_pipeline.as_deref();
            let request = WhisperRequest {
                language,
                translate,
                max_new_tokens: bundle.config.max_target_positions,
            };
            let mut reader = WhisperReader::new(bundle.as_ref(), &device, request);
            let long = transcribe_long(&samples, &mut reader, language)?;
            Ok((long, reader.language()))
        })
        .await
        {
            Ok(r) => r,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("Transcription task join error: {e}"),
                    "server_error",
                );
            }
        };
    let (long, decoded) = match read {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Transcription failed: {e}"),
                "server_error",
            );
        }
    };

    // The language the first piece decoded under: the request's own, or the
    // detected one. An english-only checkpoint has none. A translation is
    // English text.
    let decoded_language = decoded
        .map(str::to_string)
        .or(language)
        .unwrap_or_else(|| "en".to_string());
    let separator = if translate {
        " "
    } else {
        word_separator(Some(&decoded_language))
    };
    let full_text = long.text(separator);
    let total_duration = sample_count as f64 / WHISPER_SAMPLE_RATE as f64;
    let segments: Vec<TranscriptionSegment> = long
        .segments
        .iter()
        .enumerate()
        .map(|(i, s)| TranscriptionSegment {
            id: i,
            start: s.start,
            end: s.end,
            text: s.text(separator),
        })
        .collect();

    match fmt.as_str() {
        "text" => (StatusCode::OK, full_text).into_response(),
        "srt" => (StatusCode::OK, render_srt(&segments)).into_response(),
        "vtt" => (StatusCode::OK, render_vtt(&segments)).into_response(),
        "verbose_json" => {
            let words: Vec<TranscriptionWord> = long
                .words()
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
pub async fn transcriptions(
    State(state): State<Arc<AppState>>,
    limit: Option<Extension<BodyLimit>>,
    multipart: Multipart,
) -> Response {
    run_audio_pipeline(state, limit.map(|Extension(l)| l), multipart, false).await
}

/// `POST /v1/audio/translations` — Speech-to-English translation (Whisper).
pub async fn translations(
    State(state): State<Arc<AppState>>,
    limit: Option<Extension<BodyLimit>>,
    multipart: Multipart,
) -> Response {
    run_audio_pipeline(state, limit.map(|Extension(l)| l), multipart, true).await
}
