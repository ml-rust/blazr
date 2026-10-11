//! `POST /v1/audio/speech` handler — text-to-speech synthesis.
//!
//! One prologue validates the request, finds the bundle and resolves the
//! voice; then the body is either rendered whole and encoded
//! (`stream: false`) or streamed chunk by chunk as the engine produces it
//! (`stream: true`, see [`super::stream`]).

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::server::gen_types::error_response;
use crate::server::handlers::{AppState, TtsBundle};
use voxr::{SynthesizeOptions, TtsError};

use super::encode::encode_speech;
use super::request::{SpeechRequest, NATIVE_TTS_FORMATS, VALID_TTS_FORMATS};
use super::stream::{stream_speech, StreamFormat};
use super::voice::resolve_voice;

/// `POST /v1/audio/speech` — Text-to-speech synthesis.
pub async fn speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SpeechRequest>,
) -> Response {
    let (bundle, voice_id) = match prologue(&state, &request).await {
        Ok(ok) => ok,
        Err(resp) => return *resp,
    };

    if request.stream {
        let format = StreamFormat::from_response_format(&request.response_format);
        return stream_speech(bundle, request.input, voice_id, request.speed, format).await;
    }

    // Run synthesis off the reactor on a blocking worker.
    let input = request.input.clone();
    let speed = request.speed;
    let synth_bundle = Arc::clone(&bundle);
    let samples = tokio::task::spawn_blocking(move || {
        synth_bundle.synthesize(&input, &voice_id, &SynthesizeOptions { speed })
    })
    .await;

    let samples = match samples {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return tts_error_response(e),
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Synthesis task join error: {e}"),
                "server_error",
            );
        }
    };

    // Encode the produced waveform.
    let (bytes, content_type) =
        match encode_speech(&request.response_format, &samples, bundle.sample_rate) {
            Ok(encoded) => encoded,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("WAV encode failed: {e}"),
                    "server_error",
                );
            }
        };

    let mut resp = (StatusCode::OK, bytes).into_response();
    // `content_type` is one of the fixed literals `encode_speech` returns,
    // so a static header value cannot fail.
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static(content_type),
    );
    resp
}

/// Everything before synthesis, shared by the buffered and streamed paths:
/// input and format validation, the speed range, the bundle lookup and the
/// voice resolution. Returns the bundle and the resolved voice id, or the
/// error response to send back verbatim.
async fn prologue(
    state: &AppState,
    request: &SpeechRequest,
) -> Result<(Arc<TtsBundle>, String), Box<Response>> {
    if request.input.is_empty() {
        return Err(Box::new(error_response(
            StatusCode::BAD_REQUEST,
            "Input text must not be empty",
            "invalid_request_error",
        )));
    }

    if !VALID_TTS_FORMATS.contains(&request.response_format.as_str()) {
        return Err(Box::new(error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "Invalid response_format '{}'. Must be one of: {}",
                request.response_format,
                VALID_TTS_FORMATS.join(", ")
            ),
            "invalid_request_error",
        )));
    }

    if !NATIVE_TTS_FORMATS.contains(&request.response_format.as_str()) {
        return Err(Box::new(error_response(
            StatusCode::NOT_IMPLEMENTED,
            &format!(
                "response_format '{}' is not yet encoded natively. Use 'wav', 'pcm', or 'f32'.",
                request.response_format
            ),
            "not_implemented",
        )));
    }

    if !(0.25..=4.0).contains(&request.speed) {
        return Err(Box::new(error_response(
            StatusCode::BAD_REQUEST,
            "Speed must be between 0.25 and 4.0",
            "invalid_request_error",
        )));
    }

    // Look up the loaded TTS bundle.
    let bundle = match state.tts_model(&request.model).await {
        Some(b) => b,
        None => {
            return Err(Box::new(error_response(
                StatusCode::NOT_FOUND,
                &format!(
                    "TTS model '{}' is not loaded. Start the server with \
                     `--tts-model {}=PATH` to register a Kokoro checkpoint.",
                    request.model, request.model
                ),
                "invalid_request_error",
            )));
        }
    };

    let model_voices = state
        .tts_path(&request.model)
        .await
        .map(|dir| dir.join("voices"));
    let voice_id = resolve_voice(
        state,
        &bundle,
        request.voice.as_deref(),
        &request.model,
        model_voices.as_deref(),
    )?;
    Ok((bundle, voice_id))
}

/// The HTTP status and message for every [`TtsError`]. Shared by the
/// buffered path and the streamed path's pre-first-byte failure, so the two
/// answer the same request the same way.
pub(super) fn tts_error_response(err: TtsError) -> Response {
    match err {
        TtsError::NotImplemented => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "this TTS model has no synthesis engine: its directory carried no \
             loadable config.json, so only voice lookup ran",
            "not_implemented",
        ),
        TtsError::UnknownVoice(v) => error_response(
            StatusCode::BAD_REQUEST,
            &format!("Unknown voice: {v}"),
            "invalid_request_error",
        ),
        TtsError::Load(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("TTS bundle load error: {e}"),
            "server_error",
        ),
        TtsError::InvalidRequest(e) => {
            error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error")
        }
        TtsError::Engine(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("TTS synthesis failed: {e}"),
            "server_error",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_range_validation() {
        assert!((0.25..=4.0).contains(&0.25));
        assert!((0.25..=4.0).contains(&4.0));
        assert!((0.25..=4.0).contains(&1.0));
        assert!(!(0.25..=4.0).contains(&0.24));
        assert!(!(0.25..=4.0).contains(&4.1));
    }

    #[test]
    fn tts_error_status_mapping() {
        assert_eq!(
            tts_error_response(TtsError::NotImplemented).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            tts_error_response(TtsError::UnknownVoice("x".into())).status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            tts_error_response(TtsError::InvalidRequest("speed".into())).status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            tts_error_response(TtsError::Engine("boom".into())).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            tts_error_response(TtsError::Load("boom".into())).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
