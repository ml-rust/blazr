//! `POST /v1/audio/speech` handler — text-to-speech synthesis.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use crate::server::gen_types::error_response;
use crate::server::handlers::AppState;
use boostr_audio::{SynthesizeOptions, TtsError};

use super::encode::encode_speech;
use super::request::{SpeechRequest, NATIVE_TTS_FORMATS, VALID_TTS_FORMATS};
use super::voice::resolve_voice;

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

    let voice_id = match resolve_voice(&state, &bundle, request.voice.as_deref(), &request.model) {
        Ok(id) => id,
        Err(resp) => return *resp,
    };

    // Run synthesis off the reactor — neural inference is CPU-bound.
    let input = request.input.clone();
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
                "this TTS model has no synthesis engine: its directory carried no \
                 loadable config.json, so only voice lookup and G2P ran",
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
        Ok(Err(TtsError::InvalidRequest(e))) => {
            return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
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

#[cfg(test)]
mod tests {
    #[test]
    fn speed_range_validation() {
        assert!((0.25..=4.0).contains(&0.25));
        assert!((0.25..=4.0).contains(&4.0));
        assert!((0.25..=4.0).contains(&1.0));
        assert!(!(0.25..=4.0).contains(&0.24));
        assert!(!(0.25..=4.0).contains(&4.1));
    }
}
