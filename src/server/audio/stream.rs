//! Streaming body for `/v1/audio/speech` with `"stream": true`.
//!
//! The engine renders on a blocking worker and hands each chunk of samples
//! to a channel; the response body drains it. The handler waits for the
//! FIRST chunk before committing a status, so every error the engine raises
//! before producing audio (unknown voice, a `speed` it has no control for,
//! no engine) maps to the same HTTP status as the buffered path. Once the
//! header is out, an engine error can only end the body early: the stream
//! yields an `io::Error`, which aborts the connection — the one signal a
//! client can still observe.
//!
//! A client that goes away drops the receiver; the next `blocking_send`
//! fails, the sink returns an error, and the engine stops generating.

use std::io;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::stream;
use tokio::sync::mpsc;

use boostr_audio::{
    encode_f32_raw, encode_pcm16_raw, wav_stream_header_f32, wav_stream_header_pcm16,
    SynthesizeOptions, TtsBundle, TtsError,
};

use crate::server::gen_types::error_response;

/// The native formats a stream can carry, with their per-chunk encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StreamFormat {
    /// 16-bit PCM inside a WAV container.
    Wav,
    /// 32-bit float inside a WAV container.
    F32,
    /// Headerless 16-bit PCM.
    Pcm,
}

impl StreamFormat {
    /// Caller has already validated `response_format` against
    /// [`super::request::NATIVE_TTS_FORMATS`].
    pub(super) fn from_response_format(response_format: &str) -> Self {
        match response_format {
            "pcm" => Self::Pcm,
            "f32" => Self::F32,
            _ => Self::Wav,
        }
    }

    pub(super) fn content_type(self) -> &'static str {
        match self {
            Self::Wav | Self::F32 => "audio/wav",
            Self::Pcm => "audio/L16",
        }
    }

    /// The bytes before the first chunk: an unknown-length WAV header, or
    /// nothing for raw PCM.
    pub(super) fn header(self, sample_rate: u32) -> Result<Vec<u8>, boostr_audio::Error> {
        match self {
            Self::Wav => wav_stream_header_pcm16(sample_rate),
            Self::F32 => wav_stream_header_f32(sample_rate),
            Self::Pcm => Ok(Vec::new()),
        }
    }

    pub(super) fn encode(self, samples: &[f32]) -> Vec<u8> {
        match self {
            Self::Wav | Self::Pcm => encode_pcm16_raw(samples),
            Self::F32 => encode_f32_raw(samples),
        }
    }
}

/// Chunks the producer hands the body, in order.
type Chunk = Result<Bytes, TtsError>;

/// Render `input` in `voice_id` on a blocking worker and stream the encoded
/// chunks. Resolves once the engine has produced its first chunk or failed
/// before doing so; the status and headers of the returned response reflect
/// that first outcome.
pub(super) async fn stream_speech(
    bundle: Arc<TtsBundle>,
    input: String,
    voice_id: String,
    speed: f32,
    format: StreamFormat,
) -> Response {
    let header_bytes = match format.header(bundle.sample_rate) {
        Ok(h) => h,
        Err(e) => {
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("WAV header encode failed: {e}"),
                "server_error",
            );
        }
    };

    let (tx, mut rx) = mpsc::channel::<Chunk>(8);
    tokio::task::spawn_blocking(move || {
        let sender = tx.clone();
        let mut sink = move |samples: &[f32]| -> boostr_audio::Result<()> {
            sender
                .blocking_send(Ok(Bytes::from(format.encode(samples))))
                .map_err(|_| boostr_audio::Error::ModelError {
                    reason: "client disconnected before the stream finished".into(),
                })
        };
        if let Err(e) =
            bundle.synthesize_stream(&input, &voice_id, &SynthesizeOptions { speed }, &mut sink)
        {
            // A closed channel means the client is gone; nothing to report to.
            let _ = tx.blocking_send(Err(e));
        }
    });

    // The first outcome decides the status: an error here has produced no
    // audio, so it maps exactly as the buffered path maps it.
    let first = match rx.recv().await {
        Some(Ok(chunk)) => Some(chunk),
        Some(Err(e)) => return super::speech::tts_error_response(e),
        None => None,
    };

    let mut prelude = header_bytes;
    if let Some(chunk) = first {
        prelude.extend_from_slice(&chunk);
    }
    let prelude = stream::once(async move { Ok::<Bytes, io::Error>(Bytes::from(prelude)) });
    let rest = stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|chunk| (chunk.map_err(|e| io::Error::other(e.to_string())), rx))
    });
    let body = Body::from_stream(stream::StreamExt::chain(prelude, rest));

    let mut resp = (StatusCode::OK, body).into_response();
    // `content_type` is one of the fixed literals above, so a static header
    // value cannot fail.
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(format.content_type()),
    );
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_selection_and_content_types() {
        assert_eq!(StreamFormat::from_response_format("wav"), StreamFormat::Wav);
        assert_eq!(StreamFormat::from_response_format("f32"), StreamFormat::F32);
        assert_eq!(StreamFormat::from_response_format("pcm"), StreamFormat::Pcm);
        assert_eq!(StreamFormat::Wav.content_type(), "audio/wav");
        assert_eq!(StreamFormat::F32.content_type(), "audio/wav");
        assert_eq!(StreamFormat::Pcm.content_type(), "audio/L16");
    }

    #[test]
    fn pcm_has_no_header_and_wav_headers_are_44_bytes() {
        assert!(StreamFormat::Pcm.header(48_000).unwrap().is_empty());
        assert_eq!(StreamFormat::Wav.header(48_000).unwrap().len(), 44);
        assert_eq!(StreamFormat::F32.header(48_000).unwrap().len(), 44);
    }

    #[test]
    fn chunk_encoding_widths() {
        let s = [0.0f32, 0.5, -0.5];
        assert_eq!(StreamFormat::Wav.encode(&s).len(), 6);
        assert_eq!(StreamFormat::Pcm.encode(&s).len(), 6);
        assert_eq!(StreamFormat::F32.encode(&s).len(), 12);
    }
}
