//! The multipart form of `/v1/audio/transcriptions` and
//! `/v1/audio/translations`, read field by field.
//!
//! A field that fails to read stops the read: the request answers with the
//! read's own error, never with a later "field is required".

use axum::{
    extract::multipart::{Multipart, MultipartError},
    http::StatusCode,
    response::Response,
};
use http_body_util::LengthLimitError;

use crate::server::audio_decode::extension_hint;
use crate::server::body_limit::{body_too_large, BodyLimit};
use crate::server::gen_types::error_response;

/// The fields of one transcription form. Unknown fields are ignored.
#[derive(Debug, Default)]
pub struct TranscriptionForm {
    pub file: Option<Vec<u8>>,
    /// The `file` part's extension, from its file name.
    pub file_ext: Option<String>,
    pub model: Option<String>,
    /// Empty reads as absent.
    pub language: Option<String>,
    /// Empty reads as absent.
    pub response_format: Option<String>,
    pub temperature: Option<f32>,
    /// `task=translate` on the transcriptions route.
    pub translate: bool,
    /// One value per `timestamp_granularities[]` part.
    pub timestamp_granularities: Vec<String>,
}

/// Why a form could not be read.
#[derive(Debug)]
pub enum FormError {
    /// The multipart body failed to read.
    Read(MultipartError),
    /// `temperature` is not a number.
    Temperature(String),
}

impl FormError {
    /// The response for this error. `limit` is the server's body limit,
    /// named in a `413`.
    pub fn into_response(self, limit: Option<BodyLimit>) -> Response {
        match self {
            Self::Read(e) if over_limit(&e) => body_too_large(None, limit.map(|l| l.0)),
            Self::Read(e) => {
                let status = e.status();
                let kind = if status.is_server_error() {
                    "server_error"
                } else {
                    "invalid_request_error"
                };
                error_response(
                    status,
                    &format!(
                        "Reading the multipart form failed: {}. Send the request as \
                         multipart/form-data with a `file` part and a `model` part.",
                        e.body_text()
                    ),
                    kind,
                )
            }
            Self::Temperature(v) => error_response(
                StatusCode::BAD_REQUEST,
                &format!(
                    "Field 'temperature' is '{v}', not a number. Send a number from 0.0 to 1.0."
                ),
                "invalid_request_error",
            ),
        }
    }
}

/// Whether `e` is the body limit cutting the form: axum's own status says
/// so, or a [`LengthLimitError`] sits in its source chain, as it does when
/// the server-wide limit layer cuts a streamed body.
fn over_limit(e: &MultipartError) -> bool {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return true;
    }
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(c) = cause {
        if c.is::<LengthLimitError>() {
            return true;
        }
        cause = c.source();
    }
    false
}

impl From<MultipartError> for FormError {
    fn from(e: MultipartError) -> Self {
        Self::Read(e)
    }
}

/// Reads every part of `multipart`. `force_translate` is the translations
/// route: it ignores `task` and `timestamp_granularities[]`.
pub async fn read_form(
    multipart: &mut Multipart,
    force_translate: bool,
) -> Result<TranscriptionForm, FormError> {
    let mut form = TranscriptionForm {
        translate: force_translate,
        ..TranscriptionForm::default()
    };
    while let Some(field) = multipart.next_field().await? {
        let Some(name) = field.name().map(str::to_string) else {
            continue;
        };
        match name.as_str() {
            "file" => {
                form.file_ext = field
                    .file_name()
                    .and_then(extension_hint)
                    .map(str::to_string);
                form.file = Some(field.bytes().await?.to_vec());
            }
            "model" => form.model = Some(field.text().await?),
            "language" => form.language = non_empty(field.text().await?),
            "response_format" => form.response_format = non_empty(field.text().await?),
            "temperature" => {
                let text = field.text().await?;
                let t = text.trim();
                if !t.is_empty() {
                    form.temperature = Some(
                        t.parse()
                            .map_err(|_| FormError::Temperature(text.clone()))?,
                    );
                }
            }
            "timestamp_granularities[]" if !force_translate => {
                if let Some(v) = non_empty(field.text().await?) {
                    form.timestamp_granularities.push(v);
                }
            }
            // Some clients pass `task=translate` on the transcriptions
            // endpoint: honor it when the route itself is not forcing it.
            "task" if !force_translate => {
                form.translate = field.text().await?.eq_ignore_ascii_case("translate");
            }
            _ => {}
        }
    }
    Ok(form)
}

fn non_empty(s: String) -> Option<String> {
    (!s.is_empty()).then_some(s)
}
