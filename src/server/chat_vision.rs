//! Media in a chat request: the decoded image and audio parts, which
//! generation path serves them, and the marker check that path needs.

use axum::{http::StatusCode, response::Response};
use splintr::FxHashSet;

use super::generation::error_response;
use super::tools::ChatRequestMessage;
use crate::engine::vision_tower::{IMAGE_MARKER_TEXT, MARKER_NAMES, NO_TOWER_ERROR};

/// Decoded media of one request, in message order.
#[derive(Default)]
pub(super) struct RequestMedia {
    /// Encoded image bytes, one entry per `image_url` part.
    pub images: Vec<Vec<u8>>,
    /// PCM samples, one entry per `input_audio` part.
    pub audio: Vec<Vec<f32>>,
}

impl RequestMedia {
    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.audio.is_empty()
    }
}

/// Which generation path a request's media takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ImageRoute {
    /// Text only.
    Text,
    /// The `qwen35` vision tower: markers in the text, image rows spliced
    /// into the prefill.
    Tower,
    /// A model with its own vision or audio encoder: media prepended to
    /// the text embeddings.
    Multimodal,
}

impl ImageRoute {
    /// What replaces an image part in the message text.
    pub fn image_marker(self) -> Option<&'static str> {
        match self {
            ImageRoute::Tower => Some(IMAGE_MARKER_TEXT),
            ImageRoute::Text | ImageRoute::Multimodal => None,
        }
    }

    /// Control tokens the server wrote into the messages on this route.
    pub fn server_markers(self) -> FxHashSet<String> {
        match self {
            ImageRoute::Tower => MARKER_NAMES.iter().map(|s| s.to_string()).collect(),
            ImageRoute::Text | ImageRoute::Multimodal => FxHashSet::default(),
        }
    }
}

/// Pick the route for `media` on an executor that has a vision tower
/// (`has_tower`) or a model with its own encoders (`multimodal_model`).
///
/// # Errors
///
/// The message for a 400: images without a tower or encoder, or audio on
/// a tower.
pub(super) fn image_route(
    has_tower: bool,
    multimodal_model: bool,
    media: &RequestMedia,
) -> Result<ImageRoute, String> {
    if media.is_empty() {
        return Ok(ImageRoute::Text);
    }
    if has_tower {
        if !media.audio.is_empty() {
            return Err("audio input is not supported by the vision tower".to_string());
        }
        return Ok(ImageRoute::Tower);
    }
    if multimodal_model {
        return Ok(ImageRoute::Multimodal);
    }
    if media.images.is_empty() {
        return Err("audio input is not supported by this model".to_string());
    }
    Err(NO_TOWER_ERROR.to_string())
}

/// Decode every image and audio part of `messages`, in order.
///
/// # Errors
///
/// The 400 response naming the part that did not decode.
pub(super) async fn decode_request_media(
    messages: &[ChatRequestMessage],
) -> Result<RequestMedia, Box<Response>> {
    let mut media = RequestMedia::default();
    for msg in messages {
        let Some(ref content) = msg.content else {
            continue;
        };
        for img_url in content.image_urls() {
            match super::multimodal::decode_image(&img_url.url).await {
                Ok(decoded) => media.images.push(decoded.data),
                Err(e) => {
                    return Err(Box::new(error_response(
                        StatusCode::BAD_REQUEST,
                        &format!("Failed to decode image: {e}"),
                        "invalid_request_error",
                    )));
                }
            }
        }
        for audio_input in content.audio_inputs() {
            match super::multimodal::decode_audio(audio_input) {
                Ok(samples) => media.audio.push(samples),
                Err(e) => {
                    return Err(Box::new(error_response(
                        StatusCode::BAD_REQUEST,
                        &format!("Failed to decode audio: {e}"),
                        "invalid_request_error",
                    )));
                }
            }
        }
    }
    if !media.is_empty() {
        tracing::info!(
            images = media.images.len(),
            audio_segments = media.audio.len(),
            "Multimodal content extracted from request"
        );
    }
    Ok(media)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_images(n: usize) -> RequestMedia {
        RequestMedia {
            images: vec![vec![0u8; 4]; n],
            audio: Vec::new(),
        }
    }

    #[test]
    fn text_only_takes_the_text_route_whatever_is_loaded() {
        let media = RequestMedia::default();
        assert_eq!(image_route(false, false, &media), Ok(ImageRoute::Text));
        assert_eq!(image_route(true, false, &media), Ok(ImageRoute::Text));
        assert_eq!(image_route(false, true, &media), Ok(ImageRoute::Text));
    }

    #[test]
    fn images_without_a_tower_are_refused_naming_mmproj() {
        let err = image_route(false, false, &with_images(1)).expect_err("no tower, no encoder");
        assert_eq!(err, NO_TOWER_ERROR);
        assert!(err.contains("--mmproj"), "{err}");
    }

    #[test]
    fn a_tower_takes_images_and_refuses_audio() {
        assert_eq!(
            image_route(true, false, &with_images(2)),
            Ok(ImageRoute::Tower)
        );
        let media = RequestMedia {
            images: Vec::new(),
            audio: vec![vec![0.0; 8]],
        };
        assert!(image_route(true, false, &media).is_err());
    }

    #[test]
    fn a_multimodal_model_keeps_its_own_path() {
        assert_eq!(
            image_route(false, true, &with_images(1)),
            Ok(ImageRoute::Multimodal)
        );
        assert!(ImageRoute::Multimodal.image_marker().is_none());
        assert!(ImageRoute::Multimodal.server_markers().is_empty());
    }

    #[test]
    fn the_tower_route_names_its_three_markers() {
        let markers = ImageRoute::Tower.server_markers();
        assert_eq!(markers.len(), 3);
        for name in MARKER_NAMES {
            assert!(markers.contains(name), "{name}");
        }
        assert_eq!(ImageRoute::Tower.image_marker(), Some(IMAGE_MARKER_TEXT));
    }
}
