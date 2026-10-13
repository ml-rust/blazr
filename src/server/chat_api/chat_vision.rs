//! Media in a chat request: the decoded image and audio parts, which
//! generation path serves them, and the marker layout that path needs. Every
//! chat route shares these pieces.

use splintr::FxHashSet;

use super::tools::ChatRequestMessage;
use crate::engine::vision_tower::{IMAGE_MARKER_TEXT, MARKER_NAMES, NO_TOWER_ERROR};

/// Decoded media of one request, in message order.
#[derive(Default)]
pub(in crate::server) struct RequestMedia {
    /// Encoded image bytes, one entry per image part.
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
pub(in crate::server) enum ImageRoute {
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
pub(in crate::server) fn image_route(
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
/// The 400 message naming the part that did not decode.
pub(in crate::server) async fn decode_request_media(
    messages: &[ChatRequestMessage],
) -> Result<RequestMedia, String> {
    let mut media = RequestMedia::default();
    for msg in messages {
        let Some(ref content) = msg.content else {
            continue;
        };
        for img_url in content.image_urls() {
            let decoded = crate::server::multimodal::decode_image(&img_url.url)
                .await
                .map_err(|e| format!("Failed to decode image: {e}"))?;
            media.images.push(decoded.data);
        }
        for audio_input in content.audio_inputs() {
            let samples = crate::server::multimodal_audio::decode_audio(audio_input)
                .map_err(|e| format!("Failed to decode audio: {e}"))?;
            media.audio.push(samples);
        }
    }
    log_media(&media);
    Ok(media)
}

/// Where one image part's bytes come from.
#[derive(Debug)]
pub(in crate::server) enum ImageSource<'a> {
    /// An OpenAI-style `image_url`: a data URL, an `http(s)` URL, or base64.
    Url(&'a str),
    /// An Anthropic-style `source`: the declared media type and base64 bytes.
    Base64 { media_type: &'a str, data: &'a str },
}

/// Decode `sources` in order, for a route whose messages carry images only.
///
/// # Errors
///
/// The 400 message naming the 1-based part that did not decode.
pub(in crate::server) async fn decode_image_sources(
    sources: &[ImageSource<'_>],
) -> Result<RequestMedia, String> {
    let mut media = RequestMedia::default();
    for (index, source) in sources.iter().enumerate() {
        let decoded = match source {
            ImageSource::Url(url) => crate::server::multimodal::decode_image(url).await,
            ImageSource::Base64 { media_type, data } => {
                crate::server::multimodal::decode_image_source(media_type, data)
            }
        };
        let decoded =
            decoded.map_err(|e| format!("Failed to decode image part {}: {e}", index + 1))?;
        media.images.push(decoded.data);
    }
    log_media(&media);
    Ok(media)
}

fn log_media(media: &RequestMedia) {
    if !media.is_empty() {
        tracing::info!(
            images = media.images.len(),
            audio_segments = media.audio.len(),
            "Multimodal content extracted from request"
        );
    }
}

/// One part of a message as the marker helper reads it.
#[derive(Debug)]
pub(in crate::server) enum MediaPart<'a> {
    Text(&'a str),
    Image,
    /// A part the prompt carries nothing for.
    Dropped,
}

/// `parts` as message text with its images marked, the one layout every chat
/// route renders.
///
/// With `Some(marker)` each image becomes that text at its own position among
/// the parts, joined by newlines. With `None` the images become a single
/// `[image]` or `[N images]` note in front of the text. A `Dropped` part
/// contributes nothing either way.
pub(in crate::server) fn marked_text(
    parts: &[MediaPart<'_>],
    image_marker: Option<&str>,
) -> String {
    if let Some(marker) = image_marker {
        return parts
            .iter()
            .filter_map(|p| match p {
                MediaPart::Text(text) => Some(*text),
                MediaPart::Image => Some(marker),
                MediaPart::Dropped => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
    }
    let text = parts
        .iter()
        .filter_map(|p| match p {
            MediaPart::Text(text) => Some(*text),
            MediaPart::Image | MediaPart::Dropped => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let images = parts
        .iter()
        .filter(|p| matches!(p, MediaPart::Image))
        .count();
    if images == 0 {
        return text;
    }
    let note = if images == 1 {
        "[image]".to_string()
    } else {
        format!("[{images} images]")
    };
    if text.is_empty() {
        note
    } else {
        format!("{note}\n{text}")
    }
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

    /// With a marker, two images give two marker groups in message order and
    /// the text keeps its order around them.
    #[test]
    fn marked_text_keeps_part_order_with_one_marker_per_image() {
        let parts = [
            MediaPart::Text("First:"),
            MediaPart::Image,
            MediaPart::Text("Second:"),
            MediaPart::Image,
            MediaPart::Text("Compare them."),
        ];
        let rendered = marked_text(&parts, Some(IMAGE_MARKER_TEXT));
        assert_eq!(
            rendered,
            format!("First:\n{IMAGE_MARKER_TEXT}\nSecond:\n{IMAGE_MARKER_TEXT}\nCompare them.")
        );
        assert_eq!(rendered.matches(IMAGE_MARKER_TEXT).count(), 2);
    }

    #[test]
    fn marked_text_without_a_marker_notes_the_image_count() {
        let one = [MediaPart::Text("What is this?"), MediaPart::Image];
        assert_eq!(marked_text(&one, None), "[image]\nWhat is this?");
        let two = [MediaPart::Image, MediaPart::Image, MediaPart::Text("Both?")];
        assert_eq!(marked_text(&two, None), "[2 images]\nBoth?");
        let bare = [MediaPart::Image];
        assert_eq!(marked_text(&bare, None), "[image]");
    }

    #[test]
    fn marked_text_drops_parts_the_prompt_carries_nothing_for() {
        let parts = [
            MediaPart::Text("Listen:"),
            MediaPart::Dropped,
            MediaPart::Text("and answer."),
        ];
        assert_eq!(marked_text(&parts, None), "Listen:\nand answer.");
        assert_eq!(
            marked_text(&parts, Some(IMAGE_MARKER_TEXT)),
            "Listen:\nand answer."
        );
    }

    #[test]
    fn marked_text_of_plain_text_is_that_text() {
        let parts = [MediaPart::Text("Hello")];
        assert_eq!(marked_text(&parts, None), "Hello");
        assert_eq!(marked_text(&parts, Some(IMAGE_MARKER_TEXT)), "Hello");
    }
}
