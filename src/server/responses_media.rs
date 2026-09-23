//! Image parts of a Responses API request: where an `input_image` part's
//! bytes come from, and how the parts render with a marker per image.

use super::chat_vision::{ImageSource, MediaPart};
use super::responses_types::{ResponsesContent, ResponsesContentPart, ResponsesInput};

impl ResponsesContent {
    /// The content as the marker helper reads it: `input_text` as text,
    /// `input_image` as an image, `input_audio` dropped.
    pub(super) fn media_parts(&self) -> Vec<MediaPart<'_>> {
        match self {
            ResponsesContent::Text(s) => vec![MediaPart::Text(s.as_str())],
            ResponsesContent::Parts(parts) => parts
                .iter()
                .map(|p| match p {
                    ResponsesContentPart::InputText { text } => MediaPart::Text(text.as_str()),
                    ResponsesContentPart::InputImage { .. } => MediaPart::Image,
                    ResponsesContentPart::InputAudio { .. } => MediaPart::Dropped,
                })
                .collect(),
        }
    }

    /// Where each `input_image` part's bytes come from, in order.
    ///
    /// # Errors
    ///
    /// An `input_image` part carries no `image_url`.
    pub(super) fn image_sources(&self) -> Result<Vec<ImageSource<'_>>, String> {
        let ResponsesContent::Parts(parts) = self else {
            return Ok(Vec::new());
        };
        let mut sources = Vec::new();
        for (index, part) in parts.iter().enumerate() {
            if let ResponsesContentPart::InputImage { image_url, .. } = part {
                let url = image_url.as_deref().ok_or_else(|| {
                    format!(
                        "input_image part {} has no image_url; send a data URL or an http(s) URL",
                        index + 1
                    )
                })?;
                sources.push(ImageSource::Url(url));
            }
        }
        Ok(sources)
    }
}

impl ResponsesInput {
    /// Where every `input_image` part's bytes come from, across the items in
    /// order.
    ///
    /// # Errors
    ///
    /// An `input_image` part carries no `image_url`.
    pub(super) fn image_sources(&self) -> Result<Vec<ImageSource<'_>>, String> {
        let ResponsesInput::Items(items) = self else {
            return Ok(Vec::new());
        };
        let mut sources = Vec::new();
        for item in items {
            sources.extend(item.content.image_sources()?);
        }
        Ok(sources)
    }
}

#[cfg(test)]
mod tests {
    use super::super::chat_vision::marked_text;
    use super::super::responses_types::ResponsesRequest;
    use super::*;

    /// `input_image` parses in both URL forms, and the data URL keeps its
    /// full string so the decoder sees the media type.
    #[test]
    fn test_responses_input_image_url_forms() {
        let json = r#"{
            "model": "test",
            "input": [
                {"role": "user", "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
                    {"type": "input_image", "image_url": "https://example.com/img.png", "detail": "high"}
                ]}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        let sources = req.input.image_sources().expect("both parts carry a url");
        assert_eq!(sources.len(), 2);
        let urls: Vec<&str> = sources
            .iter()
            .map(|s| match s {
                ImageSource::Url(url) => *url,
                ImageSource::Base64 { .. } => panic!("expected url sources"),
            })
            .collect();
        assert_eq!(urls[0], "data:image/png;base64,iVBORw0KGgo=");
        assert_eq!(urls[1], "https://example.com/img.png");
    }

    /// A mixed item keeps the text and marks the image at its own position.
    #[test]
    fn test_responses_input_mixed_text_and_image() {
        let json = r#"{
            "model": "test",
            "input": [
                {"role": "user", "content": [
                    {"type": "input_text", "text": "What is this?"},
                    {"type": "input_image", "image_url": "https://example.com/a.png"},
                    {"type": "input_text", "text": "Be brief."}
                ]}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        let ResponsesInput::Items(items) = &req.input else {
            panic!("expected items input");
        };
        assert_eq!(
            marked_text(&items[0].content.media_parts(), None),
            "[image]\nWhat is this?\nBe brief."
        );
        let marked = super::super::chat_vision::marked_text(
            &items[0].content.media_parts(),
            Some("<marker>"),
        );
        assert_eq!(marked, "What is this?\n<marker>\nBe brief.");
        assert_eq!(req.input.image_sources().map(|s| s.len()), Ok(1));
    }

    #[test]
    fn test_responses_input_image_without_url_is_named() {
        let json = r#"{
            "model": "test",
            "input": [
                {"role": "user", "content": [{"type": "input_image"}]}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        let err = req.input.image_sources().expect_err("no image_url");
        assert!(err.contains("input_image part 1"), "{err}");
    }

    /// A plain string input names no images at all.
    #[test]
    fn test_responses_text_input_has_no_image_sources() {
        let json = r#"{"model": "test", "input": "Hello"}"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input.image_sources().map(|s| s.len()), Ok(0));
    }
}
