//! Image blocks of an Anthropic Messages request: the `source` shapes the
//! server reads, and the conversion of a conversation into chat messages
//! with a marker at each image.

use serde::Deserialize;

use super::anthropic_types::{AnthropicContent, AnthropicContentBlock, AnthropicRequest};
use crate::model::chat_template::ChatMessage;
use crate::server::chat_vision::{marked_text, ImageSource, MediaPart};

/// Where an `image` block's bytes come from.
#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum AnthropicImageSource {
    /// Inline bytes with their declared media type.
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
    /// A remote `http(s)` URL the server fetches.
    #[serde(rename = "url")]
    Url { url: String },
    /// A source kind this server does not read, such as `file`.
    #[serde(other)]
    Other,
}

impl AnthropicImageSource {
    /// Where this source's bytes come from.
    ///
    /// # Errors
    ///
    /// The source kind is neither `base64` nor `url`.
    fn image_source(&self) -> Result<ImageSource<'_>, String> {
        match self {
            AnthropicImageSource::Base64 { media_type, data } => Ok(ImageSource::Base64 {
                media_type: media_type.as_str(),
                data: data.as_str(),
            }),
            AnthropicImageSource::Url { url } => Ok(ImageSource::Url(url.as_str())),
            AnthropicImageSource::Other => Err(
                "image source type is not supported; send a `base64` or a `url` source".to_string(),
            ),
        }
    }
}

impl AnthropicContent {
    /// The content as the marker helper reads it: a `text` block as text, an
    /// `image` block as an image.
    pub(in crate::server) fn media_parts(&self) -> Vec<MediaPart<'_>> {
        match self {
            AnthropicContent::Text(s) => vec![MediaPart::Text(s.as_str())],
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .map(|b| match b {
                    AnthropicContentBlock::Text { text } => MediaPart::Text(text.as_str()),
                    AnthropicContentBlock::Image { .. } => MediaPart::Image,
                })
                .collect(),
        }
    }

    /// Where every `image` block's bytes come from, in order.
    ///
    /// # Errors
    ///
    /// An `image` block names a source kind the server does not read.
    pub(in crate::server) fn image_sources(&self) -> Result<Vec<ImageSource<'_>>, String> {
        let AnthropicContent::Blocks(blocks) = self else {
            return Ok(Vec::new());
        };
        let mut sources = Vec::new();
        for (index, block) in blocks.iter().enumerate() {
            if let AnthropicContentBlock::Image { source } = block {
                let source = source
                    .image_source()
                    .map_err(|e| format!("image block {}: {e}", index + 1))?;
                sources.push(source);
            }
        }
        Ok(sources)
    }
}

impl AnthropicRequest {
    /// Where every `image` block's bytes come from, across the messages in
    /// order.
    ///
    /// # Errors
    ///
    /// An `image` block names a source kind the server does not read.
    pub(in crate::server) fn image_sources(&self) -> Result<Vec<ImageSource<'_>>, String> {
        let mut sources = Vec::new();
        for (index, message) in self.messages.iter().enumerate() {
            let found = message
                .content
                .image_sources()
                .map_err(|e| format!("message {}: {e}", index + 1))?;
            sources.extend(found);
        }
        Ok(sources)
    }

    /// The conversation as chat messages, with `image_marker` written in at
    /// each image block's own position.
    pub(in crate::server) fn chat_messages(&self, image_marker: Option<&str>) -> Vec<ChatMessage> {
        let mut msgs: Vec<ChatMessage> = Vec::new();
        if let Some(ref sys) = self.system {
            msgs.push(ChatMessage::new("system", sys.text()));
        }
        for m in &self.messages {
            let content = marked_text(&m.content.media_parts(), image_marker);
            msgs.push(ChatMessage::new(m.role.clone(), content));
        }
        msgs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both documented source kinds parse, keeping the media type and the
    /// URL the decoder needs.
    #[test]
    fn test_anthropic_image_blocks_base64_and_url() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
                ]}
            ]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        let sources = req.image_sources().expect("both sources are readable");
        assert_eq!(sources.len(), 2);
        match &sources[0] {
            ImageSource::Base64 { media_type, data } => {
                assert_eq!(*media_type, "image/png");
                assert_eq!(*data, "iVBORw0KGgo=");
            }
            other => panic!("expected a base64 source, got {other:?}"),
        }
        match &sources[1] {
            ImageSource::Url(url) => assert_eq!(*url, "https://example.com/img.png"),
            other => panic!("expected a url source, got {other:?}"),
        }
    }

    /// A mixed message keeps the text and marks the image in place.
    #[test]
    fn test_anthropic_mixed_text_and_image_blocks() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                    {"type": "text", "text": "Be brief."}
                ]}
            ]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            marked_text(&req.messages[0].content.media_parts(), None),
            "[image]\nWhat is this?\nBe brief."
        );
        assert_eq!(req.image_sources().map(|s| s.len()), Ok(1));
        let msgs = req.chat_messages(Some("<marker>"));
        assert_eq!(msgs[0].content, "What is this?\n<marker>\nBe brief.");
        let unmarked = req.chat_messages(None);
        assert_eq!(unmarked[0].content, "[image]\nWhat is this?\nBe brief.");
    }

    #[test]
    fn test_anthropic_unsupported_image_source_is_named() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image", "source": {"type": "file", "file_id": "f_1"}}
                ]}
            ]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        let err = req.image_sources().expect_err("file sources are not read");
        assert!(err.contains("message 1"), "{err}");
        assert!(err.contains("image block 2"), "{err}");
    }

    /// A plain-string message names no images and renders as its own text.
    #[test]
    fn test_anthropic_text_message_has_no_image_sources() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "system": "Be concise",
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;
        let req: AnthropicRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.image_sources().map(|s| s.len()), Ok(0));
        let msgs = req.chat_messages(Some("<marker>"));
        assert_eq!(msgs[0].content, "Be concise");
        assert_eq!(msgs[1].content, "Hello");
    }
}
