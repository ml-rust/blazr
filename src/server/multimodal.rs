//! Multimodal content types and image/audio processing utilities.
//!
//! Supports OpenAI-compatible content arrays where each message can contain
//! a mix of text, image, and audio content parts. Also handles image URL
//! fetching and base64 decoding.

use serde::{Deserialize, Serialize};

use super::encoding::base64_decode;

/// Message content that can be either a simple string or an array of content parts.
///
/// Matches the OpenAI API format where `content` can be:
/// - A plain string: `"content": "Hello"`
/// - An array of parts: `"content": [{"type": "text", "text": "Hello"}, {"type": "image_url", ...}]`
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content (most common case)
    Text(String),
    /// Array of content parts (multimodal: text + images + audio)
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract all text from the content, concatenating text parts with newlines.
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// Check if this content contains any image parts.
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        }
    }

    /// Check if this content contains any audio parts.
    pub fn has_audio(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::InputAudio { .. })),
        }
    }

    /// Extract all image URLs/data from the content.
    pub fn image_urls(&self) -> Vec<&ImageUrl> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Extract all audio inputs from the content.
    pub fn audio_inputs(&self) -> Vec<&InputAudio> {
        match self {
            MessageContent::Text(_) => Vec::new(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::InputAudio { input_audio } => Some(input_audio),
                    _ => None,
                })
                .collect(),
        }
    }
}

impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

/// A single content part within a multimodal message.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },
    /// Image content via URL or base64
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    /// Audio content via base64
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
}

/// Image URL or base64-encoded image data.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    /// URL (http/https), base64 data URI (`data:image/png;base64,...`), or local path.
    pub url: String,
    /// Optional detail level: "auto", "low", "high"
    #[serde(default = "default_detail")]
    pub detail: String,
}

fn default_detail() -> String {
    "auto".to_string()
}

/// Audio input as base64-encoded data.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputAudio {
    /// Base64-encoded audio data
    pub data: String,
    /// Audio format: "wav", "mp3", "flac", "ogg", "pcm16"
    pub format: String,
}

/// Decoded image data ready for model processing.
pub struct DecodedImage {
    /// Raw image bytes (PNG, JPEG, WebP, etc.)
    pub data: Vec<u8>,
    /// Detected MIME type
    pub mime_type: String,
}

/// Decode an image from a URL string.
///
/// Supports:
/// - `data:image/png;base64,...` — inline base64 data URI
/// - `http://` / `https://` — remote URL (downloaded via reqwest)
/// - Plain base64 string — treated as raw image bytes
pub async fn decode_image(url: &str) -> Result<DecodedImage, String> {
    if let Some(rest) = url.strip_prefix("data:") {
        // Data URI: data:image/png;base64,<data>
        let (meta, data) = rest
            .split_once(",")
            .ok_or_else(|| "Invalid data URI: missing comma".to_string())?;
        let mime_type = meta.split(';').next().unwrap_or("image/png").to_string();
        let bytes = base64_decode(data)?;
        Ok(DecodedImage {
            data: bytes,
            mime_type,
        })
    } else if url.starts_with("http://") || url.starts_with("https://") {
        // Remote URL
        let response = reqwest::get(url)
            .await
            .map_err(|e| format!("Failed to fetch image from {}: {}", url, e))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!(
                "Image fetch failed with status {}: {}",
                status, url
            ));
        }
        let mime_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("image/png")
            .to_string();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read image bytes: {}", e))?;
        Ok(DecodedImage {
            data: bytes.to_vec(),
            mime_type,
        })
    } else {
        // Assume plain base64
        let bytes = base64_decode(url)?;
        let mime_type = detect_image_mime(&bytes);
        Ok(DecodedImage {
            data: bytes,
            mime_type,
        })
    }
}

/// Detect MIME type from image magic bytes.
fn detect_image_mime(data: &[u8]) -> String {
    if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        "image/png".to_string()
    } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg".to_string()
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WEBP" {
        "image/webp".to_string()
    } else if data.starts_with(b"GIF8") {
        "image/gif".to_string()
    } else {
        "application/octet-stream".to_string()
    }
}

/// Decode base64 audio data.
pub fn decode_audio(input: &InputAudio) -> Result<Vec<u8>, String> {
    base64_decode(&input.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_content_text() {
        let json = r#""Hello world""#;
        let content: MessageContent = serde_json::from_str(json).unwrap();
        assert_eq!(content.text(), "Hello world");
        assert!(!content.has_images());
    }

    #[test]
    fn test_message_content_parts() {
        let json = r#"[
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
        ]"#;
        let content: MessageContent = serde_json::from_str(json).unwrap();
        assert_eq!(content.text(), "What is this?");
        assert!(content.has_images());
        assert_eq!(content.image_urls().len(), 1);
        assert_eq!(content.image_urls()[0].url, "https://example.com/img.png");
    }

    #[test]
    fn test_message_content_text_only_parts() {
        let json = r#"[{"type": "text", "text": "Hello"}]"#;
        let content: MessageContent = serde_json::from_str(json).unwrap();
        assert_eq!(content.text(), "Hello");
        assert!(!content.has_images());
    }

    #[test]
    fn test_message_content_multiple_text_parts() {
        let json = r#"[
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"}
        ]"#;
        let content: MessageContent = serde_json::from_str(json).unwrap();
        assert_eq!(content.text(), "Line 1\nLine 2");
    }

    #[test]
    fn test_message_content_audio() {
        let json = r#"[
            {"type": "text", "text": "Transcribe this"},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
        ]"#;
        let content: MessageContent = serde_json::from_str(json).unwrap();
        assert!(content.has_audio());
        assert_eq!(content.audio_inputs().len(), 1);
        assert_eq!(content.audio_inputs()[0].format, "wav");
    }

    #[test]
    fn test_image_url_detail_default() {
        let json = r#"{"url": "https://example.com/img.png"}"#;
        let img: ImageUrl = serde_json::from_str(json).unwrap();
        assert_eq!(img.detail, "auto");
    }

    #[test]
    fn test_base64_decode_simple() {
        let encoded = "SGVsbG8="; // "Hello"
        let decoded = base64_decode(encoded).unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_base64_decode_url_safe() {
        // URL-safe base64 with - and _
        let standard = base64_decode("SGVsbG8=").unwrap();
        assert_eq!(standard, b"Hello");
    }

    #[test]
    fn test_base64_decode_no_padding() {
        let decoded = base64_decode("SGVsbG8").unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_detect_image_mime() {
        assert_eq!(
            detect_image_mime(&[0x89, 0x50, 0x4E, 0x47, 0x0D]),
            "image/png"
        );
        assert_eq!(detect_image_mime(&[0xFF, 0xD8, 0xFF, 0xE0]), "image/jpeg");
        assert_eq!(detect_image_mime(&[0x00, 0x01]), "application/octet-stream");
    }

    #[test]
    fn test_data_uri_parsing() {
        // Test synchronous base64 decode from data URI format
        let uri = "data:image/png;base64,iVBORw0KGgo=";
        let rest = uri.strip_prefix("data:").unwrap();
        let (meta, data) = rest.split_once(",").unwrap();
        let mime = meta.split(';').next().unwrap();
        assert_eq!(mime, "image/png");
        assert!(base64_decode(data).is_ok());
    }
}
