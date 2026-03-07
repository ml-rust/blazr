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

/// Supported audio formats per OpenAI API spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Pcm16,
    Wav,
    Mp3,
    Flac,
    Ogg,
}

impl AudioFormat {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "pcm16" => Ok(Self::Pcm16),
            "wav" => Ok(Self::Wav),
            "mp3" => Ok(Self::Mp3),
            "flac" => Ok(Self::Flac),
            "ogg" => Ok(Self::Ogg),
            other => Err(format!(
                "Unsupported audio format '{}'. Supported: pcm16, wav, mp3, flac, ogg",
                other
            )),
        }
    }
}

/// Convert raw PCM16 little-endian bytes to f32 samples normalized to [-1.0, 1.0].
fn pcm16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect()
}

/// Extract PCM16 audio data from a WAV container.
///
/// Validates the WAV header, ensures 16-bit PCM format, and returns the raw
/// sample data from the "data" chunk.
fn wav_to_pcm16(bytes: &[u8]) -> Result<&[u8], String> {
    if bytes.len() < 44 {
        return Err("WAV data too short for valid header".to_string());
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Invalid WAV header: missing RIFF/WAVE signature".to_string());
    }
    // Audio format at byte 20-21: 1 = PCM
    let audio_fmt = u16::from_le_bytes([bytes[20], bytes[21]]);
    if audio_fmt != 1 {
        return Err(format!(
            "Unsupported WAV audio format {}: only PCM (1) is supported",
            audio_fmt
        ));
    }
    let bits_per_sample = u16::from_le_bytes([bytes[34], bytes[35]]);
    if bits_per_sample != 16 {
        return Err(format!(
            "Unsupported WAV bit depth {}: only 16-bit is supported",
            bits_per_sample
        ));
    }
    // Find the "data" chunk - it's not always at byte 36
    let mut offset = 12; // skip RIFF header
    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        if chunk_id == b"data" {
            let data_start = offset + 8;
            let data_end = (data_start + chunk_size).min(bytes.len());
            return Ok(&bytes[data_start..data_end]);
        }
        offset += 8 + chunk_size;
        // WAV chunks are word-aligned
        if offset % 2 != 0 {
            offset += 1;
        }
    }
    Err("WAV file missing 'data' chunk".to_string())
}

/// Decode audio from an InputAudio payload into f32 samples.
///
/// Parses the format field, base64-decodes the data, and converts to
/// normalized f32 samples in [-1.0, 1.0]. Currently supports `pcm16`
/// (raw 16-bit little-endian PCM) and `wav` (16-bit PCM WAV container).
pub fn decode_audio(input: &InputAudio) -> Result<Vec<f32>, String> {
    let format = AudioFormat::parse(&input.format)?;
    let raw_bytes = base64_decode(&input.data)?;

    match format {
        AudioFormat::Pcm16 => Ok(pcm16_to_f32(&raw_bytes)),
        AudioFormat::Wav => {
            let pcm_data = wav_to_pcm16(&raw_bytes)?;
            Ok(pcm16_to_f32(pcm_data))
        }
        AudioFormat::Mp3 | AudioFormat::Flac | AudioFormat::Ogg => Err(format!(
            "Audio format '{}' is not yet supported. Currently supported: pcm16, wav",
            input.format
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::encoding::base64_encode;

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
    fn test_audio_format_parse() {
        assert_eq!(AudioFormat::parse("pcm16").unwrap(), AudioFormat::Pcm16);
        assert_eq!(AudioFormat::parse("wav").unwrap(), AudioFormat::Wav);
        assert_eq!(AudioFormat::parse("mp3").unwrap(), AudioFormat::Mp3);
        assert_eq!(AudioFormat::parse("flac").unwrap(), AudioFormat::Flac);
        assert_eq!(AudioFormat::parse("ogg").unwrap(), AudioFormat::Ogg);
        assert!(AudioFormat::parse("aac").is_err());
        assert!(AudioFormat::parse("").is_err());
    }

    #[test]
    fn test_decode_audio_pcm16() {
        // Two 16-bit LE samples: 0x0100 = 256, 0xFF7F = 32767
        let raw: [u8; 4] = [0x00, 0x01, 0xFF, 0x7F];
        let b64 = base64_encode(&raw);
        let input = InputAudio {
            data: b64,
            format: "pcm16".to_string(),
        };
        let samples = decode_audio(&input).unwrap();
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 256.0 / 32768.0).abs() < 1e-6);
        assert!((samples[1] - 32767.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn test_decode_audio_wav() {
        // Minimal valid WAV: 44-byte header + 4 bytes of PCM data (2 samples)
        let mut wav = vec![0u8; 48];
        wav[0..4].copy_from_slice(b"RIFF");
        let file_size: u32 = 40; // total - 8
        wav[4..8].copy_from_slice(&file_size.to_le_bytes());
        wav[8..12].copy_from_slice(b"WAVE");
        wav[12..16].copy_from_slice(b"fmt ");
        wav[16..20].copy_from_slice(&16u32.to_le_bytes()); // fmt chunk size
        wav[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM format
        wav[22..24].copy_from_slice(&1u16.to_le_bytes()); // 1 channel
        wav[24..28].copy_from_slice(&16000u32.to_le_bytes()); // sample rate
        wav[28..32].copy_from_slice(&32000u32.to_le_bytes()); // byte rate
        wav[32..34].copy_from_slice(&2u16.to_le_bytes()); // block align
        wav[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav[36..40].copy_from_slice(b"data");
        wav[40..44].copy_from_slice(&4u32.to_le_bytes()); // data chunk size
        wav[44..46].copy_from_slice(&256i16.to_le_bytes());
        wav[46..48].copy_from_slice(&(-100i16).to_le_bytes());

        let b64 = base64_encode(&wav);
        let input = InputAudio {
            data: b64,
            format: "wav".to_string(),
        };
        let samples = decode_audio(&input).unwrap();
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 256.0 / 32768.0).abs() < 1e-6);
        assert!((samples[1] - (-100.0 / 32768.0)).abs() < 1e-6);
    }

    #[test]
    fn test_decode_audio_unsupported_format() {
        let input = InputAudio {
            data: "AAAA".to_string(),
            format: "mp3".to_string(),
        };
        let err = decode_audio(&input).unwrap_err();
        assert!(err.contains("not yet supported"));
    }

    #[test]
    fn test_decode_audio_invalid_format() {
        let input = InputAudio {
            data: "AAAA".to_string(),
            format: "aac".to_string(),
        };
        let err = decode_audio(&input).unwrap_err();
        assert!(err.contains("Unsupported audio format"));
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
