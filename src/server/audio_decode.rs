//! Audio decoding + resampling to Whisper's expected input (16 kHz mono f32).
//!
//! blazr holds no core logic — the container decode and the polyphase
//! resampler both live in `boostr::model::audio`, so training and any other
//! inference server get them too. This module is a thin delegation shim that
//! adapts boostr's `Result` to `anyhow::Result` for the HTTP handlers in
//! [`super::audio`].

use anyhow::{anyhow, Result};

use boostr::model::audio::decode_audio_mono_at;

/// Target sample rate for Whisper input.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Decode arbitrary audio bytes into 16 kHz mono f32 samples in `[-1, 1]`.
///
/// The caller typically passes the bytes from the multipart `file` field of
/// `/v1/audio/transcriptions`. `hint` is an optional file-extension hint
/// (`"wav"`, `"mp3"`, ...) to help boostr's format probe.
pub fn decode_to_whisper_input(bytes: &[u8], hint: Option<&str>) -> Result<Vec<f32>> {
    decode_audio_mono_at(bytes, hint, WHISPER_SAMPLE_RATE)
        .map_err(|e| anyhow!("decoding audio to whisper input: {e}"))
}

/// Extract a likely file-extension hint from a multipart filename like
/// `"audio.mp3"` or `"recording.wav"`. Returns `None` for bare names.
pub fn extension_hint(filename: &str) -> Option<&str> {
    boostr::model::audio::extension_hint(filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_hint_parses() {
        assert_eq!(extension_hint("a.wav"), Some("wav"));
        assert_eq!(extension_hint("foo.bar.mp3"), Some("mp3"));
        assert_eq!(extension_hint("noext"), None);
    }
}
