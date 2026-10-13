//! Audio parts of a chat request: the formats the server reads and the
//! conversion from an `input_audio` payload to f32 samples.

use super::encoding::base64_decode;
use super::multimodal::InputAudio;

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
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| {
            let sample = i16::from_le_bytes(*chunk);
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
}
