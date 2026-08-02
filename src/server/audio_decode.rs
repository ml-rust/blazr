//! Audio decoding + resampling to Whisper's expected input (16 kHz mono f32).
//!
//! Input: arbitrary audio bytes (WAV/MP3/FLAC/OGG-Vorbis detected by symphonia).
//! Output: `Vec<f32>` of mono samples in `[-1, 1]` at 16 kHz.
//!
//! Multi-channel audio is mixed down by averaging channels. Sample rates other
//! than 16 kHz are passed through `rubato::FftFixedInOut` for high-quality
//! band-limited resampling.

use anyhow::{anyhow, Context, Result};
use rubato::{FftFixedInOut, Resampler};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Target sample rate for Whisper input.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Decode arbitrary audio bytes into 16 kHz mono f32 samples in `[-1, 1]`.
///
/// The caller typically passes the bytes from the multipart `file` field of
/// `/v1/audio/transcriptions`. `hint` is an optional file-extension hint
/// (`"wav"`, `"mp3"`, ...) to help symphonia's format probe.
pub fn decode_to_whisper_input(bytes: &[u8], hint: Option<&str>) -> Result<Vec<f32>> {
    let cursor = std::io::Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let mut probe_hint = Hint::new();
    if let Some(ext) = hint {
        probe_hint.with_extension(ext);
    }

    let probe = symphonia::default::get_probe()
        .format(
            &probe_hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("probing audio format")?;
    let mut format = probe.format;

    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("audio file has no default track"))?;
    let codec_params = track.codec_params.clone();
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .context("creating audio decoder")?;

    let input_rate = codec_params
        .sample_rate
        .ok_or_else(|| anyhow!("audio file missing sample_rate"))?;
    let channels = codec_params
        .channels
        .ok_or_else(|| anyhow!("audio file missing channel layout"))?
        .count();

    // Collect all mono samples at the source rate first, then resample once.
    let mut mono: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(anyhow!("reading packet: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let audio = match decoder.decode(&packet) {
            Ok(a) => a,
            Err(SymphoniaError::DecodeError(_)) => continue, // skip corrupt frames
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => return Err(anyhow!("decoding packet: {e}")),
        };
        append_mono_samples(&audio, channels, &mut mono);
    }

    if mono.is_empty() {
        return Err(anyhow!("decoded audio is empty"));
    }

    if input_rate == WHISPER_SAMPLE_RATE {
        return Ok(mono);
    }

    resample_to_16k(&mono, input_rate)
}

fn append_mono_samples(audio: &AudioBufferRef<'_>, channels: usize, out: &mut Vec<f32>) {
    macro_rules! mix {
        ($buf:expr, $convert:expr) => {{
            let frames = $buf.frames();
            if channels <= 1 {
                for i in 0..frames {
                    out.push($convert($buf.chan(0)[i]));
                }
            } else {
                for i in 0..frames {
                    let mut acc = 0.0f32;
                    for c in 0..channels {
                        acc += $convert($buf.chan(c)[i]);
                    }
                    out.push(acc / channels as f32);
                }
            }
        }};
    }
    match audio {
        AudioBufferRef::F32(b) => mix!(b, |x: f32| x),
        AudioBufferRef::F64(b) => mix!(b, |x: f64| x as f32),
        AudioBufferRef::S16(b) => mix!(b, |x: i16| x as f32 / i16::MAX as f32),
        AudioBufferRef::S32(b) => mix!(b, |x: i32| x as f32 / i32::MAX as f32),
        AudioBufferRef::U8(b) => mix!(b, |x: u8| (x as f32 - 128.0) / 128.0),
        AudioBufferRef::U16(b) => {
            mix!(b, |x: u16| (x as f32 - u16::MAX as f32 / 2.0)
                / (u16::MAX as f32 / 2.0))
        }
        AudioBufferRef::U32(b) => mix!(b, |x: u32| (x as f32 / u32::MAX as f32) * 2.0 - 1.0),
        AudioBufferRef::S8(b) => mix!(b, |x: i8| x as f32 / i8::MAX as f32),
        AudioBufferRef::S24(b) => mix!(b, |x: symphonia::core::sample::i24| x.inner() as f32
            / 8_388_607.0),
        AudioBufferRef::U24(b) => mix!(b, |x: symphonia::core::sample::u24| (x.inner() as f32
            / 16_777_215.0)
            * 2.0
            - 1.0),
    }
}

fn resample_to_16k(input: &[f32], input_rate: u32) -> Result<Vec<f32>> {
    // rubato's FFT resampler wants fixed-size input chunks. We pick a chunk
    // size that's friendly to both rates and pad the final partial chunk with
    // zeros — the tail is trimmed by the known output-length ratio.
    let chunk = 1024usize;
    let mut resampler = FftFixedInOut::<f32>::new(
        input_rate as usize,
        WHISPER_SAMPLE_RATE as usize,
        chunk,
        1, // mono
    )
    .map_err(|e| anyhow!("building resampler: {e}"))?;

    let in_chunk_size = resampler.input_frames_next();
    let mut out = Vec::with_capacity(
        input.len() * WHISPER_SAMPLE_RATE as usize / input_rate as usize + chunk,
    );
    let mut scratch_in = vec![vec![0.0f32; in_chunk_size]];

    let mut pos = 0;
    while pos < input.len() {
        let take = (input.len() - pos).min(in_chunk_size);
        scratch_in[0][..take].copy_from_slice(&input[pos..pos + take]);
        if take < in_chunk_size {
            for v in &mut scratch_in[0][take..] {
                *v = 0.0;
            }
        }
        let produced = resampler
            .process(&scratch_in, None)
            .map_err(|e| anyhow!("resampling: {e}"))?;
        out.extend_from_slice(&produced[0]);
        pos += take;
    }

    // Trim any extra tail from zero-padding the final chunk.
    let target_len = (input.len() as u64 * WHISPER_SAMPLE_RATE as u64 / input_rate as u64) as usize;
    if out.len() > target_len {
        out.truncate(target_len);
    }
    Ok(out)
}

/// Extract a likely file-extension hint from a multipart filename like
/// `"audio.mp3"` or `"recording.wav"`. Returns `None` for bare names.
pub fn extension_hint(filename: &str) -> Option<&str> {
    filename.rsplit('.').next().filter(|ext| *ext != filename)
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
