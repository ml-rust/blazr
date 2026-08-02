//! Whisper transcription orchestration.
//!
//! Pipeline: 16 kHz mono f32 samples → pad/trim to 30 s →
//! mel spectrogram → encoder → greedy decode with KV cache → detokenize.

use anyhow::{anyhow, Result};

use boostr::model::audio::{mel::compute_mel_spectrogram, GenerateOptions, WhisperBundle};
use boostr::{Runtime, Tensor};

use super::handlers::ServerRuntime;

/// Whisper operates on 30-second audio windows at 16 kHz.
pub const WHISPER_CHUNK_SAMPLES: usize = 30 * 16_000;
/// Mel frames per 30 s window (hop_size = 160).
pub const WHISPER_CHUNK_FRAMES: usize = 3_000;

/// One transcribed chunk with its source time range (for segmentation output).
pub struct ChunkTranscript {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Transcribe any-length audio by splitting into non-overlapping 30-second
/// windows and concatenating results. Each chunk is processed independently
/// (no context carryover yet — that's a prompt-tokens improvement for later).
pub fn transcribe_chunks(
    bundle: &WhisperBundle<ServerRuntime>,
    samples: &[f32],
    language: Option<&str>,
    translate: bool,
    max_new_tokens: usize,
) -> Result<Vec<ChunkTranscript>> {
    if samples.is_empty() {
        return Err(anyhow!("no audio samples to transcribe"));
    }
    let total = samples.len();
    let mut out = Vec::with_capacity(total.div_ceil(WHISPER_CHUNK_SAMPLES));
    let mut pos = 0usize;
    while pos < total {
        let end = (pos + WHISPER_CHUNK_SAMPLES).min(total);
        let text = transcribe_once(
            bundle,
            &samples[pos..end],
            language,
            translate,
            max_new_tokens,
        )?;
        out.push(ChunkTranscript {
            start: pos as f64 / 16_000.0,
            end: end as f64 / 16_000.0,
            text,
        });
        pos = end;
    }
    Ok(out)
}

/// Transcribe up to 30 s of audio through a loaded [`WhisperBundle`].
///
/// - `samples`: 16 kHz mono f32 in `[-1, 1]`. Shorter than 30 s is zero-padded;
///   longer is truncated (multi-chunk transcription is a follow-up).
/// - `language`: BCP-47 code like `"en"`; `None` forces the default (no lang token).
/// - `translate`: `true` emits `<|translate|>` (to English) instead of `<|transcribe|>`.
/// - `max_new_tokens`: decoder step budget. `448 - prefix_len` is the safe ceiling.
pub fn transcribe_once(
    bundle: &WhisperBundle<ServerRuntime>,
    samples: &[f32],
    language: Option<&str>,
    translate: bool,
    max_new_tokens: usize,
) -> Result<String> {
    // 1. Pad/trim to exactly 30 s of samples.
    let mut chunk = vec![0.0f32; WHISPER_CHUNK_SAMPLES];
    let take = samples.len().min(WHISPER_CHUNK_SAMPLES);
    chunk[..take].copy_from_slice(&samples[..take]);

    // 2. Mel spectrogram. Output layout: `[num_mel_bins, num_frames]`.
    let mel = compute_mel_spectrogram(&chunk, bundle.num_mel_bins, 16_000);
    let num_frames = mel.len() / bundle.num_mel_bins;
    if num_frames == 0 {
        return Err(anyhow!("mel spectrogram produced zero frames"));
    }

    // Trim/pad the mel grid to exactly WHISPER_CHUNK_FRAMES so the encoder's
    // positional embedding lines up with what the checkpoint was trained on.
    let mel = fit_mel_frames(&mel, bundle.num_mel_bins, num_frames, WHISPER_CHUNK_FRAMES);

    // 3. Build tensor `[1, num_mel_bins, num_frames]` on the runtime's device.
    let device = <ServerRuntime as Runtime>::default_device();
    let mel_tensor = Tensor::<ServerRuntime>::from_slice(
        &mel,
        &[1, bundle.num_mel_bins, WHISPER_CHUNK_FRAMES],
        &device,
    );

    // 4. Encoder forward.
    let client = <ServerRuntime as Runtime>::default_client(&device);
    let encoder_out = bundle
        .model
        .encode(&client, &mel_tensor)
        .map_err(|e| anyhow!("whisper encode: {e}"))?;

    // 5. Greedy decode with KV cache.
    let prompt = bundle.sot_prompt(language, translate);
    let prefix_budget = prompt.len();
    let options = GenerateOptions {
        max_new_tokens: max_new_tokens.min(bundle.config.max_target_positions - prefix_budget),
        eos_token_ids: vec![bundle.variant.eos_token_id()],
        suppress_tokens: Vec::new(),
    };
    let token_ids = bundle
        .model
        .generate(&client, &encoder_out, &prompt, &options)
        .map_err(|e| anyhow!("whisper generate: {e}"))?;

    // 6. Decode tokens. Drop any residual timestamp/control tokens the decoder
    // might emit even with <|notimestamps|> in the prefix — keep only ids below
    // the first timestamp sentinel.
    let first_ts = bundle.variant.first_timestamp_token_id();
    let filtered: Vec<u32> = token_ids
        .into_iter()
        .filter(|&t| t < first_ts && t != bundle.variant.eos_token_id())
        .collect();

    let text = bundle
        .tokenizer
        .decode(&filtered)
        .map_err(|e| anyhow!("whisper decode: {e}"))?;
    Ok(text.trim().to_string())
}

/// Pad or truncate a `[num_mel_bins, src_frames]` row-major mel matrix so it
/// has exactly `target_frames` time steps. Padding is log-domain silence
/// (`ln(1e-10)` = the epsilon floor used in [`compute_mel_spectrogram`]).
fn fit_mel_frames(
    mel: &[f32],
    num_mel_bins: usize,
    src_frames: usize,
    target_frames: usize,
) -> Vec<f32> {
    if src_frames == target_frames {
        return mel.to_vec();
    }
    let pad_value = (1e-10f32).ln();
    let mut out = vec![pad_value; num_mel_bins * target_frames];
    let copy_frames = src_frames.min(target_frames);
    for m in 0..num_mel_bins {
        let src_row = &mel[m * src_frames..m * src_frames + copy_frames];
        let dst_row = &mut out[m * target_frames..m * target_frames + copy_frames];
        dst_row.copy_from_slice(src_row);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_mel_pads_shorter() {
        // 2 mel bins × 3 src frames → 5 target frames
        let mel = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = fit_mel_frames(&mel, 2, 3, 5);
        assert_eq!(out.len(), 10);
        // First 3 of each bin preserved
        assert_eq!(&out[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&out[5..8], &[4.0, 5.0, 6.0]);
        // Padding is ln(1e-10) ≈ -23.03
        let pad = (1e-10f32).ln();
        assert!((out[3] - pad).abs() < 1e-4);
        assert!((out[9] - pad).abs() < 1e-4);
    }

    #[test]
    fn fit_mel_truncates_longer() {
        let mel = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = fit_mel_frames(&mel, 2, 3, 2);
        assert_eq!(out.len(), 4);
        assert_eq!(out, vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn fit_mel_identity() {
        let mel = vec![1.0, 2.0, 3.0, 4.0];
        let out = fit_mel_frames(&mel, 2, 2, 2);
        assert_eq!(out, mel);
    }
}
