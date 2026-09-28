//! Whisper transcription orchestration.
//!
//! blazr holds no core logic: the mel front end, encoder, greedy decode and
//! language detection all live in [`WhisperBundle::transcribe`]. This module
//! only cuts arbitrary-length audio into the 30 s windows that call accepts
//! and stitches the per-window results back together with their time ranges.

use anyhow::{anyhow, Result};

use boostr::Runtime;
use voxr::{TranscribeOptions, WhisperBundle, WordTiming};

use super::handlers::ServerRuntime;

/// Sample rate the audio must arrive at.
pub const WHISPER_SAMPLE_RATE: usize = 16_000;
/// Whisper's encoder window: 30 s at 16 kHz.
pub const WHISPER_CHUNK_SAMPLES: usize = 30 * WHISPER_SAMPLE_RATE;

/// One transcribed chunk with its source time range (for segmentation output).
pub struct ChunkTranscript {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// Language the chunk was decoded under: the request's, or the detected
    /// one. `None` only for an english-only checkpoint.
    pub language: Option<&'static str>,
    /// Word timings from this chunk, already offset onto the full input's
    /// clock. Empty when word timestamps were not requested.
    pub words: Vec<WordTiming>,
}

/// Transcribe any-length audio by splitting into non-overlapping 30-second
/// windows and concatenating results. Each chunk is processed independently
/// (no context carryover yet — that's a prompt-tokens improvement for later).
///
/// - `samples`: 16 kHz mono f32 in `[-1, 1]`.
/// - `language`: BCP-47 code like `"en"`; `None` detects it per chunk.
/// - `translate`: `true` emits `<|translate|>` (to English) instead of `<|transcribe|>`.
/// - `max_new_tokens`: decoder step budget per chunk, capped by the checkpoint's own.
/// - `want_words`: runs [`WhisperBundle::transcribe_words`] per chunk instead
///   of the plain decode, and offsets each returned word by the chunk's start
///   so every timestamp lands on the input's clock, not the chunk's.
pub fn transcribe_chunks(
    bundle: &WhisperBundle<ServerRuntime>,
    samples: &[f32],
    language: Option<&str>,
    translate: bool,
    max_new_tokens: usize,
    want_words: bool,
) -> Result<Vec<ChunkTranscript>> {
    if samples.is_empty() {
        return Err(anyhow!("no audio samples to transcribe"));
    }
    let device = <ServerRuntime as Runtime>::default_device();
    let client = <ServerRuntime as Runtime>::default_client(&device);
    let budget = bundle.generate_options().max_new_tokens.min(max_new_tokens);
    let opts = TranscribeOptions {
        language,
        translate,
        max_new_tokens: Some(budget),
    };

    let total = samples.len();
    let mut out = Vec::with_capacity(total.div_ceil(WHISPER_CHUNK_SAMPLES));
    for (index, window) in samples.chunks(WHISPER_CHUNK_SAMPLES).enumerate() {
        let start = index * WHISPER_CHUNK_SAMPLES;
        let end = start + window.len();
        let chunk_start_secs = start as f64 / WHISPER_SAMPLE_RATE as f64;

        let (text, language_used, words) = if want_words {
            let (result, words) = bundle
                .transcribe_words(&client, window, WHISPER_SAMPLE_RATE, &opts)
                .map_err(|e| anyhow!("whisper transcribe_words (window {index}): {e}"))?;
            let offset_words = words
                .into_iter()
                .map(|w| WordTiming {
                    text: w.text,
                    start: w.start + chunk_start_secs,
                    end: w.end + chunk_start_secs,
                })
                .collect();
            (result.text, result.language, offset_words)
        } else {
            let result = bundle
                .transcribe(&client, window, WHISPER_SAMPLE_RATE, &opts)
                .map_err(|e| anyhow!("whisper transcribe (window {index}): {e}"))?;
            (result.text, result.language, Vec::new())
        };

        out.push(ChunkTranscript {
            start: chunk_start_secs,
            end: end as f64 / WHISPER_SAMPLE_RATE as f64,
            text: text.trim().to_string(),
            language: language_used,
            words,
        });
    }
    Ok(out)
}
