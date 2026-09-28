//! `POST /v1/audio/transcriptions` word timestamps, in-process against the
//! real router (no live server) with a real `whisper-tiny` checkpoint.
//!
//! Fixtures (all under `$BOOSTR_MODELS_DIR`, each overridable by its own var,
//! see `tests/common/mod.rs`):
//! * `WHISPER_TINY_DIR` → `whisper-tiny`, else the HF hub cache layout
//!   (`~/.cache/huggingface/hub/models--openai--whisper-tiny/snapshots/<rev>`)
//! * `BLAZR_ASR_TEST_WAV` → `librispeech_en_sample.wav`
//!
//! Both fixture-gated tests skip with a printed notice when either is
//! absent. The `response_format` validation test needs no model or audio and
//! always runs.
#![cfg(feature = "audio")]

mod common;

use common::{
    asr_app_state, asr_test_wav, audio_router, model_fixture, multipart_body, post_multipart_json,
    skip_notice, whisper_tiny_dir,
};

const MODEL_NAME: &str = "whisper-tiny";
const BOUNDARY: &str = "BlazrAudioTestBoundary7MA4YWxkTrZu0gW";

/// Everything the fixture-gated tests need, or `None` when either fixture is
/// missing.
struct Fixtures {
    checkpoint_dir: std::path::PathBuf,
    wav_bytes: Vec<u8>,
    /// The clip's real duration, seconds — decoded independently of the
    /// server's own resample path, so it bounds word timestamps without
    /// assuming anything about the pipeline under test.
    duration_secs: f64,
}

fn fixtures() -> Option<Fixtures> {
    let checkpoint_dir = whisper_tiny_dir()?;
    let wav_path = asr_test_wav()?;
    let wav_bytes =
        std::fs::read(&wav_path).unwrap_or_else(|e| panic!("reading {}: {e}", wav_path.display()));
    let wav = voxr::decode_wav(&wav_bytes)
        .unwrap_or_else(|e| panic!("decoding {}: {e}", wav_path.display()));
    let duration_secs = wav.frames() as f64 / wav.sample_rate as f64;
    Some(Fixtures {
        checkpoint_dir,
        wav_bytes,
        duration_secs,
    })
}

fn skip_if_missing() -> Option<Fixtures> {
    let fx = fixtures();
    if fx.is_none() {
        // Named separately so a reader knows which of the two is missing.
        if whisper_tiny_dir().is_none() {
            skip_notice("whisper-tiny checkpoint", "WHISPER_TINY_DIR");
        }
        if model_fixture("BLAZR_ASR_TEST_WAV", "librispeech_en_sample.wav").is_none() {
            skip_notice("ASR test clip", "BLAZR_ASR_TEST_WAV");
        }
    }
    fx
}

fn file_part(bytes: &[u8]) -> (&str, &str, &str, &[u8]) {
    ("file", "clip.wav", "audio/wav", bytes)
}

/// `timestamp_granularities[]=word` with `verbose_json`: `words` present, in
/// order, each span monotonic and non-overlapping, all inside
/// `[0, clip duration]`; `segments` absent (word-only drops it).
#[tokio::test]
async fn word_granularity_returns_words_without_segments() {
    let Some(fx) = skip_if_missing() else {
        return;
    };
    let state = asr_app_state(Some((MODEL_NAME, &fx.checkpoint_dir))).await;
    let router = audio_router(state);

    let body = multipart_body(
        BOUNDARY,
        &[
            ("model", MODEL_NAME),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
        ],
        file_part(&fx.wav_bytes),
    );
    let (status, json) =
        post_multipart_json(router, "/v1/audio/transcriptions", BOUNDARY, body).await;

    assert_eq!(status, 200, "response body: {json}");
    assert!(json.get("segments").is_none(), "response body: {json}");
    let words = json["words"]
        .as_array()
        .unwrap_or_else(|| panic!("no `words` array in {json}"));
    assert!(!words.is_empty(), "transcript produced no words: {json}");

    let tolerance = 0.5; // resample/window rounding at the clip's edges
    let mut prev_end = 0.0f64;
    for (i, w) in words.iter().enumerate() {
        let word = w["word"]
            .as_str()
            .unwrap_or_else(|| panic!("word {i} carries no `word` string: {w}"));
        let start = w["start"]
            .as_f64()
            .unwrap_or_else(|| panic!("word {i} ({word:?}) carries no numeric `start`: {w}"));
        let end = w["end"]
            .as_f64()
            .unwrap_or_else(|| panic!("word {i} ({word:?}) carries no numeric `end`: {w}"));
        assert!(
            start <= end,
            "word {i} ({word:?}): start {start} > end {end}"
        );
        assert!(
            start >= -tolerance && end <= fx.duration_secs + tolerance,
            "word {i} ({word:?}): span ({start}, {end}) outside clip duration {}",
            fx.duration_secs
        );
        assert!(
            start + tolerance >= prev_end,
            "word {i} ({word:?}): starts at {start}, before the previous word's end {prev_end}"
        );
        prev_end = end;
    }
}

/// `timestamp_granularities[]=word` AND `=segment`: both arrays present.
#[tokio::test]
async fn both_granularities_return_words_and_segments() {
    let Some(fx) = skip_if_missing() else {
        return;
    };
    let state = asr_app_state(Some((MODEL_NAME, &fx.checkpoint_dir))).await;
    let router = audio_router(state);

    let body = multipart_body(
        BOUNDARY,
        &[
            ("model", MODEL_NAME),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ],
        file_part(&fx.wav_bytes),
    );
    let (status, json) =
        post_multipart_json(router, "/v1/audio/transcriptions", BOUNDARY, body).await;

    assert_eq!(status, 200, "response body: {json}");
    assert!(
        json["words"].as_array().is_some_and(|a| !a.is_empty()),
        "expected a non-empty `words` array: {json}"
    );
    assert!(
        json["segments"].as_array().is_some_and(|a| !a.is_empty()),
        "expected a non-empty `segments` array: {json}"
    );
}

/// `timestamp_granularities[]=word` against `response_format=json` (not
/// `verbose_json`) is a 400 naming the field. Needs no model or audio fixture
/// — the handler rejects this before ever looking up the ASR bundle — so
/// this always runs.
#[tokio::test]
async fn word_granularity_without_verbose_json_is_400() {
    let state = asr_app_state(None).await;
    let router = audio_router(state);

    let body = multipart_body(
        BOUNDARY,
        &[
            ("model", MODEL_NAME),
            ("response_format", "json"),
            ("timestamp_granularities[]", "word"),
        ],
        file_part(b"not real audio, never decoded: rejected before the file is read"),
    );
    let (status, json) =
        post_multipart_json(router, "/v1/audio/transcriptions", BOUNDARY, body).await;

    assert_eq!(status, 400, "response body: {json}");
    let message = json["error"]["message"]
        .as_str()
        .unwrap_or_else(|| panic!("no `error.message` string in {json}"));
    assert!(
        message.contains("timestamp_granularities"),
        "error must name the field, got: {message}"
    );
    assert!(
        message.contains("verbose_json"),
        "error must tell the caller to use verbose_json, got: {message}"
    );
}
