//! Audio endpoints: TTS (speech), ASR (transcriptions), and voice listing.
//!
//! OpenAI-compatible `/v1/audio/speech` and `/v1/audio/transcriptions`
//! endpoints, served by the `TtsBundle`/`WhisperBundle` registered per model.
//! `GET /v1/audio/voices?model=<name>` lists a registered TTS model's voice
//! catalog (`bundle.voices()`), 400 when `model` is missing or unknown.
//!
//! `/v1/audio/transcriptions` accepts multipart `timestamp_granularities[]`
//! (`word`, `segment`, or both), valid only with `response_format=verbose_json`
//! (400 otherwise). `word` adds a `words: [{word, start, end}]` array to the
//! response and is a 400 against a checkpoint with no `alignment_heads`.
//!
//! `voice` on `/v1/audio/speech` is optional and its absence is meaningful: an
//! omitted `voice` renders zero-shot on a VoxCPM2 bundle (or the previous
//! `alloy` default on Kokoro, unchanged), while an explicit id the bundle's
//! catalog does not carry is a 400 error by design — including a client's
//! habitual `"alloy"` sent against a VoxCPM2 bundle. The error surfaces that
//! mismatch; a silent close-enough voice hides it.
//!
//! `stream` on `/v1/audio/speech` (default `false`) sends the body as the
//! engine renders it: for `wav`/`f32` a header with unknown-length size
//! fields then raw sample chunks, for `pcm` the chunks alone. The bytes after
//! the header are the same the buffered response carries.

mod encode;
mod request;
mod speech;
mod stream;
mod transcription;
mod types;
mod voice;
mod voices;

pub use speech::speech;
pub use transcription::{transcriptions, translations};
pub use voices::voices;
