//! Audio endpoints: TTS (speech), ASR (transcriptions), and voice listing.
//!
//! OpenAI-compatible `/v1/audio/speech` and `/v1/audio/transcriptions`
//! endpoints, served by the `TtsBundle`/`WhisperBundle` registered per model.

mod encode;
mod request;
mod speech;
mod stream;
mod transcription;
mod transcription_form;
#[cfg(test)]
mod transcription_tests;
mod types;
mod voice;
mod voices;

pub use speech::speech;
pub use transcription::{transcriptions, translations};
pub use voices::voices;
