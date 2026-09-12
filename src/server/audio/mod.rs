//! Audio endpoints: TTS (speech) and ASR (transcriptions)
//!
//! OpenAI-compatible `/v1/audio/speech` and `/v1/audio/transcriptions`
//! endpoints, served by the `TtsBundle`/`WhisperBundle` registered per model.
//!
//! `voice` on `/v1/audio/speech` is optional and its absence is meaningful: an
//! omitted `voice` renders zero-shot on a VoxCPM2 bundle (or the previous
//! `alloy` default on Kokoro, unchanged), while an explicit id the bundle's
//! catalog does not carry is a 400 error by design — including a client's
//! habitual `"alloy"` sent against a VoxCPM2 bundle. Silently substituting a
//! close-enough voice would hide that mismatch instead of surfacing it.

mod encode;
mod request;
mod speech;
mod transcription;
mod types;
mod voice;

pub use speech::speech;
pub use transcription::{transcriptions, translations};
