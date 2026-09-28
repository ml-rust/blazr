//! TTS model loader.
//!
//! Parses `--tts-model NAME=PATH` and builds a [`voxr::TtsBundle`].
//! `dispatch` picks the engine from the directory layout, `kokoro` and
//! `voxcpm` build each engine, `args` parses the CLI argument.

mod args;
mod dispatch;
mod kokoro;
mod voxcpm;

pub use args::parse_tts_model_arg;
pub use dispatch::load_tts_from_dir;
