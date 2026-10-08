//! `blazr speak` and `blazr transcribe`: the narration skill's commands.

mod args;
mod config;
mod download;
mod engines;
mod gpu;
mod hearing;
mod kinds;
mod models;
mod paths;
mod print;
mod speak;
mod transcribe;
mod tts_section;

pub use args::{SpeakArgs, TranscribeArgs};
pub use speak::speak;
pub use transcribe::transcribe;
