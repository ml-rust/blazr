//! `blazr speak` and `blazr transcribe`: the narration skill's commands.

mod args;
mod config;
mod engines;
mod gpu;
mod hearing;
mod paths;
mod print;
mod speak;
mod transcribe;
mod tts_section;
mod weights;

pub use args::{SpeakArgs, TranscribeArgs};
pub use speak::speak;
pub use transcribe::transcribe;
