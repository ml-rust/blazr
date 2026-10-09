//! `blazr speak`, `blazr transcribe` and `blazr sound-review`: the
//! motion-video skill's commands.

mod args;
mod config;
mod engines;
mod gpu;
mod hearing;
mod paths;
mod print;
mod sound_review;
mod speak;
mod transcribe;
mod tts_section;
mod weights;

pub use args::{SoundReviewArgs, SpeakArgs, TranscribeArgs};
pub use sound_review::sound_review;
pub use speak::speak;
pub use transcribe::transcribe;
