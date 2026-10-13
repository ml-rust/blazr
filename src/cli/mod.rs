//! CLI commands
//!
//! Provides ollama-like CLI interface for blazr.
//!
//! `serve` and `ps` need the `server` feature. `swarm` and `disagg` need
//! `distributed` and `server`. The narration commands need `audio`. Every
//! other command needs only the engine and the hub client.

mod catalog;
#[cfg(all(feature = "distributed", feature = "server"))]
mod cluster;
mod cmdline;
mod decision;
mod local;
#[cfg(feature = "audio")]
mod narration;
#[cfg(feature = "server")]
mod serving;
mod util;

pub use catalog::*;
#[cfg(all(feature = "distributed", feature = "server"))]
pub use cluster::*;
pub use cmdline::*;
pub use decision::{decision, CalibrateArgs, DecisionCommand};
pub use local::*;
#[cfg(feature = "audio")]
pub use narration::{sound_review, speak, transcribe, SoundReviewArgs, SpeakArgs, TranscribeArgs};
#[cfg(feature = "server")]
pub use serving::*;
