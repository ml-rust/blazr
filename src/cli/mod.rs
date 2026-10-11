//! CLI commands
//!
//! Provides ollama-like CLI interface for blazr.

mod args;
mod bench;
mod chat;
mod commands;
mod convert;
#[cfg(feature = "distributed")]
mod disaggregated;
#[cfg(feature = "distributed")]
mod disaggregated_forward;
mod info;
mod list;
mod load_settings;
#[cfg(feature = "audio")]
mod narration;
mod ps;
mod pull;
mod run;
mod run_repl;
mod serve;
mod serve_batching;
mod serve_replicas;
#[cfg(feature = "distributed")]
mod swarm;
#[cfg(feature = "distributed")]
mod swarm_forward;
mod util;

pub use args::{RuntimeArgs, SamplingArgs, VisionArgs};
pub use bench::bench;
pub use chat::chat;
pub use commands::{Cli, Commands};
pub use convert::convert;
#[cfg(feature = "distributed")]
pub use disaggregated::{run_disagg_decode_worker, run_disagg_prefill_worker, run_disagg_router};
pub use info::info;
pub use list::list;
pub use load_settings::{build_cuda_index, LoadSettings, RunSettings};
#[cfg(feature = "audio")]
pub use narration::{sound_review, speak, transcribe, SoundReviewArgs, SpeakArgs, TranscribeArgs};
pub use ps::ps;
pub use pull::pull;
pub use run::run;
pub use serve::serve;
#[cfg(feature = "distributed")]
pub use swarm::swarm;
