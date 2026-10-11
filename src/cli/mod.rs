//! CLI commands
//!
//! Provides ollama-like CLI interface for blazr.
//!
//! `serve` and `ps` need the `server` feature. `swarm` and `disagg` need
//! `distributed` and `server`. The narration commands need `audio`. Every
//! other command needs only the engine and the hub client.

mod args;
mod bench;
mod chat;
mod commands;
mod convert;
#[cfg(all(feature = "distributed", feature = "server"))]
mod disaggregated;
#[cfg(all(feature = "distributed", feature = "server"))]
mod disaggregated_forward;
mod info;
mod list;
mod load_settings;
#[cfg(feature = "audio")]
mod narration;
#[cfg(feature = "server")]
mod ps;
mod pull;
mod run;
mod run_repl;
#[cfg(feature = "server")]
mod serve;
#[cfg(feature = "server")]
mod serve_batching;
#[cfg(feature = "server")]
mod serve_replicas;
#[cfg(all(feature = "distributed", feature = "server"))]
mod swarm;
#[cfg(all(feature = "distributed", feature = "server"))]
mod swarm_forward;
mod util;

pub use args::{DtypeArgs, RuntimeArgs, SamplingArgs, VisionArgs};
pub use bench::bench;
pub use chat::chat;
pub use commands::{Cli, Commands};
pub use convert::convert;
#[cfg(all(feature = "distributed", feature = "server"))]
pub use disaggregated::{run_disagg_decode_worker, run_disagg_prefill_worker, run_disagg_router};
pub use info::info;
pub use list::list;
pub use load_settings::{build_gpu_index, LoadSettings, RunSettings};
#[cfg(feature = "audio")]
pub use narration::{sound_review, speak, transcribe, SoundReviewArgs, SpeakArgs, TranscribeArgs};
#[cfg(feature = "server")]
pub use ps::ps;
pub use pull::pull;
pub use run::run;
#[cfg(feature = "server")]
pub use serve::serve;
#[cfg(all(feature = "distributed", feature = "server"))]
pub use swarm::swarm;
