//! The command line: argument structs, the command enum, and load settings.

pub(super) mod args;
#[cfg(all(feature = "distributed", feature = "server"))]
pub(super) mod cluster_args;
pub(super) mod commands;
pub(super) mod decision_args;
pub(super) mod load_settings;
pub(super) mod version;

pub use args::{DtypeArgs, LoraArgs, RuntimeArgs, SamplingArgs, VisionArgs};
#[cfg(all(feature = "distributed", feature = "server"))]
pub use cluster_args::{DisaggArgs, DisaggRoleArg, MeshArgs, SwarmArgs, SwarmRoleArg};
pub use commands::{Cli, Commands};
pub use decision_args::DecisionArgs;
pub use load_settings::{build_gpu_index, LoadSettings, RunSettings};
