//! Configuration system for blazr
//!
//! `~/.blazr/config.yaml` holds the user keys and the `server`,
//! `inference` and `generation` sections. `resolve` applies CLI flags and
//! environment variables over it. A model's `config.json` holds the model
//! and its per-model `inference` section.

mod blazr;
mod defaults;
mod device;
mod dtype;
mod env;
mod file;
mod generation;
mod inference;
mod live;
mod overlay;
pub mod reload;
pub mod resolve;
mod resolve_cluster;
mod resolve_serve;
mod server;
mod vision;

pub use blazr::{parse_dtype, BlazrConfig};
pub use defaults::{DEFAULT_MODEL_DIR, DEFAULT_NUM_CTX};
pub use device::DeviceChoice;
pub use dtype::{
    compute_dtype, dtype_name, fixed_format_dtype, parse_dtype_for, torch_dtype, DtypeRequest,
    DtypeSource, BACKEND_DEFAULT_DTYPE,
};
pub use env::EnvVars;
pub use file::{FileConfig, ParsedFile};
pub use generation::GenerationConfig;
pub use inference::InferenceConfig;
pub use live::LiveConfig;
pub use overlay::InferenceOverlay;
pub use resolve::GenerationFlags;
pub use resolve_cluster::{resolve_cluster, ClusterFlags, ClusterSettings};
pub use resolve_serve::{
    check_data_parallel, check_tls_pair, resolve_serve, tp_conflict, ServeFlags, ServeSettings,
};
pub use server::{LatencySlo, ReasoningFormat, ServerConfig};
pub use vision::VisionOverrides;
