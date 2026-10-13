//! Value precedence: CLI flags, then environment variables, then the user file.

pub(super) mod decision_flags;
pub(super) mod env;
pub mod resolve;
pub(super) mod resolve_cluster;
pub(super) mod resolve_serve;

pub use decision_flags::{resolve_decision, DecisionFlags};
pub use env::EnvVars;
pub use resolve::GenerationFlags;
pub use resolve_cluster::{resolve_cluster, ClusterFlags, ClusterSettings};
pub use resolve_serve::{
    check_data_parallel, check_tls_pair, resolve_serve, tp_conflict, ServeFlags, ServeSettings,
};
