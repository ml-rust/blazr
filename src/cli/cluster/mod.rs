//! Multi-node commands: `swarm` and `disagg`.

pub(super) mod disaggregated;
pub(super) mod disaggregated_forward;
pub(super) mod disaggregated_layout;
pub(super) mod disaggregated_mesh;
pub(super) mod disaggregated_params;
pub(super) mod disaggregated_router;
pub(super) mod disaggregated_scheduler;
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod disaggregated_test_rig;
pub(super) mod mesh;
pub(super) mod swarm;
pub(super) mod swarm_request;
pub(super) mod swarm_scheduler;
pub(super) mod swarm_serving;
pub(super) mod swarm_stage;
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod swarm_test_rig;

pub use disaggregated::run_disagg;
pub use swarm::swarm;
