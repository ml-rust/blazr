//! System One decision models on the engine: the service and worker thread
//! of each loaded decision model, its readout head, request preparation
//! and limits, Kev models over their base, the scheduler's decision entry
//! points, and temperature calibration.

pub mod calibration;
pub mod error;
pub mod kev_base;
mod kev_model;
pub mod limits;
mod prefix_cache;
pub mod prepare;
pub mod readout;
pub mod registry;
pub mod service;
pub mod stats;
mod worker;

pub use error::DecisionFailure;
pub use kev_base::{resolve_kev_base, KevBase};
pub use limits::DecisionLimits;
pub use prepare::{prepare, unique_tokens, PrefixRule, PreparedRequest};
pub use readout::{ReadoutHead, ReadoutItem};
pub use registry::DecisionRegistry;
pub use service::{Answered, DecisionService, Pending};
pub use stats::{DecisionCounts, DecisionStats, PrefixCacheReport};
