//! Engine telemetry: metric names, recording helpers and latency SLO checks.
//!
//! The engine records through the `metrics` facade. With no recorder
//! installed every call is a no-op. The `server` feature installs the
//! Prometheus recorder and serves it on `/metrics`.

pub mod decision;
pub mod names;
pub mod record;
pub mod slo;
#[cfg(test)]
pub(crate) mod test_recorder;
