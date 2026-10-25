//! The speculative token stream of one single-sequence request.
//!
//! - `route`: the per-request plan and the target each plan runs over
//! - `drive`: the speculative loop and the stats it records
//! - `model_drafter`: the drafter over the draft model

mod drive;
mod model_drafter;
mod route;
#[cfg(test)]
mod test_rig;
