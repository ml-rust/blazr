//! Speculative decoding over boostr's generator: the target adapter, the
//! sampling-rule mapping and the per-request eligibility check.

pub(in crate::engine) mod eligibility;
pub(in crate::engine) mod rule;
pub(in crate::engine) mod target;

pub(in crate::engine) use eligibility::{Fallback, SpeculativePlan};
pub(in crate::engine) use rule::sample_rule;
pub(in crate::engine) use target::ContiguousTarget;
