//! Speculative decoding over boostr's generator: the target adapters, the
//! sampling-rule mapping, the per-request eligibility check, the token
//! stream and the load-time report.

pub(in crate::engine) mod decode;
pub(in crate::engine) mod draft_load;
pub(in crate::engine) mod eligibility;
pub(in crate::engine) mod paged_target;
pub(crate) mod rule;
pub(in crate::engine) mod startup;
pub(in crate::engine) mod target;
mod target_glue;

pub(in crate::engine) use eligibility::{Fallback, SpeculativePlan};
pub(in crate::engine) use paged_target::PagedTarget;
pub(in crate::engine) use rule::sample_rule;
pub(in crate::engine) use startup::report_speculative;
pub(in crate::engine) use target::ContiguousTarget;
