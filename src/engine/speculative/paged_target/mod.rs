//! The speculative target over an executor's paged KV pool.
//!
//! - `target`: the [`PagedTarget`] type, its prompt lease and its forward
//! - `model`: the boostr speculative traits over it

mod model;
mod target;
#[cfg(test)]
mod test_rig;

pub(in crate::engine) use target::PagedTarget;
