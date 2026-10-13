//! PEFT LoRA adapters as model variants: each adapter applies to a loaded
//! base model as the variant `<model>:<name>`, which shares the base
//! weights and has its own executor.

pub mod derive;
pub mod error;
pub(crate) mod eviction;
pub mod registry;
pub mod select;
pub mod spec;
mod variants;

pub use derive::{derive_run_model, run_adapter, DerivedVariant};
pub use error::LoraError;
pub use registry::{LoraRegistry, VariantInfo};
pub use select::select_model;
pub use spec::{configured_adapters, split_variant, variant_id, AdapterSpec};
