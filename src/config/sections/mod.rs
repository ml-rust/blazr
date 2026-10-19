//! Config sections: `server`, `inference`, `generation`, decision, and vision.

pub(super) mod constraint;
pub(super) mod decision;
pub(super) mod generation;
pub(super) mod inference;
pub(super) mod overlay;
pub(super) mod server;
pub(super) mod speculative;
pub(super) mod speculative_drafter;
pub(super) mod vision;

pub use constraint::{CompiledConstraint, Constraint};
pub use decision::{DecisionConfig, DecisionFamily};
pub use generation::GenerationConfig;
pub use inference::InferenceConfig;
pub use overlay::InferenceOverlay;
pub use server::{LatencySlo, ReasoningFormat, ServerConfig};
pub use speculative::SpeculativeDecodingConfig;
pub use speculative_drafter::{DrafterKind, DrafterSpec, PromptLookupConfig};
pub use vision::VisionOverrides;
