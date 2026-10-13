//! Config sections: `server`, `inference`, `generation`, decision, and vision.

pub(super) mod decision;
pub(super) mod generation;
pub(super) mod inference;
pub(super) mod overlay;
pub(super) mod server;
pub(super) mod vision;

pub use decision::{DecisionConfig, DecisionFamily};
pub use generation::GenerationConfig;
pub use inference::InferenceConfig;
pub use overlay::InferenceOverlay;
pub use server::{LatencySlo, ReasoningFormat, ServerConfig};
pub use vision::VisionOverrides;
