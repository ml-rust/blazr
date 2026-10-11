//! HTTP request handlers
//!
//! Core handlers: health, models, tokenize/detokenize, slots. Auxiliary
//! (vision/ASR/TTS) model bookkeeping lives in `auxiliary`. Completion and
//! chat handlers are in separate top-level server modules.

mod auxiliary;
mod health;
mod models;
mod slots;
mod state;
mod template;
mod tokenize;

pub use auxiliary::{AuxiliaryKind, AuxiliaryModel};
pub use health::health;
pub use models::{get_model, list_models, ModelInfo, ModelsResponse};
pub use slots::{create_slot, delete_slot, list_slots};
#[cfg(feature = "audio")]
pub use state::AsrBundle;
#[cfg(feature = "audio")]
pub use state::TtsBundle;
pub use state::{AppState, ServerDevice, ServerRuntime, VisionEmbedder};
pub use template::apply_template;
pub use tokenize::{detokenize, tokenize};
