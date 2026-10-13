//! HTTP request handlers
//!
//! Core handlers: health, models, tokenize/detokenize, slots. Auxiliary
//! (vision/ASR/TTS) model bookkeeping lives in `auxiliary`.

mod auxiliary;
mod health;
mod models;
mod route;
mod slots;
mod state;
mod template;
mod tokenize;
mod variant;

pub use auxiliary::{AuxiliaryKind, AuxiliaryModel};
pub use health::health;
pub use models::{get_model, list_models, ModelInfo, ModelsResponse};
pub use route::{Route, ServeReplica, Serving};
pub use slots::{create_slot, delete_slot, list_slots};
#[cfg(feature = "audio")]
pub use state::AsrBundle;
#[cfg(feature = "audio")]
pub use state::TtsBundle;
pub use state::{AppState, ServerDevice, ServerRuntime, VisionEmbedder};
pub use template::apply_template;
pub use tokenize::{detokenize, tokenize};
pub use variant::{
    classify_lookup, lookup_error_response, lora_error_response, select_executor, LookupFailure,
};
