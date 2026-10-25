//! HTTP request handlers: health, models, tokenize, slots, and auxiliary model bookkeeping.

mod auxiliary;
mod budget;
mod health;
mod models;
mod route;
mod slots;
mod state;
mod template;
mod tokenize;
mod variant;

pub use auxiliary::{AuxiliaryKind, AuxiliaryModel};
pub use budget::{request_tokens, BudgetPermit, DecodeSlot, TokenBudget};
pub use health::health;
pub use models::{get_model, list_models, ModelInfo, ModelsResponse};
pub use route::{Route, ServeReplica, Serving};
pub use slots::{create_slot, delete_slot, list_slots};
#[cfg(test)]
pub(crate) use state::test_state;
pub use state::{AppState, ServerDevice, ServerRuntime, VisionEmbedder};
#[cfg(feature = "audio")]
pub use state::{AsrBundle, TtsBundle};
pub use template::apply_template;
pub use tokenize::{detokenize, tokenize};
pub use variant::{
    classify_lookup, lookup_error_response, lora_error_response, select_executor, LookupFailure,
};
