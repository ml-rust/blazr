//! Model loading utilities
//!
//! This module provides unified model loading from various formats:
//! - SafeTensors (HuggingFace standard, including AWQ quantized)
//! - GGUF (llama.cpp format, for quantized models)
//!
//! All heavy lifting is done by boostr's format module.

mod api;
pub mod decision;
mod detect;
mod formats;
pub mod hf_cache;
#[cfg(feature = "audio")]
pub mod hub;
#[cfg(feature = "hub")]
pub mod hub_client;
mod kind;
#[cfg(feature = "audio")]
mod speech_load;
mod vision_load;

pub use api::{
    load_model, load_model_as, load_model_layers, load_model_tp, load_model_with_config,
    load_model_with_offloading,
};
pub use detect::{detect_model_source, gguf_models_in_dir, same_weights, ModelFormat, ModelSource};
pub use formats::*;
#[cfg(any(test, feature = "cli"))]
pub(crate) use kind::holds_weight_file;
pub use kind::{check_loadable, detect_model_kind, model_kind_at, ModelKind, NotLoadable};
pub(crate) use kind::{gguf_file_is_model, safetensors_file_is_model};
#[cfg(feature = "audio")]
pub use speech_load::*;
pub use vision_load::*;
