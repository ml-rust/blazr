//! Model loading utilities
//!
//! This module provides unified model loading from various formats:
//! - SafeTensors (HuggingFace standard, including AWQ quantized)
//! - GGUF (llama.cpp format, for quantized models)
//!
//! All heavy lifting is done by boostr's format module.

mod api;
mod detect;
pub(crate) mod gguf;
mod safetensors;
mod vision;

pub use api::{load_model, load_model_tp, load_model_with_config, load_model_with_offloading};
pub use detect::{detect_model_source, ModelFormat, ModelSource};
pub use gguf::{get_gguf_info, load_gguf, load_gguf_with_tokenizer, GgufInfo};
pub use safetensors::{
    load_safetensors, load_safetensors_with_offloading, OffloadingInfo, OffloadingOptions,
};
pub use vision::{load_gguf_with_mmproj, load_mmproj_tensors};
