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
pub(crate) mod gguf;
mod gguf_config;
mod gguf_info;
mod gguf_rope;
mod gguf_template;
pub mod hf_cache;
#[cfg(feature = "audio")]
pub mod hub;
#[cfg(feature = "hub")]
pub mod hub_client;
mod safetensors;
#[cfg(feature = "audio")]
mod tts;
mod vision;
mod vision_embedder;
#[cfg(feature = "audio")]
mod whisper_loader;

pub use api::{
    load_model, load_model_as, load_model_tp, load_model_with_config, load_model_with_offloading,
};
pub use detect::{detect_model_source, ModelFormat, ModelSource};
pub use gguf::{
    load_gguf, load_gguf_bytes_with_tokenizer_as, load_gguf_with_tokenizer,
    load_gguf_with_tokenizer_as, GgufBytesModel,
};
pub use gguf_info::{get_gguf_info, GgufInfo};
pub use gguf_template::{chat_template_from_metadata, gguf_chat_template};
pub use safetensors::{
    load_safetensors, load_safetensors_bytes, load_safetensors_with_offloading,
    load_safetensors_with_report, model_config_inference, model_torch_dtype, OffloadingInfo,
    OffloadingOptions, SafeTensorsBytes, TensorLoadReport,
};
#[cfg(feature = "audio")]
pub use tts::{
    kokoro_g2p, load_kokoro_engine, load_tts_from_dir, parse_tts_model_arg, voxcpm2_source,
    voxcpm2_source_at, Voxcpm2Source,
};
pub use vision::{load_gguf_with_mmproj, load_mmproj_tensors};
pub use vision_embedder::{load_vision_embedder_from_dir, parse_vision_model_arg};
#[cfg(feature = "audio")]
pub use whisper_loader::{load_whisper_from_dir, parse_asr_model_arg};
