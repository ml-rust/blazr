//! Checkpoint formats: GGUF (config, rope, template, header info) and SafeTensors.

pub(crate) mod gguf;
pub(super) mod gguf_config;
pub(super) mod gguf_info;
pub(super) mod gguf_rope;
pub(super) mod gguf_template;
pub(super) mod safetensors;

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
