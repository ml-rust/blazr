pub mod awq;
mod bytes;
mod checkpoint_map;
mod config;
mod config_dtype;
mod decomposed;
mod detect_arch;
pub mod gptq;
mod hf_qwen35;
mod loaders;
mod meta;
mod offloading;
mod regular;
mod tensors;

pub use bytes::{load_safetensors_bytes, SafeTensorsBytes};
pub use checkpoint_map::TensorLoadReport;
pub use config::model_config_inference;
pub use config_dtype::model_torch_dtype;
pub(crate) use detect_arch::detect_architecture_from_loader;
pub use loaders::{
    load_safetensors, load_safetensors_tp, load_safetensors_with_config,
    load_safetensors_with_offloading,
};
pub use meta::read_safetensors_config;
pub use offloading::{OffloadingInfo, OffloadingOptions};
pub use regular::{load_safetensors_layers, load_safetensors_with_report};
