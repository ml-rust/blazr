//! Vision loaders: mmproj towers beside a GGUF, and standalone vision embedders.

pub(super) mod vision;
pub(super) mod vision_embedder;

pub use vision::{load_gguf_with_mmproj, load_mmproj_tensors};
pub use vision_embedder::{load_vision_embedder_from_dir, parse_vision_model_arg};
