//! Ollama API: `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`.

#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
pub(in crate::server) mod e2e_rig;
pub(super) mod model_name;
pub(super) mod ndjson;
pub(super) mod ollama_chat;
pub(super) mod ollama_convert;
pub(super) mod ollama_embed;
pub(super) mod ollama_error;
pub(super) mod ollama_generate;
pub(super) mod ollama_load;
pub(super) mod ollama_options;
pub(super) mod ollama_render;
pub(super) mod ollama_reply;
pub(super) mod ollama_stream;
pub(super) mod ollama_types;
