//! Embeddings endpoint (`POST /v1/embeddings`).

pub(in crate::server) mod handler;
pub(in crate::server) mod options;
pub(super) mod request;
pub(super) mod response;

pub use handler::embeddings;
pub(crate) use handler::hidden_states;
