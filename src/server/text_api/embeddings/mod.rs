//! Embeddings endpoint (`POST /v1/embeddings`).

pub(super) mod handler;
pub(super) mod options;
pub(super) mod request;
pub(super) mod response;

pub use handler::embeddings;
pub(crate) use handler::hidden_states;
