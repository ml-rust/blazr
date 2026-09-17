//! Model management API handlers
//!
//! Provides endpoints for listing, inspecting, and monitoring loaded models,
//! Ollama-style (`/api/tags`, `/api/show`, `/api/ps`, `/api/delete`,
//! `/api/copy`, `/api/pull`).

mod common;
mod copy;
mod delete;
mod pull;
mod running;
mod show;
mod tags;

pub use copy::copy_model;
pub use delete::delete_model;
pub use pull::pull_model;
pub use running::list_running;
pub use show::show_model;
pub use tags::list_tags;
