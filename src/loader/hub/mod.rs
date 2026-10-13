//! Finding a model on disk, and downloading a missing one from the
//! Hugging Face hub on first use.

mod download;
mod env;
mod kinds;
mod search;

pub use super::hf_cache::home;
pub use download::{offline, repo_files};
pub use env::{default_hubs, env};
pub use kinds::{
    checkpoint_id, missing_file, ModelKind, CLAP_ID, KOKORO_G2P_ID, OOV_FILES, TAGGER_FILES,
    VOXCPM_ID,
};
pub use search::ModelSearch;
