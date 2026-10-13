//! The environment a model search reads: variables and the Hugging Face
//! hub caches resolved against a working directory.

use std::path::{Path, PathBuf};

use voxr_narrate::resolve;

use crate::loader::hf_cache::hub_caches;

/// `$name`, when set and not empty.
pub fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// The Hugging Face hub caches of [`hub_caches`], each resolved against
/// `cwd`.
pub fn default_hubs(cwd: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for dir in hub_caches() {
        let dir = resolve(cwd, &dir);
        if !out.contains(&dir) {
            out.push(dir);
        }
    }
    out
}
