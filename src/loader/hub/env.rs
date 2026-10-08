//! The environment a model search reads: variables, the home directory
//! and the Hugging Face hub caches.

use std::path::{Path, PathBuf};

use voxr_narrate::resolve;

/// `$name`, when set and not empty.
pub fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// The home directory. A machine without one reads as `/`.
pub fn home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

/// The Hugging Face hub caches: `$HF_HUB_CACHE`, then `$HF_HOME/hub`
/// (else `$XDG_CACHE_HOME/huggingface/hub`, else
/// `~/.cache/huggingface/hub`), each resolved against `cwd`.
pub fn default_hubs(cwd: &Path) -> Vec<PathBuf> {
    let hf_home = env("HF_HOME").map_or_else(
        || {
            env("XDG_CACHE_HOME")
                .map_or_else(|| home().join(".cache"), PathBuf::from)
                .join("huggingface")
        },
        PathBuf::from,
    );
    let mut out: Vec<PathBuf> = Vec::new();
    for dir in [
        env("HF_HUB_CACHE").map(PathBuf::from),
        Some(hf_home.join("hub")),
    ]
    .into_iter()
    .flatten()
    {
        let dir = resolve(cwd, &dir);
        if !out.contains(&dir) {
            out.push(dir);
        }
    }
    out
}
