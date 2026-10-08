//! The narration skill's own paths: its config file, its cache root, the
//! model folder and the Hugging Face hub caches it shares, and VoxCPM2's
//! take cache.

use std::path::{Path, PathBuf};

use voxr_narrate::{resolve, CommandError};

use super::config::MotionConfig;

/// `$name`, when set and not empty.
pub(super) fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// The home directory. A machine without one reads as `/`.
pub(super) fn home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

/// The working directory, which every relative path resolves against.
pub fn working_dir() -> Result<PathBuf, CommandError> {
    std::env::current_dir().map_err(|e| {
        CommandError::Failed(format!(
            "reading the working directory: {e}. Run the command from a directory that exists"
        ))
    })
}

/// `$XDG_CONFIG_HOME/motion-video/config.json`, else
/// `~/.config/motion-video/config.json`.
pub fn config_file() -> PathBuf {
    env("XDG_CONFIG_HOME")
        .map_or_else(|| home().join(".config"), PathBuf::from)
        .join("motion-video")
        .join("config.json")
}

/// `$XDG_CACHE_HOME/motion-video`, else `~/.cache/motion-video`.
pub fn cache_root() -> PathBuf {
    env("XDG_CACHE_HOME")
        .map_or_else(|| home().join(".cache"), PathBuf::from)
        .join("motion-video")
}

/// `~` at the start of `path`, alone or before a slash, as the home
/// directory.
fn expand_home(path: &str) -> PathBuf {
    match path.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\') => {
            home().join(rest.trim_start_matches(['/', '\\']))
        }
        _ => PathBuf::from(path),
    }
}

/// The skill's model folder: `$MOTION_VIDEO_MODELS`, else the config's
/// `"models"`, else `<cache root>/models`.
pub fn models_dir(config: &MotionConfig, cwd: &Path) -> PathBuf {
    if let Some(dir) = env("MOTION_VIDEO_MODELS") {
        return resolve(cwd, Path::new(&dir));
    }
    if let Some(dir) = &config.models {
        return resolve(cwd, &expand_home(dir));
    }
    cache_root().join("models")
}

/// The Hugging Face hub caches that can hold a model, in the skill's
/// order: the model folder, `$HF_HUB_CACHE`, then `$HF_HOME/hub` (else
/// `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`).
pub fn hub_caches(config: &MotionConfig, cwd: &Path) -> Vec<PathBuf> {
    let hf_home = env("HF_HOME").map_or_else(
        || {
            env("XDG_CACHE_HOME")
                .map_or_else(|| home().join(".cache"), PathBuf::from)
                .join("huggingface")
        },
        PathBuf::from,
    );
    let mut out: Vec<PathBuf> = Vec::new();
    let candidates = [
        Some(models_dir(config, cwd)),
        env("HF_HUB_CACHE").map(PathBuf::from),
        Some(hf_home.join("hub")),
    ];
    for dir in candidates.into_iter().flatten() {
        let dir = resolve(cwd, &dir);
        if !out.contains(&dir) {
            out.push(dir);
        }
    }
    out
}

/// VoxCPM2's take cache, shared with the skill:
/// `<cache root>/speak/voxcpm`.
pub fn take_cache() -> PathBuf {
    cache_root().join("speak").join("voxcpm")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn home_expands_only_at_the_start() {
        assert_eq!(expand_home("~/models"), home().join("models"));
        assert_eq!(expand_home("~"), home());
        assert_eq!(expand_home("~x/models"), PathBuf::from("~x/models"));
        assert_eq!(expand_home("/m"), PathBuf::from("/m"));
    }
}
