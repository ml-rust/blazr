//! The narration skill's own paths: its config file, its cache root, the
//! model folder and the Hugging Face hub caches it shares, and VoxCPM2's
//! take cache.

use std::path::{Path, PathBuf};

use voxr_narrate::{resolve, CommandError};

use super::config::MotionConfig;
use crate::loader::hub::{default_hubs, env, home, ModelSearch};

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
/// order: the model folder, then [`default_hubs`].
pub fn hub_caches(config: &MotionConfig, cwd: &Path) -> Vec<PathBuf> {
    let mut out = vec![models_dir(config, cwd)];
    for dir in default_hubs(cwd) {
        if !out.contains(&dir) {
            out.push(dir);
        }
    }
    out
}

/// Where the narration commands look for models: `$BOOSTR_MODELS_DIR`,
/// then the skill's model folder, by name, then [`hub_caches`]. A
/// download goes into the model folder.
pub fn model_search(config: &MotionConfig, cwd: &Path) -> ModelSearch {
    let mut roots: Vec<PathBuf> = env("BOOSTR_MODELS_DIR")
        .map(|d| resolve(cwd, Path::new(&d)))
        .into_iter()
        .collect();
    let folder = models_dir(config, cwd);
    if !roots.contains(&folder) {
        roots.push(folder);
    }
    ModelSearch::new(roots, hub_caches(config, cwd), cwd)
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
