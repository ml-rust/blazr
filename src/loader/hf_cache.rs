//! The Hugging Face hub cache directories, read from the environment.
//! Nothing here touches the disk or the network.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The home directory. A machine without one reads as `/`.
pub fn home() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

/// The process environment's `$name`.
fn process_var(name: &str) -> Option<OsString> {
    std::env::var_os(name)
}

/// The Hugging Face hub caches: `$HF_HUB_CACHE`, then `$HF_HOME/hub`
/// (else `$XDG_CACHE_HOME/huggingface/hub`, else
/// `~/.cache/huggingface/hub`). An empty variable reads as unset. Paths
/// come back as the environment gives them, relative ones unresolved.
pub fn hub_caches() -> Vec<PathBuf> {
    hub_caches_with(process_var, &home())
}

/// [`hub_caches`] over the variable lookup `lookup` and the home
/// directory `home`.
fn hub_caches_with(lookup: impl Fn(&str) -> Option<OsString>, home: &Path) -> Vec<PathBuf> {
    let var = |name: &str| lookup(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let hf_home = var("HF_HOME").unwrap_or_else(|| {
        var("XDG_CACHE_HOME")
            .unwrap_or_else(|| home.join(".cache"))
            .join("huggingface")
    });
    let mut out = Vec::with_capacity(2);
    for dir in [var("HF_HUB_CACHE"), Some(hf_home.join("hub"))]
        .into_iter()
        .flatten()
    {
        if !out.contains(&dir) {
            out.push(dir);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caches(vars: &[(&str, &str)]) -> Vec<PathBuf> {
        let lookup = |name: &str| {
            vars.iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| OsString::from(v))
        };
        hub_caches_with(lookup, Path::new("/home/u"))
    }

    #[test]
    fn the_hub_cache_comes_first_then_the_hf_home_hub() {
        let found = caches(&[("HF_HUB_CACHE", "/hub"), ("HF_HOME", "/hf")]);
        assert_eq!(found, [PathBuf::from("/hub"), PathBuf::from("/hf/hub")]);
    }

    #[test]
    fn xdg_then_home_stand_in_for_hf_home() {
        let xdg = caches(&[("XDG_CACHE_HOME", "/xdg")]);
        assert_eq!(xdg, [PathBuf::from("/xdg/huggingface/hub")]);
        let plain = caches(&[]);
        assert_eq!(plain, [PathBuf::from("/home/u/.cache/huggingface/hub")]);
    }

    #[test]
    fn empty_variables_read_as_unset_and_duplicates_drop() {
        let empty = caches(&[("HF_HUB_CACHE", ""), ("HF_HOME", "")]);
        assert_eq!(empty, [PathBuf::from("/home/u/.cache/huggingface/hub")]);
        let same = caches(&[("HF_HUB_CACHE", "/hf/hub"), ("HF_HOME", "/hf")]);
        assert_eq!(same, [PathBuf::from("/hf/hub")]);
    }

    #[test]
    fn relative_paths_stay_unresolved() {
        let found = caches(&[("HF_HOME", "rel")]);
        assert_eq!(found, [PathBuf::from("rel/hub")]);
    }

    #[test]
    fn the_process_list_ends_with_the_hf_home_hub() {
        let found = hub_caches();
        let last = found.last().map(|p| p.ends_with("hub"));
        assert_eq!(last, Some(true), "{found:?}");
    }
}
