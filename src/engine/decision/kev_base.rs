//! The base checkpoint a Kev adapter applies to, found on disk.
//!
//! Order: `--decision-base` (`inference.decision.base`), a path or a model
//! name as the scheduler resolves one; else the adapter's
//! `base_model_name_or_path` as a path, then by name in the model
//! directory (case-insensitive, `org/name` read as `name`), then in Hugging
//! Face hub layout (`models--org--name/snapshots/<rev>`) in the model
//! directory and in each hub cache, the snapshot of Kev's `base_revision`
//! first. Nothing is downloaded.

use std::path::{Path, PathBuf};

use crate::engine::model_dir::{find_model_path, list_available};
use crate::loader::decision::read_kev_adapter;
use crate::loader::detect_model_source;

/// A resolved base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KevBase {
    /// The name the scheduler loads it under: a model directory entry, so
    /// requests for the base by name share its weights, else its path.
    pub name: String,
    /// The checkpoint.
    pub path: PathBuf,
}

/// Whether `path` holds weights the loader reads.
fn holds_weights(path: &Path) -> bool {
    detect_model_source(path).is_ok()
}

/// The model directory entry named `short`, exact name first, then any
/// case.
fn by_name(model_dir: &Path, short: &str) -> Option<KevBase> {
    let models = list_available(model_dir).unwrap_or_default();
    let exact = models.iter().find(|m| m.name == short);
    let any_case = || models.iter().find(|m| m.name.eq_ignore_ascii_case(short));
    exact
        .or_else(any_case)
        .filter(|m| holds_weights(&m.path))
        .map(|m| KevBase {
            name: m.name.clone(),
            path: m.path.clone(),
        })
}

/// Every snapshot of `id` in hub layout under `dir`, the one named
/// `revision` first, then newest first.
fn snapshots(dir: &Path, id: &str, revision: Option<&str>) -> Vec<PathBuf> {
    let repo = format!("models--{}", id.replace('/', "--"));
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<(bool, std::time::SystemTime, PathBuf)> = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        if !entry
            .file_name()
            .to_string_lossy()
            .eq_ignore_ascii_case(&repo)
        {
            continue;
        }
        let Ok(snaps) = std::fs::read_dir(entry.path().join("snapshots")) else {
            continue;
        };
        for snap in snaps.filter_map(Result::ok).map(|e| e.path()) {
            let pinned = revision.is_some_and(|r| snap.file_name().is_some_and(|n| n == r));
            let time = snap
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            found.push((pinned, time, snap));
        }
    }
    found.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
    found.into_iter().map(|(_, _, p)| p).collect()
}

/// The base of the Kev directory `kev_dir`. `configured` is
/// `--decision-base`, `model_dir` the scheduler's model directory, `hubs`
/// the hub caches to look in.
///
/// # Errors
///
/// When `configured` names no model, when `adapter_config.json` does not
/// read, or when the adapter's base is found nowhere; the message names
/// where it looked and `--decision-base`.
pub fn resolve_kev_base(
    kev_dir: &Path,
    configured: Option<&str>,
    model_dir: &Path,
    hubs: &[PathBuf],
) -> Result<KevBase, String> {
    if let Some(spec) = configured {
        return find_model_path(model_dir, spec)
            .map(|path| KevBase {
                name: spec.to_string(),
                path,
            })
            .ok_or_else(|| {
                format!(
                    "--decision-base {spec:?} (inference.decision.base) names no model: it is \
                     neither a path nor an entry of {}",
                    model_dir.display()
                )
            });
    }
    let adapter = read_kev_adapter(kev_dir)?;
    let id = adapter.base.as_str();
    let as_path = PathBuf::from(id);
    if as_path.is_absolute() && holds_weights(&as_path) {
        return Ok(KevBase {
            name: id.to_string(),
            path: as_path,
        });
    }
    let short = id.rsplit('/').next().unwrap_or(id);
    if let Some(found) = by_name(model_dir, short) {
        return Ok(found);
    }
    let revision = adapter.base_revision.as_deref();
    let roots = || std::iter::once(model_dir).chain(hubs.iter().map(PathBuf::as_path));
    let snaps = roots().flat_map(|d| snapshots(d, id, revision));
    for path in std::iter::once(model_dir.join(id)).chain(snaps) {
        if holds_weights(&path) {
            return Ok(KevBase {
                name: path.display().to_string(),
                path,
            });
        }
    }
    let searched: Vec<String> = roots().map(|p| p.display().to_string()).collect();
    Err(format!(
        "the base model {id:?} of the Kev adapter at {} (adapter_config.json \
         base_model_name_or_path) is not found: no entry {short:?} of {} (any case), and no \
         Hugging Face snapshot in {}. Pass --decision-base <path|name> \
         (inference.decision.base)",
        kev_dir.display(),
        model_dir.display(),
        searched.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::decision::kev::tests::kev_dir;

    fn base_dir(at: &Path) {
        std::fs::create_dir_all(at).expect("mkdir");
        std::fs::write(at.join("config.json"), "{}").expect("write");
        std::fs::write(at.join("model.safetensors"), b"").expect("write");
    }

    #[test]
    fn the_flag_wins_then_the_name_then_the_hub_layout() {
        let kev = kev_dir("Qwen/Qwen3.5-4B-Base");
        let models = tempfile::tempdir().expect("temp dir");
        let hub = tempfile::tempdir().expect("temp dir");
        let hubs = [hub.path().to_path_buf()];
        let resolve = |flag: Option<&str>| resolve_kev_base(kev.path(), flag, models.path(), &hubs);

        let e = resolve(None).expect_err("nothing on disk");
        assert!(
            e.contains("--decision-base") && e.contains("Qwen3.5-4B-Base"),
            "{e}"
        );

        let snap = hub
            .path()
            .join("models--Qwen--Qwen3.5-4B-Base/snapshots/abc");
        base_dir(&snap);
        assert_eq!(resolve(None).expect("in the hub").path, snap);

        base_dir(&models.path().join("qwen3.5-4b-base"));
        let by_name = resolve(None).expect("by name");
        assert_eq!(by_name.name, "qwen3.5-4b-base");

        base_dir(&models.path().join("other"));
        let flagged = resolve(Some("other")).expect("the flag");
        assert_eq!(flagged.name, "other");
        assert_eq!(flagged.path, models.path().join("other"));
        let e = resolve(Some("absent")).expect_err("unknown flag");
        assert!(e.contains("names no model"), "{e}");
    }

    #[test]
    fn the_pinned_revision_wins_among_snapshots() {
        let kev = kev_dir("Org/Base");
        let training = serde_json::json!({"base_revision": "old"});
        let file = kev.path().join("training_config.json");
        std::fs::write(file, training.to_string()).expect("write");
        let models = tempfile::tempdir().expect("temp dir");
        let repo = models.path().join("models--org--base/snapshots");
        base_dir(&repo.join("old"));
        base_dir(&repo.join("new"));
        let found = resolve_kev_base(kev.path(), None, models.path(), &[]).expect("found");
        assert_eq!(found.path, repo.join("old"));
        assert_eq!(found.name, found.path.display().to_string());
    }
}
