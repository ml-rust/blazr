//! Models fetched on first use: a missing model downloaded from the
//! Hugging Face hub, and the on-disk copies and `config.json` of a spec.

use std::path::{Path, PathBuf};

use voxr_narrate::resolve;

use super::super::download::{download, download_file};
use super::super::kinds::{checkpoint_id, ModelKind};
use super::super::spec::looks_like_path;
use super::model_search::ModelSearch;
use crate::offline::is_offline as offline;

impl ModelSearch {
    /// Where a download goes: the first hub cache.
    pub fn download_dir(&self) -> PathBuf {
        self.hubs
            .first()
            .cloned()
            .unwrap_or_else(|| self.cwd.join("models"))
    }

    /// The model `spec` names, downloaded first when it is a Hugging Face
    /// id that is not on disk. The download prints `notice` on standard
    /// error first, or a notice of its own. Offline (`$HF_HUB_OFFLINE`), or
    /// when the download fails, the error is the reason with the next
    /// action, led by `what`, such as `speak`.
    pub fn obtain(
        &self,
        what: &str,
        spec: &str,
        kind: ModelKind,
        notice: Option<&str>,
    ) -> Result<PathBuf, String> {
        let failed = |reason: String| format!("{what}: {reason}");
        let missing = match self.look(spec, kind) {
            Ok(dir) => return Ok(dir),
            Err(m) => m,
        };
        let Some(id) = missing.id.filter(|_| !offline()) else {
            return Err(failed(missing.reason));
        };
        let hub = self.download_dir();
        match notice {
            Some(text) => eprintln!("{text}"),
            None => eprintln!(
                "{what}: no {} model {id} on disk. Downloading it into {}. This happens once.",
                kind.name(),
                hub.display()
            ),
        }
        if let Err(e) = download(&id, kind, &hub) {
            return Err(failed(format!(
                "downloading {id} failed ({e}). {}",
                missing.reason
            )));
        }
        self.find(spec, kind).map_err(failed)
    }
}

impl ModelSearch {
    /// The directories on disk the model `spec` names: its path, else every
    /// place the search holds it. With them, its Hugging Face id when
    /// `spec` is no path.
    pub fn copies(&self, spec: &str, kind: ModelKind) -> (Vec<PathBuf>, Option<String>) {
        let as_path = resolve(&self.cwd, Path::new(spec));
        if as_path.exists() {
            return (vec![as_path], None);
        }
        if looks_like_path(spec) {
            return (Vec::new(), None);
        }
        let id = checkpoint_id(spec, kind);
        let dirs = self
            .candidates(&id)
            .into_iter()
            .filter(|d| d.is_dir())
            .collect();
        (dirs, Some(id))
    }

    /// The `config.json` of the model `spec` names: on disk at its path or
    /// in the search, else fetched alone from the hub, a small file, so a
    /// model's kind is known before its weights are downloaded. `None` when
    /// it is nowhere and cannot be fetched.
    pub fn config_of(&self, spec: &str, kind: ModelKind) -> Option<PathBuf> {
        let as_path = resolve(&self.cwd, Path::new(spec));
        if as_path.exists() {
            return Some(as_path.join("config.json")).filter(|c| c.is_file());
        }
        if looks_like_path(spec) {
            return None;
        }
        let id = checkpoint_id(spec, kind);
        let on_disk = self
            .candidates(&id)
            .into_iter()
            .map(|d| d.join("config.json"))
            .find(|c| c.is_file());
        if on_disk.is_some() || offline() {
            return on_disk;
        }
        download_file(&id, "config.json", &self.download_dir()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mistyped local path used to be taken as a Hugging Face id and
    /// downloaded.
    #[test]
    fn a_missing_path_is_an_error_and_never_an_id() {
        let tmp = tempfile::tempdir().unwrap();
        let search = ModelSearch::new(Vec::new(), Vec::new(), tmp.path());
        let spec = "/no/such/blazr/whisper-base";
        let missing = search
            .look(spec, ModelKind::Whisper)
            .expect_err("missing path");
        assert!(missing.id.is_none(), "a path has no id to download");
        // The message names the resolved path: `C:\no\such\...` on Windows.
        let shown = resolve(tmp.path(), Path::new(spec));
        let want = format!("no such path: {}.", shown.display());
        assert!(missing.reason.starts_with(&want), "{}", missing.reason);
        let err = search
            .obtain("transcribe", spec, ModelKind::Whisper, None)
            .expect_err("no download");
        assert!(err.contains("no such path"), "{err}");
        assert_eq!(search.copies(spec, ModelKind::Whisper), (Vec::new(), None));
        assert_eq!(search.config_of(spec, ModelKind::Whisper), None);
    }
}
