//! Finding a speech model on disk: a path, or a Hugging Face id looked up
//! under `$BOOSTR_MODELS_DIR`, the skill's model folder and the hub
//! caches, each candidate checked for every file its loader reads. Models
//! are never downloaded: a missing one names the next action.

use std::path::{Path, PathBuf};

use voxr_narrate::{resolve, CommandError};

use super::config::MotionConfig;
use super::download::{download, download_file, offline};
use super::kinds::{checkpoint_id, missing_file, ModelKind};
use super::paths::{env, hub_caches, models_dir};

/// Where models are looked for.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSearch {
    /// Directories holding models by name: `$BOOSTR_MODELS_DIR`, then the
    /// skill's model folder.
    roots: Vec<PathBuf>,
    /// Hugging Face hub caches.
    hubs: Vec<PathBuf>,
    /// The working directory a relative path resolves against.
    cwd: PathBuf,
}

impl ModelSearch {
    /// The search the config and the environment set.
    pub fn new(config: &MotionConfig, cwd: &Path) -> Self {
        let mut roots: Vec<PathBuf> = env("BOOSTR_MODELS_DIR")
            .map(|d| resolve(cwd, Path::new(&d)))
            .into_iter()
            .collect();
        let folder = models_dir(config, cwd);
        if !roots.contains(&folder) {
            roots.push(folder);
        }
        Self {
            roots,
            hubs: hub_caches(config, cwd),
            cwd: cwd.to_path_buf(),
        }
    }

    /// Every snapshot of `id` in hub layout under `dir`, newest first.
    fn snapshots(dir: &Path, id: &str) -> Vec<PathBuf> {
        let snaps = dir
            .join(format!("models--{}", id.replace('/', "--")))
            .join("snapshots");
        let mut found: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&snaps)
            .map(|d| {
                d.filter_map(Result::ok)
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .map(|p| {
                        let t = p
                            .metadata()
                            .and_then(|m| m.modified())
                            .unwrap_or(std::time::UNIX_EPOCH);
                        (t, p)
                    })
                    .collect()
            })
            .unwrap_or_default();
        found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        found.into_iter().map(|(_, p)| p).collect()
    }

    /// Every place `id` can sit, in search order.
    fn candidates(&self, id: &str) -> Vec<PathBuf> {
        let name = id.rsplit('/').next().unwrap_or(id);
        let mut out = Vec::new();
        for root in &self.roots {
            out.push(root.join(name));
            out.push(root.join(id));
            out.extend(Self::snapshots(root, id));
        }
        for hub in &self.hubs {
            out.extend(Self::snapshots(hub, id));
        }
        out.dedup();
        out
    }

    /// The directories searched, for a message.
    fn searched(&self) -> String {
        // The skill's model folder is searched as a root and as a hub
        // cache, and is named once.
        let mut all: Vec<String> = Vec::new();
        for p in self.roots.iter().chain(&self.hubs) {
            let text = p.display().to_string();
            if !all.contains(&text) {
                all.push(text);
            }
        }
        all.join(", ")
    }

    /// The model `spec` names, checked complete: a path that exists, else a
    /// Hugging Face id (`org/name`) found in the search. The error is the
    /// reason, with the next action.
    pub fn find(&self, spec: &str, kind: ModelKind) -> Result<PathBuf, String> {
        self.look(spec, kind).map_err(|m| m.reason)
    }

    /// [`Self::find`], with what a download can fix.
    fn look(&self, spec: &str, kind: ModelKind) -> Result<PathBuf, Missing> {
        let as_path = resolve(&self.cwd, Path::new(spec));
        if as_path.exists() {
            return match missing_file(&as_path, kind) {
                None => Ok(as_path),
                Some(file) => Err(Missing {
                    reason: incomplete(kind, &as_path, &file, None),
                    id: None,
                }),
            };
        }
        let id = checkpoint_id(spec, kind);
        let mut partial: Option<(PathBuf, String)> = None;
        for dir in self.candidates(&id) {
            if !dir.exists() {
                continue;
            }
            match missing_file(&dir, kind) {
                None => return Ok(dir),
                Some(file) => {
                    partial.get_or_insert((dir, file));
                }
            }
        }
        if let Some((dir, file)) = partial {
            // The hub never holds the G2P tagger: a download cannot add it.
            let fixable = !file.starts_with("g2p/");
            return Err(Missing {
                reason: incomplete(kind, &dir, &file, Some(&id)),
                id: fixable.then_some(id),
            });
        }
        Err(Missing {
            reason: format!(
                "the {} model {id} is not on disk. Searched: {}. Download it with `hf download {id}`, or pass its directory.",
                kind.name(),
                self.searched()
            ),
            id: Some(id),
        })
    }

    /// Where a download goes: the first hub cache, the skill's model
    /// folder.
    pub fn download_dir(&self) -> PathBuf {
        self.hubs
            .first()
            .cloned()
            .unwrap_or_else(|| self.cwd.join("models"))
    }

    /// The model `spec` names, downloaded first when it is a Hugging Face
    /// id that is not on disk, as the skill downloads on first use. The
    /// download prints `notice` first, or a notice of its own. Offline
    /// (`$HF_HUB_OFFLINE`), or when the download fails, the run fails with
    /// the next action, led by `what`, such as `speak`.
    pub fn obtain(
        &self,
        what: &str,
        spec: &str,
        kind: ModelKind,
        notice: Option<&str>,
    ) -> Result<PathBuf, CommandError> {
        let failed = |reason: String| CommandError::Failed(format!("{what}: {reason}"));
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
    /// The `config.json` of the model `spec` names: on disk at its path or
    /// in the search, else fetched alone from the hub, a small file, so a
    /// model's kind is known before its weights are downloaded. `None` when
    /// it is nowhere and cannot be fetched.
    pub fn config_of(&self, spec: &str, kind: ModelKind) -> Option<PathBuf> {
        let as_path = resolve(&self.cwd, Path::new(spec));
        if as_path.exists() {
            return Some(as_path.join("config.json")).filter(|c| c.is_file());
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

/// A model the search did not find whole.
struct Missing {
    /// Why, with the next action.
    reason: String,
    /// The Hugging Face id a download can fix it from.
    id: Option<String>,
}

/// Why a model that lacks `file` cannot load, with the next action.
fn incomplete(kind: ModelKind, dir: &Path, file: &str, id: Option<&str>) -> String {
    let next = if file.starts_with("g2p/tagger/") {
        "Convert spaCy's en_core_web_sm tagger into g2p/tagger with voxr's assets/spacy_en/convert_tagger.py".to_string()
    } else {
        match id {
            Some(id) => format!("Download it again with `hf download {id}`"),
            None => format!("Point it at a complete {} checkpoint", kind.name()),
        }
    };
    format!(
        "the {} model at {} lacks {file}. {next}.",
        kind.name(),
        dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::super::kinds::{KOKORO_ID, VOXCPM_ID};
    use super::*;
    use voxr_narrate::{DEFAULT_ASR_MODEL, KOKORO_MODEL, SMALL_ASR_MODEL};

    #[test]
    fn skill_ids_name_their_checkpoints() {
        assert_eq!(
            checkpoint_id(DEFAULT_ASR_MODEL, ModelKind::Whisper),
            "openai/whisper-base"
        );
        assert_eq!(
            checkpoint_id(SMALL_ASR_MODEL, ModelKind::Whisper),
            "openai/whisper-small"
        );
        assert_eq!(
            checkpoint_id("openai/whisper-large-v3", ModelKind::Whisper),
            "openai/whisper-large-v3"
        );
        assert_eq!(checkpoint_id(KOKORO_MODEL, ModelKind::Kokoro), KOKORO_ID);
        assert_eq!(
            checkpoint_id("facebook/mms-tts-zlm", ModelKind::Vits),
            "facebook/mms-tts-zlm"
        );
    }

    fn touch(dir: &Path, files: &[&str]) {
        for f in files {
            let p = dir.join(f);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(p, b"x").unwrap();
        }
    }

    #[test]
    fn completeness_names_the_first_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let d = tmp.path();
        touch(d, &["config.json", "tokenizer.json", "model.safetensors"]);
        assert_eq!(
            missing_file(d, ModelKind::Whisper).as_deref(),
            Some("generation_config.json")
        );
        touch(d, &["generation_config.json"]);
        assert_eq!(missing_file(d, ModelKind::Whisper), None);
        assert_eq!(
            missing_file(d, ModelKind::Kokoro).as_deref(),
            Some("g2p/tagger/tagger.safetensors")
        );
        assert_eq!(
            missing_file(d, ModelKind::Voxcpm).as_deref(),
            Some("audiovae.pth")
        );
        touch(d, &["one.gguf"]);
        assert_eq!(missing_file(d, ModelKind::Voxcpm), None);
    }

    #[test]
    fn an_id_is_found_by_name_or_in_hub_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("models");
        let hub = tmp.path().join("hub");
        touch(
            &root.join("whisper-base"),
            &[
                "config.json",
                "tokenizer.json",
                "model.safetensors",
                "generation_config.json",
            ],
        );
        let snap = hub.join("models--openai--whisper-small/snapshots/abc");
        touch(
            &snap,
            &[
                "config.json",
                "tokenizer.json",
                "model.safetensors",
                "generation_config.json",
            ],
        );
        let search = ModelSearch {
            roots: vec![root.clone()],
            hubs: vec![hub],
            cwd: tmp.path().to_path_buf(),
        };
        assert_eq!(
            search.find(DEFAULT_ASR_MODEL, ModelKind::Whisper).ok(),
            Some(root.join("whisper-base"))
        );
        assert_eq!(
            search.find(SMALL_ASR_MODEL, ModelKind::Whisper).ok(),
            Some(snap)
        );
        let missing = search.find(VOXCPM_ID, ModelKind::Voxcpm).err();
        assert!(missing.is_some_and(|e| e
            .starts_with("the VoxCPM2 model openbmb/VoxCPM2 is not on disk. Searched: ")
            && e.ends_with(
                "Download it with `hf download openbmb/VoxCPM2`, or pass its directory."
            )));
    }
}
