//! The models of a model directory: the ones it holds, and the path a model
//! name resolves to.
//!
//! A request names a model inside the model directory. Only an operator, through
//! `--model`, `--decision-base` or a Kev adapter's own files, names a path.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::loader::decision::is_kev_dir;
use crate::loader::{
    detect_model_source, gguf_file_is_model, gguf_models_in_dir, holds_weight_file, model_kind_at,
    safetensors_file_is_model, same_weights, ModelFormat, ModelKind,
};

/// Information about an available model
#[derive(Debug, Clone)]
pub struct AvailableModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub format: String,
}

/// A model of the directory that the text routes do not run, for `blazr list`.
#[derive(Debug, Clone)]
pub struct AuxiliaryModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub kind: ModelKind,
}

/// Whether `name` is a relative path of plain names, so a join stays inside
/// the directory it joins to. A symlink an operator placed there stays valid.
pub(crate) fn stays_in_dir(name: &str) -> bool {
    !name.is_empty()
        && Path::new(name)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))
}

/// The path `name` names.
///
/// - When `operator` is `Some(name)`, the name as a path comes first.
/// - Then the entry of that name in `model_dir`, else `<name>.safetensors`,
///   `<name>.gguf` or `<name>/model.safetensors` there.
///
/// `None` when none exists, when `name` leaves `model_dir`, and when the path
/// is a directory of several GGUF models, because such a directory names no
/// one model. Its id is `<directory>/<file>`, one per GGUF.
pub fn find_model_path(model_dir: &Path, name: &str, operator: Option<&str>) -> Option<PathBuf> {
    let direct = PathBuf::from(name);
    let found = if operator == Some(name) && direct.exists() {
        Some(direct)
    } else if !stays_in_dir(name) {
        None
    } else {
        let in_dir = model_dir.join(name);
        if in_dir.exists() {
            Some(in_dir)
        } else {
            [
                format!("{name}.safetensors"),
                format!("{name}.gguf"),
                format!("{name}/model.safetensors"),
            ]
            .iter()
            .map(|pattern| model_dir.join(pattern))
            .find(|candidate| candidate.exists())
        }
    };
    found.filter(|path| several_ggufs(path).is_empty())
}

/// The model names an operator stated. Each can be a path. A request that spells
/// one reaches that path. Every other name stays inside the model directory.
#[derive(Debug, Clone, Default)]
pub(crate) struct OperatorNames {
    /// `--model`, or the model of a cluster role.
    start: Option<String>,
    /// `--decision-base`.
    decision_base: Option<String>,
    /// Bases that a Kev adapter's own files name.
    kev_bases: Arc<Mutex<HashSet<String>>>,
}

impl OperatorNames {
    pub(crate) fn set_start(&mut self, start: Option<String>) {
        self.start = start;
    }

    pub(crate) fn set_decision_base(&mut self, base: Option<String>) {
        self.decision_base = base;
    }

    /// Accept `name`, a base a Kev adapter's files state.
    pub(crate) fn add_kev_base(&self, name: &str) {
        self.kev_bases
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(name.to_string());
    }

    /// `name` when an operator stated it.
    fn allow<'a>(&self, name: &'a str) -> Option<&'a str> {
        let stated = self.start.as_deref() == Some(name)
            || self.decision_base.as_deref() == Some(name)
            || self
                .kev_bases
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains(name);
        stated.then_some(name)
    }

    /// The path `name` resolves to. The file name of the start model reaches
    /// it when no entry of `model_dir` holds that name, which is the id
    /// `/v1/models` reports for a start model outside the directory.
    pub(crate) fn resolve(&self, model_dir: &Path, name: &str) -> Option<PathBuf> {
        find_model_path(model_dir, name, self.allow(name)).or_else(|| {
            let start = self.start.as_deref()?;
            let spelled = Path::new(start).file_name()?.to_str()?;
            (spelled == name)
                .then(|| find_model_path(model_dir, start, Some(start)))
                .flatten()
        })
    }

    /// The id `/v1/models` reports for the start model when `listed` holds no
    /// model with its weights, with its path. The id is its file name, so the
    /// listing shows no directory, unless an entry already holds that name.
    pub(crate) fn unlisted_start(
        &self,
        model_dir: &Path,
        listed: &[AvailableModelInfo],
    ) -> Option<AvailableModelInfo> {
        let start = self.start.as_deref()?;
        let path = find_model_path(model_dir, start, Some(start))?;
        if listed.iter().any(|m| same_weights(&m.path, &path)) {
            return None;
        }
        let file_name = path.file_name()?.to_str()?;
        let name = if listed.iter().any(|m| m.name == file_name) {
            start
        } else {
            file_name
        };
        Some(AvailableModelInfo {
            name: name.to_string(),
            format: detect_format(&path),
            path,
        })
    }

    /// The name that `/health` and `/api/ps` show for the loaded `name`: the
    /// file name of a start model the operator named by path.
    pub(crate) fn public_name(&self, name: &str) -> String {
        if self.start.as_deref() == Some(name) && !stays_in_dir(name) {
            if let Some(file_name) = Path::new(name).file_name().and_then(|n| n.to_str()) {
                return file_name.to_string();
            }
        }
        name.to_string()
    }
}

/// Every text model in `model_dir`, in name order, each under an id that
/// [`find_model_path`] resolves and the loader reads:
///
/// - a `.gguf` file, under its file name. A file named `mmproj*` is a vision
///   tower, not a model. A file whose header states a refused architecture or
///   no vocabulary is left out
/// - a `.safetensors` file whose tensors form a model, under its file name
/// - a directory that holds a text model the loader finds weights in, under
///   its name
/// - a Kev directory (an adapter and a pointer head over a base it does not
///   hold), under its name
/// - each file of a directory of several GGUF models, under
///   `<directory>/<file>`
///
/// [`list_auxiliary`] holds the rest. A missing directory holds none.
///
/// # Errors
///
/// An error when the directory or one of its entries does not read.
pub fn list_available(model_dir: &Path) -> std::io::Result<Vec<AvailableModelInfo>> {
    let mut models = Vec::new();
    if !model_dir.exists() {
        return Ok(models);
    }
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if !path.is_dir() {
            if text_model_file(&path) {
                let format = detect_format(&path);
                models.push(AvailableModelInfo { name, path, format });
            }
            continue;
        }
        let ggufs = several_ggufs(&path);
        if !ggufs.is_empty() {
            for file in ggufs.into_iter().filter(|f| gguf_file_is_model(f)) {
                let file_name = file
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                models.push(AvailableModelInfo {
                    name: format!("{name}/{file_name}"),
                    format: detect_format(&file),
                    path: file,
                });
            }
        } else if is_kev_dir(&path) || crate::loader::check_loadable(&path).is_ok() {
            let format = detect_format(&path);
            models.push(AvailableModelInfo { name, path, format });
        }
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

/// Whether the file at `path` is a text model [`list_available`] lists.
fn text_model_file(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some("gguf") => gguf_file_is_model(path),
        Some("safetensors") => {
            model_kind_at(path) == ModelKind::Llm && safetensors_file_is_model(path).is_ok()
        }
        _ => false,
    }
}

/// Every model in `model_dir` that the text routes do not run, in name order:
/// a directory of speech, vision or other weights, whatever their format, and
/// a weights file of such a kind.
///
/// # Errors
///
/// An error when the directory or one of its entries does not read.
pub fn list_auxiliary(model_dir: &Path) -> std::io::Result<Vec<AuxiliaryModelInfo>> {
    let mut models = Vec::new();
    if !model_dir.exists() {
        return Ok(models);
    }
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let path = entry.path();
        let weights = if path.is_dir() {
            holds_weight_file(&path)
        } else {
            path.extension()
                .is_some_and(|e| e == "safetensors" || e == "gguf")
        };
        if !weights {
            continue;
        }
        let kind = model_kind_at(&path);
        if kind != ModelKind::Llm {
            models.push(AuxiliaryModelInfo {
                name: entry.file_name().to_string_lossy().to_string(),
                path,
                kind,
            });
        }
    }
    models.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(models)
}

/// The GGUF models of a directory the loader refuses to pick one from: empty
/// for a file, and for a directory with one GGUF or with SafeTensors weights.
fn several_ggufs(path: &Path) -> Vec<PathBuf> {
    if !path.is_dir() || detect_model_source(path).is_ok() {
        return Vec::new();
    }
    let ggufs = gguf_models_in_dir(path);
    if ggufs.len() > 1 {
        ggufs
    } else {
        Vec::new()
    }
}

/// Detect model format from path
fn detect_format(path: &Path) -> String {
    match detect_model_source(path) {
        Ok(source) => match source.format {
            ModelFormat::SafeTensors => "SafeTensors",
            ModelFormat::Gguf => "GGUF",
        },
        Err(_) if is_kev_dir(path) => "Kev",
        Err(_) => "Unknown",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_and_resolves_models() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path();
        std::fs::create_dir(root.join("hf")).expect("mkdir");
        std::fs::write(root.join("hf/config.json"), "{}").expect("write");
        std::fs::write(root.join("hf/model.safetensors"), b"").expect("write");
        std::fs::create_dir(root.join("tokenizer-only")).expect("mkdir");
        std::fs::write(root.join("tokenizer-only/config.json"), "{}").expect("write");
        std::fs::write(root.join("q.gguf"), b"").expect("write");
        std::fs::write(root.join("notes.txt"), b"").expect("write");

        let mut names: Vec<(String, String)> = list_available(root)
            .expect("lists")
            .into_iter()
            .map(|m| (m.name, m.format))
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                ("hf".to_string(), "SafeTensors".to_string()),
                ("q.gguf".to_string(), "GGUF".to_string())
            ]
        );
        assert_eq!(find_model_path(root, "hf", None), Some(root.join("hf")));
        assert_eq!(find_model_path(root, "q", None), Some(root.join("q.gguf")));
        assert_eq!(find_model_path(root, "absent", None), None);
        assert!(list_available(&root.join("missing"))
            .expect("empty")
            .is_empty());
    }

    #[test]
    fn a_directory_of_one_gguf_is_listed_under_its_name() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path();
        std::fs::create_dir(root.join("one")).expect("mkdir");
        std::fs::write(root.join("one/only.gguf"), b"").expect("write");

        let models = list_available(root).expect("lists");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "one");
        assert_eq!(models[0].format, "GGUF");
        assert_eq!(find_model_path(root, "one", None), Some(root.join("one")));
    }

    #[test]
    fn a_directory_of_several_ggufs_lists_each_file_and_not_itself() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path();
        std::fs::create_dir(root.join("x")).expect("mkdir");
        std::fs::write(root.join("x/b.gguf"), b"").expect("write");
        std::fs::write(root.join("x/a.gguf"), b"").expect("write");

        let names: Vec<String> = list_available(root)
            .expect("lists")
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert_eq!(names, ["x/a.gguf", "x/b.gguf"]);
        assert_eq!(
            find_model_path(root, "x/a.gguf", None),
            Some(root.join("x/a.gguf"))
        );
        assert_eq!(
            find_model_path(root, "x/b", None),
            Some(root.join("x/b.gguf"))
        );
        assert_eq!(find_model_path(root, "x", None), None);
    }

    /// `../x` and an absolute path resolve for no request, whether or not the
    /// target exists, so the answer probes nothing.
    #[test]
    fn a_request_never_leaves_the_model_directory() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().join("models");
        std::fs::create_dir(&root).expect("mkdir");
        let outside = dir.path().join("outside.gguf");
        std::fs::write(&outside, b"").expect("write");
        let absolute = outside.display().to_string();

        for name in [
            "../outside.gguf",
            "../outside",
            absolute.as_str(),
            "..",
            ".",
            "./x",
            "",
            "/etc",
            "x/../../outside.gguf",
        ] {
            assert_eq!(find_model_path(&root, name, None), None, "{name:?}");
        }
        assert_eq!(
            find_model_path(&root, &absolute, Some(&absolute)),
            Some(outside.clone()),
            "the operator's path still resolves"
        );
        assert_eq!(
            find_model_path(&root, &absolute, Some("other")),
            None,
            "only the operator's own spelling takes a path"
        );
    }

    /// The test process runs in the package root, which holds `src`. A request
    /// for `src` reaches the listed id, never that working directory entry.
    #[test]
    fn a_working_directory_entry_does_not_shadow_a_listed_id() {
        assert!(Path::new("src").is_dir());
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir(dir.path().join("src")).expect("mkdir");
        std::fs::write(dir.path().join("src/only.gguf"), b"").expect("write");
        assert_eq!(
            find_model_path(dir.path(), "src", None),
            Some(dir.path().join("src"))
        );
    }

    /// A symlink in the model directory that an operator placed stays valid.
    #[cfg(unix)]
    #[test]
    fn an_operator_symlink_in_the_directory_resolves() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().join("models");
        std::fs::create_dir(&root).expect("mkdir");
        let outside = dir.path().join("weights.gguf");
        std::fs::write(&outside, b"").expect("write");
        std::os::unix::fs::symlink(&outside, root.join("link.gguf")).expect("symlink");
        assert_eq!(
            find_model_path(&root, "link.gguf", None),
            Some(root.join("link.gguf"))
        );
    }

    #[test]
    fn other_kinds_leave_the_text_list_and_join_the_auxiliary_list() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path();
        let whisper = root.join("whisper-tiny");
        std::fs::create_dir(&whisper).expect("mkdir");
        std::fs::write(whisper.join("config.json"), r#"{"model_type": "whisper"}"#).expect("write");
        std::fs::write(whisper.join("model.safetensors"), b"").expect("write");
        let kokoro = root.join("Kokoro-82M");
        std::fs::create_dir(&kokoro).expect("mkdir");
        std::fs::write(kokoro.join("config.json"), "{}").expect("write");
        std::fs::write(kokoro.join("kokoro-v1_0.pth"), b"").expect("write");
        std::fs::write(root.join("whisper_mel_fixture.safetensors"), b"").expect("write");
        std::fs::write(root.join("mmproj-f16.gguf"), b"").expect("write");
        std::fs::write(root.join("not-a-model.safetensors"), b"").expect("write");
        std::fs::write(root.join("q.gguf"), b"").expect("write");

        let text: Vec<String> = list_available(root)
            .expect("lists")
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert_eq!(text, ["q.gguf"]);
        let auxiliary: Vec<(String, ModelKind)> = list_auxiliary(root)
            .expect("lists")
            .into_iter()
            .map(|m| (m.name, m.kind))
            .collect();
        assert_eq!(
            auxiliary,
            [
                ("Kokoro-82M".to_string(), ModelKind::Tts),
                ("mmproj-f16.gguf".to_string(), ModelKind::Vision),
                ("whisper-tiny".to_string(), ModelKind::Asr),
                (
                    "whisper_mel_fixture.safetensors".to_string(),
                    ModelKind::Asr
                ),
            ]
        );
    }

    #[test]
    fn the_start_model_outside_the_directory_gets_a_file_name_id() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().join("models");
        std::fs::create_dir(&root).expect("mkdir");
        std::fs::write(root.join("listed.gguf"), b"").expect("write");
        let outside = dir.path().join("start.gguf");
        std::fs::write(&outside, b"").expect("write");
        let spelled = outside.display().to_string();

        let mut names = OperatorNames::default();
        names.set_start(Some(spelled.clone()));
        let listed = list_available(&root).expect("lists");
        let start = names.unlisted_start(&root, &listed).expect("reported");
        assert_eq!(start.name, "start.gguf");
        assert_eq!(names.resolve(&root, "start.gguf"), Some(outside.clone()));
        assert_eq!(names.resolve(&root, &spelled), Some(outside.clone()));
        assert_eq!(names.public_name(&spelled), "start.gguf");
        assert_eq!(names.public_name("listed.gguf"), "listed.gguf");

        // Inside the directory, the listing already holds it.
        names.set_start(Some(root.join("listed.gguf").display().to_string()));
        assert!(names.unlisted_start(&root, &listed).is_none());
    }
}
