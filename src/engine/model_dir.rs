//! The models of a model directory: the ones it holds, and the path a model
//! name resolves to.

use std::path::{Path, PathBuf};

/// Information about an available model
#[derive(Debug, Clone)]
pub struct AvailableModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub format: String,
}

/// The path `name` names: the name as a path, else the entry of that name in
/// `model_dir`, else `<name>.safetensors`, `<name>.gguf` or
/// `<name>/model.safetensors` there. `None` when none exists.
pub fn find_model_path(model_dir: &Path, name: &str) -> Option<PathBuf> {
    let direct = PathBuf::from(name);
    if direct.exists() {
        return Some(direct);
    }
    let in_dir = model_dir.join(name);
    if in_dir.exists() {
        return Some(in_dir);
    }
    [
        format!("{name}.safetensors"),
        format!("{name}.gguf"),
        format!("{name}/model.safetensors"),
    ]
    .iter()
    .map(|pattern| model_dir.join(pattern))
    .find(|candidate| candidate.exists())
}

/// Every model in `model_dir`: a directory holding `model.safetensors`,
/// `config.json` or a `.gguf`, or a `.safetensors` or `.gguf` file. A
/// missing directory holds none.
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
        let listed = if path.is_dir() {
            path.join("model.safetensors").exists()
                || path.join("config.json").exists()
                || holds_gguf(&path)
        } else {
            path.extension()
                .is_some_and(|e| e == "safetensors" || e == "gguf")
        };
        if listed {
            let format = detect_format(&path);
            models.push(AvailableModelInfo { name, path, format });
        }
    }
    Ok(models)
}

/// Whether the directory `path` holds a `.gguf` file.
fn holds_gguf(path: &Path) -> bool {
    std::fs::read_dir(path)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
        })
        .unwrap_or(false)
}

/// Detect model format from path
fn detect_format(path: &Path) -> String {
    if path.is_file() {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => "SafeTensors",
            Some("gguf") => "GGUF",
            _ => "Unknown",
        }
    } else if path.join("model.safetensors").exists() {
        "SafeTensors"
    } else if holds_gguf(path) {
        "GGUF"
    } else {
        "Unknown"
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
                ("hf".to_string(), "Unknown".to_string()),
                ("q.gguf".to_string(), "GGUF".to_string())
            ]
        );
        assert_eq!(find_model_path(root, "hf"), Some(root.join("hf")));
        assert_eq!(find_model_path(root, "q"), Some(root.join("q.gguf")));
        assert_eq!(find_model_path(root, "absent"), None);
        assert!(list_available(&root.join("missing"))
            .expect("empty")
            .is_empty());
    }
}
