//! Whether a VITS model holds the weights voxr loads, `model.safetensors`.
//! A model with other weights only, such as ONNX or `pytorch_model.bin`,
//! runs in the skill's own runner instead.

use std::path::Path;

use crate::loader::hub::{offline, repo_files, ModelKind, ModelSearch};

/// The weights file voxr's VITS loader reads.
const WEIGHTS: &str = "model.safetensors";

/// Whether the VITS model `spec` names holds [`WEIGHTS`]: `Some(true)` when
/// a copy on disk or the hub's file list holds it, `Some(false)` when the
/// hub's list lacks it or a copy on disk holds other weights only, and
/// `None` when neither tells.
pub fn holds_safetensors(search: &ModelSearch, spec: &str) -> Option<bool> {
    let (dirs, id) = search.copies(spec, ModelKind::Vits);
    if dirs.iter().any(|d| d.join(WEIGHTS).is_file()) {
        return Some(true);
    }
    if let Some(files) = id
        .filter(|_| !offline())
        .and_then(|id| repo_files(&id).ok())
    {
        return Some(files.iter().any(|f| f == WEIGHTS));
    }
    dirs.iter().any(|d| other_weights(d)).then_some(false)
}

/// Whether `dir` or its `onnx/` holds ONNX or PyTorch `.bin` weights.
fn other_weights(dir: &Path) -> bool {
    [dir.to_path_buf(), dir.join("onnx")].iter().any(|d| {
        std::fs::read_dir(d).is_ok_and(|entries| {
            entries.filter_map(Result::ok).any(|e| {
                let p = e.path();
                p.is_file() && p.extension().is_some_and(|x| x == "onnx" || x == "bin")
            })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, file: &str) {
        let p = dir.join(file);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"x").unwrap();
    }

    #[test]
    fn a_directory_tells_its_weights() {
        let tmp = tempfile::tempdir().unwrap();
        let search = ModelSearch::new(Vec::new(), Vec::new(), tmp.path());
        let (onnx, native, bare) = ("onnx-only", "native", "bare");
        touch(&tmp.path().join(onnx), "config.json");
        touch(&tmp.path().join(onnx), "onnx/model.onnx");
        touch(&tmp.path().join(native), "model.safetensors");
        touch(&tmp.path().join(native), "onnx/model.onnx");
        touch(&tmp.path().join(bare), "config.json");
        assert_eq!(holds_safetensors(&search, onnx), Some(false));
        assert_eq!(holds_safetensors(&search, native), Some(true));
        assert_eq!(holds_safetensors(&search, bare), None);
    }
}
