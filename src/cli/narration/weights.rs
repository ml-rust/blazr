//! Whether a VITS or Whisper model holds the weights voxr loads,
//! `model.safetensors`. A model with other weights only, such as ONNX or
//! `pytorch_model.bin`, runs in the skill's own runner instead.

use std::path::Path;

use voxr_narrate::CommandError;

use crate::loader::hub::{offline, repo_files, ModelKind, ModelSearch};

/// The weights file voxr's VITS and Whisper loaders read.
const WEIGHTS: &str = "model.safetensors";

/// Whether the `kind` model `spec` names holds [`WEIGHTS`]: `Some(true)` when
/// a copy on disk or the hub's file list holds it, `Some(false)` when the
/// hub's list lacks it or a copy on disk holds other weights only, and
/// `None` when neither tells.
pub fn holds_safetensors(search: &ModelSearch, spec: &str, kind: ModelKind) -> Option<bool> {
    let (dirs, id) = search.copies(spec, kind);
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

/// Refuses, exit 3, the Whisper model `model` when it holds no
/// `model.safetensors`: the skill's own runner reads its other weights.
/// `what` names where the model came from, such as `transcribe --model X`.
/// A model the search does not tell about passes, and its download decides.
pub fn refuse_other_whisper_weights(
    search: &ModelSearch,
    what: &str,
    model: &str,
) -> std::result::Result<(), CommandError> {
    if holds_safetensors(search, model, ModelKind::Whisper) == Some(false) {
        return Err(CommandError::Unsupported(format!(
            "{what}: blazr reads Whisper checkpoints with model.safetensors, and this model holds other weights. Run the skill's own runner, or point it at a Whisper checkpoint."
        )));
    }
    Ok(())
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
        let vits = ModelKind::Vits;
        assert_eq!(holds_safetensors(&search, onnx, vits), Some(false));
        assert_eq!(holds_safetensors(&search, native, vits), Some(true));
        assert_eq!(holds_safetensors(&search, bare, vits), None);
    }

    #[test]
    fn a_whisper_directory_tells_its_weights() {
        let tmp = tempfile::tempdir().unwrap();
        let search = ModelSearch::new(Vec::new(), Vec::new(), tmp.path());
        let (onnx, bin, native, bare) = ("onnx-only", "bin-only", "native", "bare");
        touch(&tmp.path().join(onnx), "onnx/encoder_model.onnx");
        touch(&tmp.path().join(bin), "pytorch_model.bin");
        touch(&tmp.path().join(native), "model.safetensors");
        touch(&tmp.path().join(bare), "config.json");
        let whisper = ModelKind::Whisper;
        assert_eq!(holds_safetensors(&search, onnx, whisper), Some(false));
        assert_eq!(holds_safetensors(&search, bin, whisper), Some(false));
        assert_eq!(holds_safetensors(&search, native, whisper), Some(true));
        assert_eq!(holds_safetensors(&search, bare, whisper), None);
        assert!(refuse_other_whisper_weights(&search, "x", onnx).is_err());
        assert!(refuse_other_whisper_weights(&search, "x", native).is_ok());
        assert!(refuse_other_whisper_weights(&search, "x", bare).is_ok());
    }
}
