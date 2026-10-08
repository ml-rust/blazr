//! A missing model downloaded on first use, as the skill downloads its
//! own: the files its loader reads, from the Hugging Face hub into the
//! skill's model folder in hub layout, where the model search finds them
//! next time.

use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;

use super::kinds::ModelKind;

/// Whether the hub is off limits: `$HF_HUB_OFFLINE` set to anything but
/// `0` or `false`.
pub fn offline() -> bool {
    std::env::var("HF_HUB_OFFLINE")
        .is_ok_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
}

/// Whether the loader of `kind` reads the repo file `name`: the configs
/// and tokenizer files at the top, and the weights the loader opens. ONNX,
/// TensorFlow and Flax copies are left on the hub.
pub fn wanted(name: &str, kind: ModelKind) -> bool {
    let top = !name.contains('/');
    let config =
        top && (name.ends_with(".json") || name.ends_with(".txt") || name == "tokenizer.model");
    config
        || match kind {
            ModelKind::Whisper | ModelKind::Vits => name == "model.safetensors",
            ModelKind::Kokoro => {
                (top && name.ends_with(".pth"))
                    || (name.starts_with("voices/") && name.ends_with(".pt"))
            }
            ModelKind::Voxcpm => name == "model.safetensors" || name == "audiovae.pth",
        }
}

/// Downloads the files of `id` that a `kind` loader reads into the hub
/// cache `hub`, with the hub's progress bars on standard error, and
/// returns the snapshot directory. The error is the hub's reason.
pub fn download(id: &str, kind: ModelKind, hub: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(hub).map_err(|e| format!("creating {}: {e}", hub.display()))?;
    let api = ApiBuilder::from_env()
        .with_cache_dir(hub.to_path_buf())
        .with_progress(true)
        .build()
        .map_err(|e| e.to_string())?;
    let repo = api.model(id.to_string());
    let info = repo.info().map_err(|e| e.to_string())?;
    let names: Vec<String> = info
        .siblings
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|n| wanted(n, kind))
        .collect();
    // A file at the top of the repo sits right in the snapshot directory.
    let first = names
        .iter()
        .find(|n| !n.contains('/'))
        .ok_or_else(|| format!("{id} holds none of the files a {kind:?} model needs"))?;
    let mut snapshot = None;
    for name in &names {
        let path = repo.get(name).map_err(|e| format!("{name}: {e}"))?;
        if name == first {
            snapshot = path.parent().map(Path::to_path_buf);
        }
    }
    snapshot.ok_or_else(|| format!("{id}: the hub cache holds no snapshot directory"))
}

/// Downloads the one repo file `name` of `id` into the hub cache `hub`
/// and returns its path.
pub fn download_file(id: &str, name: &str, hub: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(hub).map_err(|e| format!("creating {}: {e}", hub.display()))?;
    let api = ApiBuilder::from_env()
        .with_cache_dir(hub.to_path_buf())
        .with_progress(false)
        .build()
        .map_err(|e| e.to_string())?;
    api.model(id.to_string())
        .get(name)
        .map_err(|e| format!("{name}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_files_the_loader_reads_are_fetched() {
        assert!(wanted("config.json", ModelKind::Whisper));
        assert!(wanted("merges.txt", ModelKind::Whisper));
        assert!(wanted("model.safetensors", ModelKind::Whisper));
        assert!(!wanted("onnx/encoder_model.onnx", ModelKind::Whisper));
        assert!(!wanted("pytorch_model.bin", ModelKind::Whisper));
        assert!(!wanted("tf_model.h5", ModelKind::Vits));
        assert!(wanted("kokoro-v1_0.pth", ModelKind::Kokoro));
        assert!(wanted("voices/af_heart.pt", ModelKind::Kokoro));
        assert!(!wanted("samples/HEARME.wav", ModelKind::Kokoro));
        assert!(wanted("audiovae.pth", ModelKind::Voxcpm));
        assert!(!wanted("model.safetensors", ModelKind::Kokoro));
    }
}
