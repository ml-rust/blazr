//! A missing model downloaded on first use: the files its loader reads,
//! from the Hugging Face hub into a hub cache in hub layout, where the
//! model search finds them next time.

use std::path::{Path, PathBuf};

use super::kinds::ModelKind;
use crate::loader::hub_client::{sync_client, sync_get, sync_model};

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
            ModelKind::KokoroG2p => name.starts_with("tagger/") || name.starts_with("oov/"),
            ModelKind::Clap => name == "model.safetensors" || name == "pytorch_model.bin",
        }
}

/// Downloads the files of `id` that a `kind` loader reads into the hub
/// cache `hub`, with the hub's progress bars on standard error, and
/// returns the snapshot directory. The error is the hub's reason.
pub fn download(id: &str, kind: ModelKind, hub: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(hub).map_err(|e| format!("creating {}: {e}", hub.display()))?;
    let client = sync_client(Some(hub)).map_err(|e| e.to_string())?;
    let repo = sync_model(&client, id);
    let info = repo.info().send().map_err(|e| e.to_string())?;
    let mut names: Vec<String> = info
        .siblings
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.rfilename)
        .filter(|n| wanted(n, kind))
        .collect();
    // A repo with both weights files needs one: the SafeTensors copy.
    if names.iter().any(|n| n == "model.safetensors") {
        names.retain(|n| n != "pytorch_model.bin");
    }
    if names.is_empty() {
        return Err(format!(
            "{id} holds none of the files a {kind:?} model needs"
        ));
    }
    let mut snapshot = None;
    for name in &names {
        let path = sync_get(&repo, name, true).map_err(|e| format!("{name}: {e}"))?;
        // The snapshot directory holds the file as the repo does: one
        // directory up for each part of its repo path.
        snapshot = snapshot.or_else(|| {
            path.ancestors()
                .nth(name.split('/').count())
                .map(Path::to_path_buf)
        });
    }
    snapshot.ok_or_else(|| format!("{id}: the hub cache holds no snapshot directory"))
}

/// Every file of the repo `id` on the hub, as paths in the repo. The error
/// is the hub's reason.
pub fn repo_files(id: &str) -> Result<Vec<String>, String> {
    let client = sync_client(None).map_err(|e| e.to_string())?;
    let info = sync_model(&client, id)
        .info()
        .send()
        .map_err(|e| e.to_string())?;
    Ok(info
        .siblings
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.rfilename)
        .collect())
}

/// Downloads the one repo file `name` of `id` into the hub cache `hub`
/// and returns its path.
pub fn download_file(id: &str, name: &str, hub: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(hub).map_err(|e| format!("creating {}: {e}", hub.display()))?;
    let client = sync_client(Some(hub)).map_err(|e| e.to_string())?;
    sync_get(&sync_model(&client, id), name, false).map_err(|e| format!("{name}: {e}"))
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
        assert!(wanted("tagger/tagger.json", ModelKind::KokoroG2p));
        assert!(wanted("oov/model.safetensors", ModelKind::KokoroG2p));
        assert!(!wanted("README.md", ModelKind::KokoroG2p));
        assert!(wanted("preprocessor_config.json", ModelKind::Clap));
        assert!(wanted("pytorch_model.bin", ModelKind::Clap));
        assert!(!wanted("pytorch_model.bin", ModelKind::Whisper));
    }
}
