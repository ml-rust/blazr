//! Builds a [`KokoroEngine`] from a model directory.
//!
//! ```text
//! PATH/
//!   config.json          — KokoroConfig
//!   vocab.json           — phoneme → id map
//!   model.safetensors    — weights (may be sharded)
//!   voices/               — optional per-voice .safetensors / .pt files
//!   g2p/tagger/           — spaCy tagger (tagger.safetensors, tagger.json)
//!   g2p/oov/              — g2p_lm out-of-vocabulary fallback
//!                           either missing: the repo fs90/kokoro-g2p
//! ```

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::model::audio::kokoro::KokoroClient;
use boostr::{DType, Runtime};
use voxr::kokoro::KokoroEngine;
use voxr::{default_kokoro_voices, TtsBundle};

use super::kokoro_g2p::kokoro_g2p;
use crate::loader::hub::{missing_file, ModelKind, ModelSearch};
use crate::server::handlers::{ServerDevice, ServerRuntime};

/// Builds a Kokoro engine from `path`'s `config.json` + weights on the
/// server's `device` and wraps it in a scaffolding bundle. Kokoro ships the
/// vocab inlined in `config.json` under a `"vocab"` key. A separate
/// `vocab.json` beside it takes precedence when it exists (custom
/// variants). Its G2P models are found by [`ModelSearch::from_env`], or
/// downloaded ([`kokoro_g2p`]).
pub(super) fn load_kokoro(path: &Path, device: &ServerDevice) -> Result<TtsBundle> {
    let cwd = std::env::current_dir().map_err(|e| anyhow!("reading the working directory: {e}"))?;
    let engine =
        load_kokoro_engine::<ServerRuntime>(path, &ModelSearch::from_env(&cwd), "tts", device)
            .map_err(|e| anyhow!("{e}"))?;
    let sample_rate = engine.sample_rate();
    Ok(TtsBundle::scaffolding(default_kokoro_voices(), sample_rate).with_engine(Arc::new(engine)))
}

/// Builds a Kokoro engine from the checkpoint at `path` on `device`: the
/// weights, the phoneme vocab, the voices under `voices/` (then the
/// bundled defaults), and the G2P models [`kokoro_g2p`] resolves through
/// `search`. The G2P models stay on the host. A checkpoint that lacks its
/// config or weights is refused before any G2P model downloads. The error
/// is the reason with the next action, led by `what`, such as `speak`.
pub fn load_kokoro_engine<R>(
    path: &Path,
    search: &ModelSearch,
    what: &str,
    device: &R::Device,
) -> std::result::Result<KokoroEngine<R>, String>
where
    R: Runtime<DType = DType>,
    R::Client: KokoroClient<R>,
{
    if let Some(file) = missing_file(path, ModelKind::Kokoro) {
        return Err(format!(
            "{what}: the Kokoro model at {} lacks {file}. Point it at a complete Kokoro checkpoint.",
            path.display()
        ));
    }
    let g2p = kokoro_g2p(path, search, what)?;
    KokoroEngine::from_dir_on(path, Some(g2p), device)
        .map_err(|e| format!("{what}: loading Kokoro from {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use boostr::Runtime;
    use voxr::voxcpm::VoxCpm2SynthOptions;

    use super::super::load_tts_from_dir;
    use crate::server::handlers::ServerRuntime;

    #[test]
    fn load_errors_when_config_present_but_weights_missing() {
        // Config + vocab-inlined but no .safetensors / .pth — load_kokoro_v2
        // should fail cleanly since there's nothing to load weights from.
        let tmp = std::env::temp_dir().join("blazr_tts_loader_no_weights");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("config.json"), br#"{"vocab": {"a": 1}}"#).unwrap();
        let device = ServerRuntime::default_device();
        assert!(load_tts_from_dir(&tmp, VoxCpm2SynthOptions::default(), &device).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
