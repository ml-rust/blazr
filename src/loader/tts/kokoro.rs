//! Builds a [`KokoroEngine`] from a model directory.
//!
//! ```text
//! PATH/
//!   config.json          — KokoroConfig
//!   vocab.json           — phoneme → id map
//!   model.safetensors    — weights (may be sharded)
//!   voices/               — optional per-voice .safetensors / .pt files
//!   g2p/tagger/           — spaCy tagger (tagger.safetensors, tagger.json),
//!                           required — see KokoroG2pModels::resolve
//!   g2p/oov/              — optional g2p_lm out-of-vocabulary fallback
//! ```

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use voxr::kokoro::KokoroEngine;
use voxr::{default_kokoro_voices, TtsBundle};

/// Builds a Kokoro engine from `path`'s `config.json` + weights and wraps it
/// in a scaffolding bundle. Kokoro ships the vocab inlined in `config.json`
/// under a `"vocab"` key; a separate `vocab.json` beside it takes
/// precedence when it exists (custom variants).
pub(super) fn load_kokoro(path: &Path) -> Result<TtsBundle> {
    let engine = load_kokoro_engine(path)?;
    let sample_rate = engine.sample_rate();
    Ok(TtsBundle::scaffolding(default_kokoro_voices(), sample_rate).with_engine(Arc::new(engine)))
}

/// Builds a Kokoro engine from `path` on the CPU: the weights, the phoneme
/// vocab, the voices under `voices/` (then the bundled defaults), and the
/// G2P models under `g2p/`.
pub fn load_kokoro_engine(path: &Path) -> Result<KokoroEngine> {
    KokoroEngine::from_dir(path, None)
        .map_err(|e| anyhow!("loading Kokoro from {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use voxr::voxcpm::VoxCpm2SynthOptions;

    use super::super::load_tts_from_dir;

    #[test]
    fn load_errors_when_config_present_but_weights_missing() {
        // Config + vocab-inlined but no .safetensors / .pth — load_kokoro_v2
        // should fail cleanly since there's nothing to load weights from.
        let tmp = std::env::temp_dir().join("blazr_tts_loader_no_weights");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("config.json"), br#"{"vocab": {"a": 1}}"#).unwrap();
        assert!(load_tts_from_dir(&tmp, VoxCpm2SynthOptions::default()).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
