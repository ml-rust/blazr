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

use boostr::model::audio::kokoro::{load_kokoro_v2, KokoroPhonemeVocab};
use boostr::runtime::cpu::{CpuDevice, CpuRuntime};
use boostr::Runtime;
use voxr::kokoro::{KokoroEngine, KokoroG2pModels, VoiceResolver};
use voxr::{default_kokoro_voices, TtsBundle};

/// Builds a Kokoro engine from `path`'s `config.json` + weights and wraps it
/// in a scaffolding bundle. Kokoro ships the vocab inlined in `config.json`
/// under a `"vocab"` key; a separate `vocab.json` at `vocab_path` takes
/// precedence when it exists (custom variants).
pub(super) fn load_kokoro(path: &Path, config_path: &Path, vocab_path: &Path) -> Result<TtsBundle> {
    let device = Arc::new(CpuDevice::new());
    let client = Arc::new(CpuRuntime::default_client(&device));

    let model = load_kokoro_v2::<CpuRuntime, _>(client.as_ref(), path, device.as_ref())
        .map_err(|e| anyhow!("loading kokoro weights from {}: {e}", path.display()))?;
    let vocab_source = if vocab_path.exists() {
        vocab_path
    } else {
        config_path
    };
    let vocab = KokoroPhonemeVocab::from_json_file(vocab_source)
        .map_err(|e| anyhow!("loading vocab from {}: {e}", vocab_source.display()))?;

    // Voice resolver: prefer `{model_dir}/voices/`, then fall back to the
    // system defaults (env var / bundled assets) for unknown ids.
    let voices_dir = path.join("voices");
    let resolver = if voices_dir.is_dir() {
        VoiceResolver::new().with_asset_dir(&voices_dir)
    } else {
        VoiceResolver::new()
    };

    let g2p = KokoroG2pModels::resolve(path, None, None)
        .and_then(|models| models.load())
        .map_err(|e| anyhow!("building Kokoro G2P: {e}"))?;
    let engine = KokoroEngine {
        model,
        vocab,
        resolver,
        client,
        device,
        min_frames_per_phoneme: 1,
        g2p,
    };
    let sample_rate = engine.sample_rate();

    Ok(TtsBundle::scaffolding(default_kokoro_voices(), sample_rate).with_engine(Arc::new(engine)))
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
