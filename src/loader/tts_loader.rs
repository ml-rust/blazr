//! TTS model loader.
//!
//! Parses `--tts-model NAME=PATH` and builds a [`TtsBundle`]. When the target
//! directory contains a full Kokoro checkpoint (`config.json`, `vocab.json`,
//! safetensors weights), loads everything and attaches a live
//! [`KokoroEngine`] so `/v1/audio/speech` can actually synthesize. Otherwise
//! falls back to a scaffolding bundle whose `synthesize` returns
//! `NotImplemented` (same behavior as before the Kokoro port landed).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use colored::Colorize;

use boostr::model::audio::kokoro::{
    load_kokoro_v2, KokoroEngine, KokoroPhonemeVocab, VoiceResolver,
};
use boostr::model::audio::{default_kokoro_voices, TtsBundle};
use boostr::runtime::cpu::{CpuDevice, CpuRuntime};
use boostr::Runtime;

/// Kokoro's native sample rate (used as the sample rate advertised by a
/// scaffolding bundle when no real checkpoint is present).
const KOKORO_SAMPLE_RATE: u32 = 24_000;

/// Load a TTS bundle from an HF-style directory.
///
/// The directory layout:
///
/// ```text
/// PATH/
///   config.json          — KokoroConfig
///   vocab.json           — phoneme → id map
///   model.safetensors    — weights (may be sharded)
///   voices/              — optional per-voice .safetensors / .pt files
/// ```
///
/// If `config.json` + weights are present, a real engine is built and
/// attached. Missing pieces downgrade gracefully: no config → scaffolding
/// only; config present but safetensors fails → returns the load error.
pub fn load_tts_from_dir<P: AsRef<Path>>(path: P) -> Result<TtsBundle> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(anyhow!("tts model directory not found: {}", path.display()));
    }

    let config_path = path.join("config.json");
    let vocab_path = path.join("vocab.json");
    let voices_dir = path.join("voices");

    // If config.json is missing, we can't build an engine — fall back to the
    // scaffolding bundle and let the endpoint return 503.
    if !config_path.exists() {
        eprintln!(
            "  {} no config.json in {} — TTS bundle will return 503 on synthesis",
            "!".yellow(),
            path.display(),
        );
        return Ok(TtsBundle::scaffolding(
            default_kokoro_voices(),
            KOKORO_SAMPLE_RATE,
        ));
    }
    let device = Arc::new(CpuDevice::new());
    let client = Arc::new(CpuRuntime::default_client(&device));

    let model = load_kokoro_v2::<CpuRuntime, _>(client.as_ref(), path, device.as_ref())
        .map_err(|e| anyhow!("loading kokoro weights from {}: {e}", path.display()))?;
    // Kokoro ships the vocab inlined in `config.json` under a `"vocab"` key.
    // A separate `vocab.json` takes precedence when present (custom variants).
    let vocab_source = if vocab_path.exists() {
        vocab_path.clone()
    } else {
        config_path.clone()
    };
    let vocab = KokoroPhonemeVocab::from_json_file(&vocab_source)
        .map_err(|e| anyhow!("loading vocab from {}: {e}", vocab_source.display()))?;

    // Voice resolver: prefer `{model_dir}/voices/`, then fall back to the
    // system defaults (env var / bundled assets) for unknown ids.
    let resolver = if voices_dir.is_dir() {
        VoiceResolver::new().with_asset_dir(&voices_dir)
    } else {
        VoiceResolver::new()
    };

    let engine = KokoroEngine {
        model,
        vocab,
        resolver,
        client,
        device,
        min_frames_per_phoneme: 1,
    };
    let sample_rate = engine.sample_rate();

    Ok(TtsBundle::scaffolding(default_kokoro_voices(), sample_rate).with_engine(Arc::new(engine)))
}

/// Parse `--tts-model NAME=PATH` into `(name, path)`.
pub fn parse_tts_model_arg(arg: &str) -> Result<(String, PathBuf)> {
    let (name, path) = arg
        .split_once('=')
        .ok_or_else(|| anyhow!("--tts-model expects NAME=PATH, got: {arg}"))?;
    let name = name.trim();
    let path = path.trim();
    if name.is_empty() || path.is_empty() {
        return Err(anyhow!("--tts-model NAME and PATH must both be non-empty"));
    }
    Ok((name.to_string(), PathBuf::from(path)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_name_path() {
        let (n, p) = parse_tts_model_arg("kokoro=/models/kokoro").unwrap();
        assert_eq!(n, "kokoro");
        assert_eq!(p, PathBuf::from("/models/kokoro"));
    }

    #[test]
    fn rejects_missing_eq() {
        assert!(parse_tts_model_arg("/p").is_err());
    }

    #[test]
    fn rejects_empty_parts() {
        assert!(parse_tts_model_arg("=/p").is_err());
        assert!(parse_tts_model_arg("n=").is_err());
    }

    #[test]
    fn load_errors_on_missing_dir() {
        assert!(load_tts_from_dir("/nonexistent-kokoro-path-xyz").is_err());
    }

    #[test]
    fn load_returns_scaffolding_when_config_missing() {
        let tmp = std::env::temp_dir().join("blazr_tts_loader_no_config");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let bundle = load_tts_from_dir(&tmp).unwrap();
        assert!(!bundle.has_engine());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn load_errors_when_config_present_but_weights_missing() {
        // Config + vocab-inlined but no .safetensors / .pth — load_kokoro_v2
        // should fail cleanly since there's nothing to load weights from.
        let tmp = std::env::temp_dir().join("blazr_tts_loader_no_weights");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("config.json"), br#"{"vocab": {"a": 1}}"#).unwrap();
        assert!(load_tts_from_dir(&tmp).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
