//! Picks a TTS engine for a model directory.
//!
//! The directory's `config.json` decides the engine:
//!
//! - a VoxCPM2 config (it carries `dit_config`) loads a `VoxCpm2Engine` —
//!   voice cloning from the reference recordings in `voices/`, on CUDA when
//!   this binary was built with it;
//! - a VITS config (`model_type` is `vits`, as in every MMS-TTS checkpoint)
//!   loads a `VitsEngine` on the CPU, one voice per checkpoint;
//! - a Kokoro config (`vocab.json` or an inlined `vocab`, safetensors
//!   weights) loads a `KokoroEngine` on the CPU;
//! - no `config.json`, but one `.gguf` whose `general.architecture` is
//!   `voxcpm2`, loads a `VoxCpm2Engine` from that file alone: a GGUF written
//!   by compressr embeds its config, tokenizer and AudioVAE;
//! - nothing else gives a scaffolding bundle whose `synthesize` returns
//!   `NotImplemented`.
//!
//! All engine logic lives in boostr; this file only recognises the layout
//! and picks the device.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use colored::Colorize;

use boostr::format::gguf::Gguf;
use boostr_audio::tts::{is_vits_config, load_vits_from_dir};
use boostr_audio::voxcpm::VoxCpm2SynthOptions;
use boostr_audio::{default_kokoro_voices, TtsBundle};

use super::kokoro::load_kokoro;
use super::voxcpm::load_voxcpm2;

/// Kokoro's native sample rate (used as the sample rate advertised by a
/// scaffolding bundle when no real checkpoint is present).
const KOKORO_SAMPLE_RATE: u32 = 24_000;

/// Load a TTS bundle from an HF-style directory.
///
/// If `config.json` + weights are present, a real engine is built and
/// attached. Missing pieces downgrade gracefully: no config → scaffolding
/// only; config present but safetensors fails → returns the load error.
///
/// `synth` carries the per-request generation defaults (batching included)
/// for a VoxCPM2 engine. A Kokoro bundle ignores it — Kokoro has no batching
/// knobs of its own.
pub fn load_tts_from_dir<P: AsRef<Path>>(path: P, synth: VoxCpm2SynthOptions) -> Result<TtsBundle> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(anyhow!("tts model directory not found: {}", path.display()));
    }

    let config_path = path.join("config.json");
    let vocab_path = path.join("vocab.json");

    // No config.json: a lone VoxCPM2 GGUF still loads, since compressr
    // embeds everything the engine needs. Anything else falls back to the
    // scaffolding bundle and lets the endpoint return 503.
    if !config_path.exists() {
        if let Some(gguf) = single_file_with_ext(path, "gguf")? {
            if is_voxcpm2_gguf(&gguf)? {
                return load_voxcpm2(path, None, synth);
            }
        }
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
    if is_voxcpm2_config(&config_path)? {
        return load_voxcpm2(path, Some(&config_path), synth);
    }
    if is_vits_config(&config_path).with_context(|| format!("reading {}", config_path.display()))? {
        return load_vits_from_dir(path)
            .with_context(|| format!("loading VITS model from {}", path.display()));
    }
    load_kokoro(path, &config_path, &vocab_path)
}

/// Whether `config.json` is a VoxCPM2 config: it carries the DiT sub-model's
/// config, which no Kokoro config has.
fn is_voxcpm2_config(config_path: &Path) -> Result<bool> {
    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", config_path.display()))?;
    Ok(value.get("dit_config").is_some())
}

/// Whether a GGUF names VoxCPM2 as its `general.architecture`. Both
/// compressr's and cstr's files do; only compressr's embeds the config, and
/// the engine's loader says so when it is missing.
fn is_voxcpm2_gguf(path: &Path) -> Result<bool> {
    let gguf = Gguf::open(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(gguf
        .metadata()
        .architecture()
        .is_some_and(|arch| arch.eq_ignore_ascii_case("voxcpm2")))
}

/// The one file in `dir` with extension `ext`, if exactly one exists. Two is
/// an error: the loader cannot guess which stack to serve. Shared with the
/// VoxCPM2 loader, which uses it to find a single `.tcf` or `.gguf` stack.
pub(super) fn single_file_with_ext(dir: &Path, ext: &str) -> Result<Option<PathBuf>> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some(ext))
        .collect();
    found.sort();
    match found.len() {
        0 => Ok(None),
        1 => Ok(found.pop()),
        n => Err(anyhow!(
            "{} holds {n} .{ext} files; a TTS model directory carries one transformer stack",
            dir.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_errors_on_missing_dir() {
        assert!(load_tts_from_dir(
            "/nonexistent-kokoro-path-xyz",
            VoxCpm2SynthOptions::default()
        )
        .is_err());
    }

    #[test]
    fn load_returns_scaffolding_when_config_missing() {
        let tmp = std::env::temp_dir().join("blazr_tts_loader_no_config");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let bundle = load_tts_from_dir(&tmp, VoxCpm2SynthOptions::default()).unwrap();
        assert!(!bundle.has_engine());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A GGUF v3 image with no tensors and one `general.architecture`
    /// string. Enough for the dispatch probe, which reads metadata only.
    fn gguf_with_architecture(arch: &str) -> Vec<u8> {
        fn put_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        put_str(&mut out, "general.architecture");
        out.extend_from_slice(&8u32.to_le_bytes());
        put_str(&mut out, arch);
        let aligned = out.len().div_ceil(32) * 32;
        out.resize(aligned, 0);
        out
    }

    /// No config.json, one VoxCPM2 GGUF: dispatch reaches the VoxCPM2
    /// loader (which then fails on the empty file) instead of handing back
    /// a scaffolding bundle.
    #[test]
    fn a_lone_voxcpm2_gguf_is_dispatched_without_a_config() {
        let tmp = std::env::temp_dir().join("blazr_tts_loader_lone_gguf");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let gguf = tmp.join("model.gguf");
        std::fs::write(&gguf, gguf_with_architecture("voxcpm2")).unwrap();
        assert!(is_voxcpm2_gguf(&gguf).unwrap());
        assert!(load_tts_from_dir(&tmp, VoxCpm2SynthOptions::default()).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A `model_type: vits` config reaches the VITS loader, which fails on
    /// the missing tokenizer and weights, instead of falling through to the
    /// Kokoro loader.
    #[test]
    fn a_vits_config_is_dispatched_to_the_vits_loader() {
        let tmp = std::env::temp_dir().join("blazr_tts_loader_vits_config");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let config = tmp.join("config.json");
        std::fs::write(&config, r#"{"model_type": "vits"}"#).unwrap();
        assert!(is_vits_config(&config).unwrap());
        let Err(err) = load_tts_from_dir(&tmp, VoxCpm2SynthOptions::default()) else {
            panic!("a VITS directory without weights loaded");
        };
        assert!(
            format!("{err:#}").contains("loading VITS model"),
            "unexpected error: {err:#}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// No config.json, one GGUF of another architecture: scaffolding, as
    /// with no GGUF at all.
    #[test]
    fn a_lone_foreign_gguf_still_gives_scaffolding() {
        let tmp = std::env::temp_dir().join("blazr_tts_loader_foreign_gguf");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let gguf = tmp.join("model.gguf");
        std::fs::write(&gguf, gguf_with_architecture("llama")).unwrap();
        assert!(!is_voxcpm2_gguf(&gguf).unwrap());
        let bundle = load_tts_from_dir(&tmp, VoxCpm2SynthOptions::default()).unwrap();
        assert!(!bundle.has_engine());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
