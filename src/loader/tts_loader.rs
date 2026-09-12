//! TTS model loader.
//!
//! Parses `--tts-model NAME=PATH` and builds a [`TtsBundle`]. The directory's
//! `config.json` decides the engine:
//!
//! - a VoxCPM2 config (it carries `dit_config`) loads a
//!   [`VoxCpm2Engine`] — voice cloning from the reference recordings in
//!   `voices/`, on CUDA when this binary was built with it;
//! - a Kokoro config (`vocab.json` or an inlined `vocab`, safetensors
//!   weights) loads a [`KokoroEngine`] on the CPU;
//! - no `config.json` at all gives a scaffolding bundle whose `synthesize`
//!   returns `NotImplemented`.
//!
//! All engine logic lives in boostr; this file only recognises the layout and
//! picks the device.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use colored::Colorize;

use boostr::model::audio::kokoro::{load_kokoro_v2, KokoroPhonemeVocab};
use boostr::model::audio::voxcpm::VoxCpm2Weights;
use boostr::runtime::cpu::{CpuDevice, CpuRuntime};
use boostr::Runtime;
use boostr_audio::kokoro::{KokoroEngine, VoiceResolver};
use boostr_audio::voxcpm::{VoxCpm2Engine, VoxCpm2LoadOptions, VoxCpm2SynthOptions};
use boostr_audio::{default_kokoro_voices, TtsBundle};

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
    if is_voxcpm2_config(&config_path)? {
        return load_voxcpm2(path, &config_path);
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

/// Whether `config.json` is a VoxCPM2 config: it carries the DiT sub-model's
/// config, which no Kokoro config has.
fn is_voxcpm2_config(config_path: &Path) -> Result<bool> {
    let text = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", config_path.display()))?;
    Ok(value.get("dit_config").is_some())
}

/// The one file in `dir` with extension `ext`, if exactly one exists. Two is
/// an error: the loader cannot guess which stack to serve.
fn single_file_with_ext(dir: &Path, ext: &str) -> Result<Option<PathBuf>> {
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

/// The one `.safetensors` file in `dir`, required (unlike
/// [`single_file_with_ext`], which allows zero). Mirrors its
/// exactly-one-or-error rule: an `adapters/` directory that exists but is
/// empty, or holds more than one file, can't be resolved without a caller
/// saying which adapter to apply.
fn single_adapter_file(dir: &Path) -> Result<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
        .collect();
    found.sort();
    if found.len() == 1 {
        return Ok(found.remove(0));
    }
    if found.is_empty() {
        return Err(anyhow!(
            "{} exists but holds no .safetensors file; add exactly one LoRA adapter or remove \
             the empty adapters directory",
            dir.display()
        ));
    }
    Err(anyhow!(
        "{} holds {} .safetensors files; a bundle applies one LoRA adapter — split them into \
         separate --tts-model NAME=DIR bundles",
        dir.display(),
        found.len()
    ))
}

/// This bundle's LoRA adapter, if `dir/adapters/` exists: exactly one
/// `.safetensors` file, applied to the model once at load time. Missing
/// `adapters/` means the base model, unchanged.
///
/// One adapted model per bundle: serving several adapters means repeating
/// `--tts-model NAME=DIR` with several bundles, each carrying its own
/// `adapters/` directory. Per-request adapter switching is not offered —
/// LoRA is folded into the weights, not applied per call.
fn discover_adapter(dir: &Path) -> Result<Option<PathBuf>> {
    let adapters_dir = dir.join("adapters");
    if !adapters_dir.is_dir() {
        return Ok(None);
    }
    Ok(Some(single_adapter_file(&adapters_dir)?))
}

/// Build the VoxCPM2 engine from a model directory.
///
/// ```text
/// PATH/
///   config.json               — VoxCPM2 config (has `dit_config`)
///   model.safetensors         — dense checkpoint, OR
///   <one>.tcf | <one>.gguf    — single-file quantized stack (TCF preferred)
///   tokenizer.json
///   audiovae.pth | audiovae.safetensors
///   voices/                   — optional reference recordings, one per voice id
///   adapters/<one>.safetensors — optional LoRA adapter, folded in at load
/// ```
///
/// `voices/` is optional: absent, or present but empty, the bundle still
/// loads and renders zero-shot — the adapted voice when `adapters/` is
/// present, the base model's voice otherwise.
///
/// `adapters/` holds AT MOST ONE adapter: this bundle serves one adapted
/// model for its lifetime, chosen at load time, not per request. Serving
/// several adapters means several `--tts-model NAME=DIR` bundles, each with
/// its own `adapters/` directory.
fn load_voxcpm2(dir: &Path, config_path: &Path) -> Result<TtsBundle> {
    let weights = if let Some(tcf) = single_file_with_ext(dir, "tcf")? {
        VoxCpm2Weights::Tcf {
            path: tcf,
            config: config_path.to_path_buf(),
        }
    } else if let Some(gguf) = single_file_with_ext(dir, "gguf")? {
        VoxCpm2Weights::Gguf {
            path: gguf,
            config: config_path.to_path_buf(),
        }
    } else {
        VoxCpm2Weights::Checkpoint(dir.to_path_buf())
    };
    let audiovae = ["audiovae.pth", "audiovae.safetensors"]
        .iter()
        .map(|name| dir.join(name))
        .find(|p| p.is_file())
        .ok_or_else(|| {
            anyhow!(
                "no audiovae.pth or audiovae.safetensors in {}: VoxCPM2 ships its AudioVAE as a separate file",
                dir.display()
            )
        })?;
    let voices_dir = dir.join("voices");
    // `Some` only when `voices/` actually exists: `VoxCpm2Engine::load`
    // treats `None` and a missing/empty directory identically (zero
    // encoded voices, zero-shot still available), so this guard is a
    // clarity choice, not a correctness one.
    let voices_dir = voices_dir.is_dir().then_some(voices_dir.as_path());
    let adapter = discover_adapter(dir)?;

    let load_options = VoxCpm2LoadOptions {
        dtype: None,
        synth: VoxCpm2SynthOptions::default(),
        adapter,
    };

    #[cfg(feature = "cuda")]
    let engine: Arc<dyn boostr_audio::TtsEngine> = {
        use boostr::runtime::cuda::{CudaDevice, CudaRuntime};
        let device = CudaDevice::new(0);
        let client = Arc::new(CudaRuntime::default_client(&device));
        let engine = VoxCpm2Engine::<CudaRuntime>::load(
            &weights,
            &audiovae,
            voices_dir,
            &device,
            client,
            load_options,
        )
        .map_err(|e| anyhow!("loading VoxCPM2 from {}: {e}", dir.display()))?;
        log_adapter_report(dir, engine.adapter());
        Arc::new(engine)
    };
    #[cfg(not(feature = "cuda"))]
    let engine: Arc<dyn boostr_audio::TtsEngine> = {
        let device = CpuDevice::new();
        let client = Arc::new(CpuRuntime::default_client(&device));
        let engine = VoxCpm2Engine::<CpuRuntime>::load(
            &weights,
            &audiovae,
            voices_dir,
            &device,
            client,
            load_options,
        )
        .map_err(|e| anyhow!("loading VoxCPM2 from {}: {e}", dir.display()))?;
        log_adapter_report(dir, engine.adapter());
        Arc::new(engine)
    };

    let voices = engine.voices();
    let sample_rate = engine.sample_rate();
    Ok(TtsBundle::scaffolding(voices, sample_rate).with_engine(engine))
}

/// Log the one line a loaded adapter earns: rank, alpha, targets, counts.
/// `None` (no `adapters/` directory) logs nothing — the base model needs no
/// announcement.
fn log_adapter_report(
    dir: &Path,
    report: Option<&boostr::model::audio::voxcpm::LoraAdapterReport>,
) {
    if let Some(report) = report {
        eprintln!(
            "  {} applied LoRA adapter from {}/adapters: rank={} alpha={} targets={:?} \
             projections_adapted={} tensors_loaded={}",
            "+".green(),
            dir.display(),
            report.rank,
            report.alpha,
            report.targets,
            report.projections_adapted,
            report.tensors_loaded,
        );
    }
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
