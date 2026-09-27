//! Builds the VoxCPM2 engine from a model directory.
//!
//! ```text
//! PATH/
//!   config.json               — VoxCPM2 config (has `dit_config`); optional
//!                               when the .gguf embeds one, required otherwise
//!   model.safetensors         — dense checkpoint, OR
//!   <one>.tcf | <one>.gguf    — single-file quantized stack (TCF preferred)
//!   tokenizer.json            — optional when the .gguf embeds one
//!   audiovae.pth | audiovae.safetensors — the separate AudioVAE; optional
//!                               when the .tcf/.gguf embeds one (compressr
//!                               writes it under `vae.`), required otherwise
//!   voices/                   — optional reference recordings, one per voice id
//!   adapters/<one>.safetensors — optional LoRA adapter, folded in at load
//! ```
//!
//! A `.gguf` written by compressr needs nothing beside it: config,
//! tokenizer and AudioVAE are all inside. `voices/` and `adapters/` stay
//! directory-level. `config_path` is `Some` only when `config.json` exists;
//! a TCF needs it, and is refused without it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use colored::Colorize;

use boostr::model::audio::voxcpm::VoxCpm2Weights;
use boostr_audio::voxcpm::{VoxCpm2Engine, VoxCpm2LoadOptions, VoxCpm2SynthOptions};
use boostr_audio::TtsBundle;

use super::dispatch::single_file_with_ext;

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

/// The AudioVAE file is looked for and passed when present; the engine
/// decides — an embedded copy wins, and a bundle with neither is refused by
/// the loader with an error naming both ways to supply it.
///
/// `voices/` is optional: absent, or present but empty, the bundle still
/// loads and renders zero-shot — the adapted voice when `adapters/` is
/// present, the base model's voice otherwise.
///
/// `adapters/` holds AT MOST ONE adapter: this bundle serves one adapted
/// model for its lifetime, chosen at load time, not per request. Serving
/// several adapters means several `--tts-model NAME=DIR` bundles, each with
/// its own `adapters/` directory.
pub(super) fn load_voxcpm2(
    dir: &Path,
    config_path: Option<&Path>,
    synth: VoxCpm2SynthOptions,
) -> Result<TtsBundle> {
    let weights = if let Some(tcf) = single_file_with_ext(dir, "tcf")? {
        let config = config_path.ok_or_else(|| {
            anyhow!(
                "{} holds no config.json; a TCF carries no config, so the checkpoint's \
                 config.json must sit beside it",
                dir.display()
            )
        })?;
        VoxCpm2Weights::Tcf {
            path: tcf,
            config: config.to_path_buf(),
        }
    } else if let Some(gguf) = single_file_with_ext(dir, "gguf")? {
        VoxCpm2Weights::Gguf {
            path: gguf,
            config: config_path.map(Path::to_path_buf),
        }
    } else {
        VoxCpm2Weights::Checkpoint(dir.to_path_buf())
    };
    let audiovae = ["audiovae.pth", "audiovae.safetensors"]
        .iter()
        .map(|name| dir.join(name))
        .find(|p| p.is_file());
    let voices_dir = dir.join("voices");
    // `Some` only when `voices/` actually exists: `VoxCpm2Engine::load`
    // treats `None` and a missing/empty directory identically (zero
    // encoded voices, zero-shot still available), so this guard is a
    // clarity choice, not a correctness one.
    let voices_dir = voices_dir.is_dir().then_some(voices_dir.as_path());
    let adapter = discover_adapter(dir)?;

    // The AudioVAE decoder runs in F16 on CUDA: every decoder kernel has a
    // half variant, the transposed convolutions fold from an F32 product,
    // and the output stays within the F32 decoder's tolerance while the
    // decoder's weights take half the memory. The CPU backend keeps the
    // checkpoint's F32.
    #[cfg(feature = "cuda")]
    let vae_decoder_dtype = Some(boostr::DType::F16);
    #[cfg(not(feature = "cuda"))]
    let vae_decoder_dtype = None;

    let load_options = VoxCpm2LoadOptions {
        dtype: None,
        vae_decoder_dtype,
        synth,
        adapter,
    };

    #[cfg(feature = "cuda")]
    let engine: Arc<dyn boostr_audio::TtsEngine> = {
        use boostr::runtime::cuda::{CudaDevice, CudaRuntime};
        use boostr::Runtime;
        let device = CudaDevice::new(0);
        let client = Arc::new(CudaRuntime::default_client(&device));
        let engine = VoxCpm2Engine::<CudaRuntime>::load(
            &weights,
            audiovae.as_deref(),
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
        use boostr::runtime::cpu::{CpuDevice, CpuRuntime};
        use boostr::Runtime;
        let device = CpuDevice::new();
        let client = Arc::new(CpuRuntime::default_client(&device));
        let engine = VoxCpm2Engine::<CpuRuntime>::load(
            &weights,
            audiovae.as_deref(),
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
