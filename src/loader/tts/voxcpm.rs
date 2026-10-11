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
use voxr::voxcpm::{VoxCpm2Engine, VoxCpm2LoadOptions, VoxCpm2SynthOptions};
use voxr::TtsBundle;

use super::dispatch::single_file_with_ext;
use crate::server::handlers::ServerDevice;

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

/// Where a VoxCPM2 engine loads from: the weights, the separate AudioVAE,
/// the voices directory and the one LoRA adapter.
pub struct Voxcpm2Source {
    weights: VoxCpm2Weights,
    audiovae: Option<PathBuf>,
    voices_dir: Option<PathBuf>,
    adapter: Option<PathBuf>,
    /// The directory or file, for messages.
    origin: PathBuf,
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
pub fn voxcpm2_source(dir: &Path, config_path: Option<&Path>) -> Result<Voxcpm2Source> {
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
    // `Some` only when `voices/` actually exists: `VoxCpm2Engine::load`
    // treats `None` and a missing/empty directory identically (zero
    // encoded voices, zero-shot still available), so this guard is a
    // clarity choice, not a correctness one.
    let voices_dir = Some(dir.join("voices")).filter(|d| d.is_dir());
    let adapter = discover_adapter(dir)?;
    Ok(Voxcpm2Source {
        weights,
        audiovae,
        voices_dir,
        adapter,
        origin: dir.to_path_buf(),
    })
}

/// A VoxCPM2 checkpoint at `path`: a directory as [`voxcpm2_source`]
/// reads it, or a single `.gguf` file that embeds its config, tokenizer
/// and AudioVAE.
pub fn voxcpm2_source_at(path: &Path) -> Result<Voxcpm2Source> {
    if path.is_file() {
        return Ok(Voxcpm2Source {
            weights: VoxCpm2Weights::Gguf {
                path: path.to_path_buf(),
                config: None,
            },
            audiovae: None,
            voices_dir: None,
            adapter: None,
            origin: path.to_path_buf(),
        });
    }
    let config = path.join("config.json");
    voxcpm2_source(path, config.is_file().then_some(config.as_path()))
}

impl Voxcpm2Source {
    /// The load options: the AudioVAE decoder in F16 on CUDA, the
    /// checkpoint's own dtype elsewhere, and this source's adapter.
    fn options(&self, synth: VoxCpm2SynthOptions, cuda: bool) -> VoxCpm2LoadOptions {
        // The AudioVAE decoder runs in F16 on CUDA: every decoder kernel has a
        // half variant, the transposed convolutions fold from an F32 product,
        // and the output stays within the F32 decoder's tolerance while the
        // decoder's weights take half the memory. The CPU backend keeps the
        // checkpoint's F32.
        #[cfg(feature = "cuda")]
        let vae_decoder_dtype = cuda.then_some(boostr::DType::F16);
        #[cfg(not(feature = "cuda"))]
        let vae_decoder_dtype = {
            let _ = cuda;
            None
        };
        VoxCpm2LoadOptions {
            dtype: None,
            vae_decoder_dtype,
            synth,
            adapter: self.adapter.clone(),
        }
    }

    /// Loads the engine on the CPU.
    pub fn load_cpu(
        &self,
        synth: VoxCpm2SynthOptions,
    ) -> Result<VoxCpm2Engine<boostr::runtime::cpu::CpuRuntime>> {
        use boostr::runtime::cpu::{CpuDevice, CpuRuntime};
        use boostr::Runtime;
        let device = CpuDevice::new();
        let client = Arc::new(CpuRuntime::default_client(&device));
        let engine = VoxCpm2Engine::<CpuRuntime>::load(
            &self.weights,
            self.audiovae.as_deref(),
            self.voices_dir.as_deref(),
            &device,
            client,
            self.options(synth, false),
        )
        .map_err(|e| anyhow!("loading VoxCPM2 from {}: {e}", self.origin.display()))?;
        log_adapter_report(&self.origin, engine.adapter());
        Ok(engine)
    }

    /// Loads the engine on CUDA device `index`.
    #[cfg(feature = "cuda")]
    pub fn load_cuda(
        &self,
        synth: VoxCpm2SynthOptions,
        index: usize,
    ) -> Result<VoxCpm2Engine<boostr::runtime::cuda::CudaRuntime>> {
        use boostr::runtime::cuda::{CudaDevice, CudaRuntime};
        use boostr::Runtime;
        let device = CudaDevice::new(index);
        let client = Arc::new(CudaRuntime::default_client(&device));
        let engine = VoxCpm2Engine::<CudaRuntime>::load(
            &self.weights,
            self.audiovae.as_deref(),
            self.voices_dir.as_deref(),
            &device,
            client,
            self.options(synth, true),
        )
        .map_err(|e| anyhow!("loading VoxCPM2 from {}: {e}", self.origin.display()))?;
        log_adapter_report(&self.origin, engine.adapter());
        Ok(engine)
    }
}

/// The VoxCPM2 bundle a `--tts-model` directory serves, its engine on the
/// server's `device`: a CUDA device when this binary is built with CUDA,
/// else the CPU.
pub(super) fn load_voxcpm2(
    dir: &Path,
    config_path: Option<&Path>,
    synth: VoxCpm2SynthOptions,
    device: &ServerDevice,
) -> Result<TtsBundle> {
    let source = voxcpm2_source(dir, config_path)?;
    #[cfg(feature = "cuda")]
    let engine: Arc<dyn voxr::TtsEngine> =
        Arc::new(source.load_cuda(synth, boostr::runtime::Device::id(device))?);
    #[cfg(not(feature = "cuda"))]
    let engine: Arc<dyn voxr::TtsEngine> = {
        let _ = device;
        Arc::new(source.load_cpu(synth)?)
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
