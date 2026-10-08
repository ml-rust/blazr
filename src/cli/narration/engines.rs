//! The speech engine a speak run resolves to, found on disk and loaded:
//! Kokoro, a VITS (MMS-TTS) model, or VoxCPM2 on the device the run picks.
//! A command needs no model.

use std::path::{Path, PathBuf};

use boostr::runtime::cpu::CpuRuntime;
use voxr::kokoro::KokoroEngine;
use voxr::tts::{is_vits_config, VitsEngine};
use voxr::voxcpm::{VoxCpm2Engine, VoxCpm2SynthOptions};
use voxr_narrate::{
    resolve, speak_script, CommandError, EngineChoice, Hearing, KokoroSpeech, NarrationRuns,
    Result, ScriptStep, SpeakOpts, SpeakTarget, SpokenScript, TtsConfig, VitsSpeech, VoxcpmSpeech,
};

use super::config::MotionConfig;
use super::hearing::Placement;
use super::kinds::{missing_file, ModelKind, VOXCPM_ID};
use super::models::ModelSearch;
use crate::loader::{load_kokoro_engine, voxcpm2_source_at};

/// A VoxCPM2 engine on either runtime.
pub enum LoadedVoxcpm {
    /// On the CPU.
    Cpu(VoxCpm2Engine<CpuRuntime>),
    /// On CUDA device 0.
    #[cfg(feature = "cuda")]
    Cuda(VoxCpm2Engine<boostr::runtime::cuda::CudaRuntime>),
}

/// The engines a run speaks with. A run loads at most one.
#[derive(Default)]
pub struct Engines {
    kokoro: Option<KokoroEngine>,
    vits: Option<(String, VitsEngine)>,
    voxcpm: Option<LoadedVoxcpm>,
    /// The VoxCPM2 checkpoint this run downloaded first. The skill speaks
    /// such a run with no checkpoint named: its cache keys and notes say
    /// so.
    pub downloaded: bool,
}

/// The VoxCPM2 checkpoint: the config's `tts.checkpoint` (a directory or a
/// `.gguf` file), else `openbmb/VoxCPM2` found on disk. The error is why
/// none is there, with the next action.
pub fn voxcpm_checkpoint(
    config: &MotionConfig,
    search: &ModelSearch,
    cwd: &Path,
) -> std::result::Result<PathBuf, String> {
    let Some(set) = &config.tts.checkpoint else {
        return search.find(VOXCPM_ID, ModelKind::Voxcpm);
    };
    let path = resolve(cwd, Path::new(set));
    if !path.exists() {
        return Err(format!(
            "tts.checkpoint in the config is \"{set}\", which does not exist. Point it at a VoxCPM2 checkpoint directory or a .gguf file"
        ));
    }
    match missing_file(&path, ModelKind::Voxcpm) {
        None => Ok(path),
        Some(file) => Err(format!(
            "tts.checkpoint in the config is \"{set}\", which lacks {file}. Point it at a full VoxCPM2 checkpoint directory or a .gguf file"
        )),
    }
}

/// The refusal of a Hugging Face TTS model other than VITS, exit 3: a
/// caller runs it another way.
fn not_vits(model: &str) -> CommandError {
    CommandError::Unsupported(format!(
        "speak --model {model}: only VITS (MMS-TTS) checkpoints run natively, and {model} is not one. Run it with the skill's Python runner, or pass --command."
    ))
}

fn load_failed(what: &str, e: impl std::fmt::Display) -> CommandError {
    CommandError::Failed(format!("speak: loading {what}: {e:#}"))
}

/// Loads the engine `choice` speaks with. A Hugging Face model not on
/// disk is downloaded first. VoxCPM2's is too, unless the config's
/// `tts.checkpoint` names one that is not there.
pub fn load_engines(
    choice: &EngineChoice,
    search: &ModelSearch,
    config: &MotionConfig,
    checkpoint: &std::result::Result<PathBuf, String>,
    at: Placement,
) -> std::result::Result<Engines, CommandError> {
    let mut engines = Engines::default();
    match choice {
        EngineChoice::Command { .. } => {}
        EngineChoice::Kokoro { model, .. } => {
            let dir = search.obtain("speak", model, ModelKind::Kokoro, None)?;
            engines.kokoro = Some(load_kokoro_engine(&dir).map_err(|e| load_failed("Kokoro", e))?);
        }
        EngineChoice::Model { model } => {
            // The model's kind is read before its weights are downloaded:
            // only VITS runs here.
            if let Some(config) = search.config_of(model, ModelKind::Vits) {
                let vits = is_vits_config(&config).map_err(|e| load_failed(model, e))?;
                if !vits {
                    return Err(not_vits(model));
                }
            }
            let dir = search.obtain("speak", model, ModelKind::Vits, None)?;
            if !is_vits_config(&dir.join("config.json")).map_err(|e| load_failed(model, e))? {
                return Err(not_vits(model));
            }
            let engine = VitsEngine::from_dir(&dir).map_err(|e| load_failed(model, e))?;
            engines.vits = Some((model.clone(), engine));
        }
        EngineChoice::Voxcpm { .. } => {
            let path = match checkpoint {
                Ok(path) => path.clone(),
                Err(why) if config.tts.checkpoint.is_some() => {
                    return Err(CommandError::Failed(format!("speak: {why}.")));
                }
                Err(_) => {
                    let notice = format!(
                        "speak: no VoxCPM2 checkpoint on disk. Downloading it (about 4.7 GB) into {}. This happens once.",
                        search.download_dir().display()
                    );
                    let path =
                        search.obtain("speak", VOXCPM_ID, ModelKind::Voxcpm, Some(&notice))?;
                    engines.downloaded = true;
                    path
                }
            };
            let source = voxcpm2_source_at(&path).map_err(|e| load_failed("VoxCPM2", e))?;
            let synth = VoxCpm2SynthOptions::default();
            engines.voxcpm = Some(match at {
                Placement::Cpu => LoadedVoxcpm::Cpu(
                    source
                        .load_cpu(synth)
                        .map_err(|e| load_failed("VoxCPM2", e))?,
                ),
                #[cfg(feature = "cuda")]
                Placement::Cuda => LoadedVoxcpm::Cuda(
                    source
                        .load_cuda(synth, 0)
                        .map_err(|e| load_failed("VoxCPM2", e))?,
                ),
            });
        }
    }
    Ok(engines)
}

/// What a speak run speaks: the script, its options, the config and where
/// it writes.
pub struct RunInput<'a> {
    pub script: &'a str,
    pub opts: &'a SpeakOpts<'a>,
    pub tts: &'a TtsConfig<'a>,
    pub target: &'a SpeakTarget<'a>,
}

/// Speaks the script through the loaded engines, hearing each take with
/// `hearing`, running the user's command through `spawn`, and printing
/// each step on standard error as it happens.
pub fn run_speak<'a, H: Hearing>(
    engines: &'a Engines,
    hearing: H,
    spawn: impl FnMut(&str) -> Result<Option<i32>> + 'a,
    input: &RunInput<'_>,
) -> Result<SpokenScript> {
    // Each step prints as it happens, on standard error, as the skill's
    // `console.error` prints it.
    let mut runs = NarrationRuns::new(hearing, spawn).with_progress(|step: &ScriptStep| {
        if let Some(line) = step.message() {
            eprintln!("{line}");
        }
    });
    if let Some(k) = &engines.kokoro {
        runs = runs.with_kokoro(KokoroSpeech::new(k));
    }
    if let Some((id, v)) = &engines.vits {
        runs = runs.with_model(id.clone(), VitsSpeech::new(v)?);
    }
    let RunInput {
        script,
        opts,
        tts,
        target,
    } = *input;
    match &engines.voxcpm {
        None => speak_script(script, opts, tts, target, &mut runs),
        Some(LoadedVoxcpm::Cpu(e)) => {
            let mut runs = runs.with_voxcpm(VoxcpmSpeech::new(e));
            speak_script(script, opts, tts, target, &mut runs)
        }
        #[cfg(feature = "cuda")]
        Some(LoadedVoxcpm::Cuda(e)) => {
            let mut runs = runs.with_voxcpm(VoxcpmSpeech::new(e));
            speak_script(script, opts, tts, target, &mut runs)
        }
    }
}
