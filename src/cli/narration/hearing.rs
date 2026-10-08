//! The Whisper recognizers a run hears with: the base, the small, and the
//! one the config's `asr.model` names, found on disk and loaded on the
//! device the run picks.

use std::path::Path;

use boostr::runtime::cpu::{CpuDevice, CpuRuntime};
use voxr_narrate::{
    CommandError, Error, HeardWord, Recognizer, RecognizerSource, Result, Transcriber,
    WhisperHearing, WhisperTranscriber, DEFAULT_ASR_MODEL, SMALL_ASR_MODEL,
};

use crate::loader::hub::{ModelKind, ModelSearch};

/// Where the models run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// The CPU.
    Cpu,
    /// CUDA device 0.
    #[cfg(feature = "cuda")]
    Cuda,
}

impl Placement {
    /// `--device cpu` runs on the CPU. Otherwise a CUDA build runs on its
    /// first GPU when the driver sees one, and every other build on the
    /// CPU.
    pub fn pick(device: Option<&str>) -> Self {
        if device == Some("cpu") {
            return Self::Cpu;
        }
        #[cfg(feature = "cuda")]
        if super::gpu::gpu().is_some() {
            return Self::Cuda;
        }
        Self::Cpu
    }
}

/// A recognizer loaded on either runtime.
pub struct Loaded(Box<dyn Transcriber>);

impl Transcriber for Loaded {
    fn transcribe_words(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<Vec<HeardWord>> {
        self.0.transcribe_words(samples, sample_rate, language)
    }
}

/// Loads the Whisper checkpoint at `dir` on `at`.
pub fn load_whisper(dir: &Path, at: Placement) -> Result<Loaded> {
    Ok(Loaded(match at {
        Placement::Cpu => Box::new(WhisperTranscriber::<CpuRuntime>::from_dir_on(
            dir,
            &CpuDevice::new(),
        )?),
        #[cfg(feature = "cuda")]
        Placement::Cuda => {
            use boostr::runtime::cuda::{CudaDevice, CudaRuntime};
            Box::new(WhisperTranscriber::<CudaRuntime>::from_dir_on(
                dir,
                &CudaDevice::new(0),
            )?)
        }
    }))
}

/// Finds and loads the recognizer `id` names.
pub fn find_and_load(search: &ModelSearch, id: &str, at: Placement) -> Result<Loaded> {
    let dir = search
        .obtain("speak", id, ModelKind::Whisper, None)
        .map_err(|reason| Error::Recognizer { reason })?;
    load_whisper(&dir, at)
}

/// The id a recognizer reads: the base, the small, or the configured one.
fn id_of(recognizer: Recognizer, configured: Option<&str>) -> &str {
    match recognizer {
        Recognizer::Default => DEFAULT_ASR_MODEL,
        Recognizer::Small => SMALL_ASR_MODEL,
        Recognizer::Configured => configured.unwrap_or(DEFAULT_ASR_MODEL),
    }
}

/// The hearing of a speak run: every recognizer loaded the first time a
/// take needs it. `main`, the recognizer heard first, is found on disk now,
/// downloaded when it is missing, so a model that cannot be had stops the
/// run before any audio is made.
pub fn hearing<'a>(
    search: &'a ModelSearch,
    configured: Option<&'a str>,
    main: Recognizer,
    at: Placement,
) -> std::result::Result<WhisperHearing<'a>, CommandError> {
    search
        .obtain("speak", id_of(main, configured), ModelKind::Whisper, None)
        .map_err(CommandError::Failed)?;
    let source = move |id: &'a str| RecognizerSource::lazy(move || find_and_load(search, id, at));
    let mut hearing = WhisperHearing::new(source(DEFAULT_ASR_MODEL), source(SMALL_ASR_MODEL));
    if let Some(id) = configured.filter(|c| *c != DEFAULT_ASR_MODEL && *c != SMALL_ASR_MODEL) {
        hearing = hearing.with_configured(source(id));
    }
    Ok(hearing)
}
