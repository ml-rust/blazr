//! The `"tts"` block of the narration config, owned, and as the take loop
//! reads it.

use voxr_narrate::TtsConfig;

/// The config's `"tts"` block, owned.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TtsSection {
    pub engine: Option<String>,
    pub command: Option<String>,
    pub model: Option<String>,
    pub voice: Option<String>,
    pub reference: Option<String>,
    pub style: Option<String>,
    pub speed: Option<f64>,
    pub pace: Option<f64>,
    pub seed: Option<f64>,
    pub cfg: Option<f64>,
    pub steps: Option<u32>,
    pub hifi: Option<bool>,
    pub one_call: Option<bool>,
    /// A VoxCPM2 checkpoint: a directory, or a `.gguf` file.
    pub checkpoint: Option<String>,
}

impl TtsSection {
    /// The block as the take loop reads it.
    pub fn config(&self) -> TtsConfig<'_> {
        TtsConfig {
            engine: self.engine.as_deref(),
            command: self.command.as_deref(),
            model: self.model.as_deref(),
            voice: self.voice.as_deref(),
            speed: self.speed,
            reference: self.reference.as_deref(),
            style: self.style.as_deref(),
            hifi: self.hifi,
            cfg: self.cfg,
            steps: self.steps,
            seed: self.seed,
            one_call: self.one_call,
        }
    }
}
