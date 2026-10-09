//! The kinds of speech model and what a complete one holds on disk.

use std::path::Path;

use voxr_narrate::KOKORO_MODEL;

/// The kind of model, which names the files a complete one holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// A Whisper recognizer with word timings.
    Whisper,
    /// Kokoro-82M: its config, weights and voices.
    Kokoro,
    /// A VITS (MMS-TTS) checkpoint.
    Vits,
    /// VoxCPM2: a checkpoint directory, or one `.gguf` file.
    Voxcpm,
    /// Kokoro's G2P models: the spaCy tagger under `tagger/`, the
    /// out-of-vocabulary model under `oov/`.
    KokoroG2p,
    /// A CLAP audio-text model: its configs, tokenizer and weights.
    Clap,
}

impl ModelKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Whisper => "Whisper",
            Self::Kokoro => "Kokoro",
            Self::Vits => "VITS",
            Self::Voxcpm => "VoxCPM2",
            Self::KokoroG2p => "Kokoro G2P",
            Self::Clap => "CLAP",
        }
    }
}

/// The CLAP model the sound review ranks with.
pub const CLAP_ID: &str = "laion/larger_clap_general";
/// VoxCPM2's Hugging Face id.
pub const VOXCPM_ID: &str = "openbmb/VoxCPM2";
/// Kokoro's PyTorch checkpoint, the one voxr loads.
pub(crate) const KOKORO_ID: &str = "hexgrad/Kokoro-82M";
/// Kokoro's G2P models, for a Kokoro checkpoint with no `g2p/` beside it.
pub const KOKORO_G2P_ID: &str = "fs90/kokoro-g2p";
/// The tagger's files, under `tagger/`.
pub const TAGGER_FILES: [&str; 2] = ["tagger.json", "tagger.safetensors"];
/// The out-of-vocabulary model's files, under `oov/`.
pub const OOV_FILES: [&str; 3] = ["config.json", "model.safetensors", "vocab.json"];

/// The checkpoint a skill model id names: the skill's ONNX Whisper builds
/// (`onnx-community/whisper-base_timestamped`) are the `openai/whisper-*`
/// checkpoints, and its ONNX Kokoro build is `hexgrad/Kokoro-82M`. Every
/// other id is itself.
pub fn checkpoint_id(id: &str, kind: ModelKind) -> String {
    match kind {
        ModelKind::Whisper => {
            let Some(rest) = id.strip_prefix("onnx-community/whisper-") else {
                return id.to_string();
            };
            format!("openai/whisper-{}", rest.trim_end_matches("_timestamped"))
        }
        ModelKind::Kokoro if id == KOKORO_MODEL => KOKORO_ID.to_string(),
        _ => id.to_string(),
    }
}

/// The first file a complete model of `kind` at `path` lacks, as a path
/// relative to it. `None` when the model is complete.
pub fn missing_file(path: &Path, kind: ModelKind) -> Option<String> {
    let has = |f: &str| path.join(f).is_file();
    let any = |ext: &str| {
        std::fs::read_dir(path).is_ok_and(|dir| {
            dir.filter_map(Result::ok)
                .any(|e| e.path().is_file() && e.path().extension().is_some_and(|x| x == ext))
        })
    };
    let first = |files: &[&str]| files.iter().find(|f| !has(f)).map(|f| (*f).to_string());
    match kind {
        ModelKind::Whisper => first(&[
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "generation_config.json",
        ]),
        ModelKind::Vits => first(&[
            "config.json",
            "vocab.json",
            "tokenizer_config.json",
            "model.safetensors",
        ]),
        ModelKind::Kokoro => {
            if !has("config.json") {
                Some("config.json".to_string())
            } else if !any("pth") && !any("safetensors") {
                Some("kokoro-v1_0.pth".to_string())
            } else {
                None
            }
        }
        ModelKind::KokoroG2p => {
            let tagger = TAGGER_FILES.map(|f| format!("tagger/{f}"));
            let oov = OOV_FILES.map(|f| format!("oov/{f}"));
            tagger.into_iter().chain(oov).find(|f| !has(f))
        }
        ModelKind::Clap => first(&["config.json", "preprocessor_config.json", "tokenizer.json"])
            .or_else(|| {
                (!has("model.safetensors") && !has("pytorch_model.bin"))
                    .then(|| "model.safetensors".to_string())
            }),
        ModelKind::Voxcpm => {
            if path.is_file() || any("gguf") {
                None
            } else if !has("audiovae.pth") && !has("audiovae.safetensors") {
                first(&["config.json", "model.safetensors"])
                    .or_else(|| Some("audiovae.pth".to_string()))
            } else {
                first(&["config.json", "model.safetensors"])
            }
        }
    }
}
