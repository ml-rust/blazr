//! Speech loaders: TTS bundles and Whisper models from model directories.

pub(super) mod tts;
pub(super) mod whisper_loader;

pub use tts::{
    kokoro_g2p, load_kokoro_engine, load_tts_from_dir, parse_tts_model_arg, voxcpm2_source,
    voxcpm2_source_at, Voxcpm2Source,
};
pub use whisper_loader::{load_whisper_from_dir, parse_asr_model_arg};
