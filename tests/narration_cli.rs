//! `blazr speak` and `blazr transcribe` end to end: the built binary run on
//! real models in a sandbox, its files, exit codes and `--json` checked.
//!
//! Models come from env vars, else `$BOOSTR_MODELS_DIR`:
//! * `KOKORO_DIR` → `Kokoro-82M`
//! * `WHISPER_BASE_DIR` → `whisper-base`
//! * `WHISPER_SMALL_DIR` → `whisper-small` (else the base checkpoint, with
//!   a notice)
//! * `MMS_TTS_ZLM_DIR` → `facebook/mms-tts-zlm`
//! * `VOXCPM2_GGUF`, no fallback
//!
//! A test whose model is missing prints a skip notice and passes.
#![cfg(feature = "audio")]

mod common;
mod narration;
