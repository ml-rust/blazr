//! Shared fixtures for blazr's `tests/*.rs` integration crates: model-fixture
//! resolution (`$SPECIFIC_VAR` else `$BOOSTR_MODELS_DIR/<relative>`, mirroring
//! voxr's `tests/common/mod.rs`), an in-process ASR-only `AppState` +
//! router, and a hand-rolled multipart body encoder so a test can drive
//! `/v1/audio/transcriptions` without a live server.
//!
//! Each file in `tests/` compiles as its own crate and pulls this module in
//! wholesale, so a helper a given test file does not call reads as dead code
//! there. The helpers ARE used — just not all of them by every crate.
#![allow(dead_code, unused_imports)]

#[cfg(feature = "audio")]
mod app;
mod fixtures;
mod multipart;

#[cfg(feature = "audio")]
pub use app::*;
pub use fixtures::*;
pub use multipart::*;
