//! Model architecture components.
//!
//! This module contains the model configuration and detection.

mod config;
pub mod detect;

pub use config::Config;
pub use detect::{detect_architecture, DetectedConfig, LayerType, ModelFormat};
