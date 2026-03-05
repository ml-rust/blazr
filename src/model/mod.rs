//! Model architecture components.
//!
//! This module contains the model configuration and detection.

pub mod chat_template;
mod config;
pub mod detect;
pub mod think;

pub use chat_template::{ChatMessage, ChatTemplate};
pub use config::Config;
pub use detect::{detect_architecture, DetectedConfig, LayerType, ModelFormat};
