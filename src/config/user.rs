//! User configuration file (~/.blazr/config.yaml)
//!
//! Provides persistent defaults that can be overridden by CLI flags.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// User-level configuration stored at `~/.blazr/config.yaml`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserConfig {
    /// Default model name or path
    #[serde(default)]
    pub default_model: Option<String>,

    /// Default device ("cuda" or "cpu")
    #[serde(default)]
    pub device: Option<String>,

    /// Default context window size
    #[serde(default)]
    pub num_ctx: Option<usize>,

    /// Model search directory
    #[serde(default)]
    pub model_dir: Option<String>,

    /// Default server port
    #[serde(default)]
    pub port: Option<u16>,

    /// Default sampling temperature
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Default max tokens
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

impl UserConfig {
    /// Load from the default config path (`~/.blazr/config.yaml`).
    /// Returns `Default` if the file doesn't exist.
    pub fn load() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        if !path.exists() {
            return Self::default();
        }
        match std::fs::read_to_string(&path) {
            Ok(content) => serde_saphyr::from_str(&content).unwrap_or_else(|e| {
                tracing::warn!("Failed to parse {}: {}", path.display(), e);
                Self::default()
            }),
            Err(e) => {
                tracing::warn!("Failed to read {}: {}", path.display(), e);
                Self::default()
            }
        }
    }

    /// Get the config file path: `~/.blazr/config.yaml`
    pub fn config_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".blazr").join("config.yaml"))
    }
}
