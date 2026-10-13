//! The environment variables that set config values.
//!
//! The resolver in `resolve.rs` reads these values. It never reads the
//! process environment itself, so its tests stay pure.

use std::path::PathBuf;

use anyhow::{anyhow, Result};

/// Config values taken from the environment. An empty variable is unset.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnvVars {
    /// `BLAZR_MODEL_DIR`: model search directory.
    pub model_dir: Option<PathBuf>,
    /// `BLAZR_TP_SIZE`: tensor parallel degree.
    pub tp_size: Option<usize>,
    /// `BLAZR_API_KEY`: API key for `blazr serve`.
    pub api_key: Option<String>,
    /// `BLAZR_VOICE_DIR`: Kokoro voice directory for `blazr serve`.
    pub voice_dir: Option<PathBuf>,
}

impl EnvVars {
    /// Read the variables from the process environment.
    ///
    /// # Errors
    ///
    /// An error when `BLAZR_TP_SIZE` is set and is not a whole number.
    pub fn from_process() -> Result<Self> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// Read the variables through `lookup`.
    ///
    /// # Errors
    ///
    /// An error when `BLAZR_TP_SIZE` is set and is not a whole number.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let get = |name: &str| lookup(name).filter(|value| !value.trim().is_empty());
        let tp_size = match get("BLAZR_TP_SIZE") {
            Some(text) => Some(text.trim().parse::<usize>().map_err(|_| {
                anyhow!("BLAZR_TP_SIZE='{text}' is not a whole number. Set a size such as 2, or unset it")
            })?),
            None => None,
        };
        Ok(Self {
            model_dir: get("BLAZR_MODEL_DIR").map(PathBuf::from),
            tp_size,
            api_key: get("BLAZR_API_KEY"),
            voice_dir: get("BLAZR_VOICE_DIR").map(PathBuf::from),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> Result<EnvVars> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        EnvVars::from_lookup(|name| {
            pairs
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        })
    }

    #[test]
    fn reads_each_variable() {
        let vars = env(&[
            ("BLAZR_MODEL_DIR", "/m"),
            ("BLAZR_TP_SIZE", "2"),
            ("BLAZR_API_KEY", "k"),
            ("BLAZR_VOICE_DIR", "/v"),
        ])
        .unwrap();
        assert_eq!(vars.model_dir, Some(PathBuf::from("/m")));
        assert_eq!(vars.tp_size, Some(2));
        assert_eq!(vars.api_key.as_deref(), Some("k"));
        assert_eq!(vars.voice_dir, Some(PathBuf::from("/v")));
    }

    #[test]
    fn empty_values_are_unset() {
        let vars = env(&[("BLAZR_MODEL_DIR", ""), ("BLAZR_TP_SIZE", " ")]).unwrap();
        assert_eq!(vars, EnvVars::default());
    }

    #[test]
    fn a_bad_tp_size_is_an_error() {
        let err = env(&[("BLAZR_TP_SIZE", "two")]).unwrap_err();
        assert!(err.to_string().contains("BLAZR_TP_SIZE='two'"), "{err}");
    }
}
