//! The `inference.speculative` config section.

use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

/// Speculative decoding config.
///
/// `enabled` turns on prompt-lookup drafting for eligible requests. It is off
/// by default. `draft_model` and `adaptive_depth` parse for old config files.
/// Blazr does not read them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeDecodingConfig {
    /// Speculative decoding runs for eligible requests. Default: off.
    #[serde(default)]
    pub enabled: bool,
    /// Path to a draft model. Parsed, not read.
    #[serde(default)]
    pub draft_model: String,
    /// Drafts proposed per step (default: 5).
    #[serde(default = "default_spec_tokens")]
    pub num_speculative_tokens: usize,
    /// Adaptive draft depth. Parsed, not read.
    #[serde(default)]
    pub adaptive_depth: bool,
}

fn default_spec_tokens() -> usize {
    boostr::inference::speculative::DEFAULT_NUM_DRAFT_TOKENS
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            draft_model: String::new(),
            num_speculative_tokens: default_spec_tokens(),
            adaptive_depth: false,
        }
    }
}

/// Set once the ignored-`speculative` warning has been logged.
static SPECULATIVE_WARNED: AtomicBool = AtomicBool::new(false);

/// Log the ignored-`speculative` warning the first time `flag` is unset.
/// Return `true` when this call logged it.
fn warn_speculative_once(
    flag: &AtomicBool,
    speculative: Option<&SpeculativeDecodingConfig>,
) -> bool {
    if speculative.is_none() || flag.swap(true, Ordering::Relaxed) {
        return false;
    }
    tracing::warn!(
        "inference.speculative is ignored: no request path runs speculative decoding yet. \
         Requests decode with the normal sampler"
    );
    true
}

impl super::inference::InferenceConfig {
    /// Log once per process that `inference.speculative` is ignored.
    /// Return `true` when this call logged the warning.
    pub fn warn_speculative_ignored(&self) -> bool {
        warn_speculative_once(&SPECULATIVE_WARNED, self.speculative.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::InferenceConfig;

    #[test]
    fn an_old_speculative_section_parses() -> Result<(), serde_saphyr::Error> {
        let cfg: InferenceConfig =
            serde_saphyr::from_str("speculative:\n  draft_model: tiny\n  adaptive_depth: true\n")?;
        let spec = cfg.speculative;
        assert_eq!(spec.as_ref().map(|s| s.draft_model.as_str()), Some("tiny"));
        assert_eq!(spec.as_ref().map(|s| s.num_speculative_tokens), Some(5));
        assert_eq!(spec.as_ref().map(|s| s.adaptive_depth), Some(true));
        assert_eq!(spec.map(|s| s.enabled), Some(false), "off unless set");
        Ok(())
    }

    #[test]
    fn enabled_parses_without_a_draft_model() -> Result<(), serde_saphyr::Error> {
        let cfg: InferenceConfig = serde_saphyr::from_str("speculative:\n  enabled: true\n")?;
        let want = SpeculativeDecodingConfig {
            enabled: true,
            ..SpeculativeDecodingConfig::default()
        };
        assert_eq!(cfg.speculative, Some(want));
        Ok(())
    }

    #[test]
    fn the_warning_logs_once() {
        let flag = AtomicBool::new(false);
        let spec = SpeculativeDecodingConfig::default();
        assert!(!warn_speculative_once(&flag, None), "unset logs nothing");
        assert!(warn_speculative_once(&flag, Some(&spec)));
        assert!(!warn_speculative_once(&flag, Some(&spec)));
        assert!(!warn_speculative_once(&flag, Some(&spec)));
    }
}
