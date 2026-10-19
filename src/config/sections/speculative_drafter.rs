//! The drafter keys of the `inference.speculative` config section.

use anyhow::{anyhow, Result};
use boostr::error::Error;
use boostr::inference::speculative::PromptLookup;
use serde::{Deserialize, Serialize};

/// The drafter that proposes speculative tokens.
///
/// `prompt_lookup` is the only drafter. A draft model is not supported.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrafterKind {
    /// Copy the tokens that followed an earlier n-gram of the context.
    #[default]
    PromptLookup,
}

impl DrafterKind {
    /// The config value of this drafter.
    pub fn name(self) -> &'static str {
        match self {
            Self::PromptLookup => "prompt_lookup",
        }
    }
}

/// The n-gram search of the `prompt_lookup` drafter.
///
/// It tries n-grams of `max_ngram` down to `min_ngram` tokens in the last
/// `window` tokens of the context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptLookupConfig {
    /// Shortest n-gram tried. Default: 1.
    #[serde(default = "default_min_ngram")]
    pub min_ngram: usize,
    /// Longest n-gram tried. Default: 3.
    #[serde(default = "default_max_ngram")]
    pub max_ngram: usize,
    /// Context tokens scanned for a match. Default: 4096.
    #[serde(default = "default_window")]
    pub window: usize,
}

fn default_min_ngram() -> usize {
    1
}

fn default_max_ngram() -> usize {
    3
}

fn default_window() -> usize {
    4096
}

impl Default for PromptLookupConfig {
    fn default() -> Self {
        Self {
            min_ngram: default_min_ngram(),
            max_ngram: default_max_ngram(),
            window: default_window(),
        }
    }
}

impl PromptLookupConfig {
    /// The boostr drafter of these keys.
    ///
    /// # Errors
    ///
    /// An error naming the refused key: `min_ngram` is 0, `max_ngram` is
    /// below `min_ngram`, or `window` is not above `max_ngram`.
    pub fn build(&self) -> Result<PromptLookup> {
        PromptLookup::new(self.min_ngram, self.max_ngram, self.window).map_err(|e| match e {
            Error::InvalidArgument { arg, reason } => {
                anyhow!("inference.speculative.prompt_lookup.{arg}: {reason}")
            }
            other => anyhow!("inference.speculative.prompt_lookup: {other}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_build_boostr_default_lookup() -> Result<()> {
        assert_eq!(
            PromptLookupConfig::default().build()?,
            PromptLookup::default()
        );
        Ok(())
    }

    #[test]
    fn a_refused_ngram_names_its_key() {
        let cases = [
            (0, 3, 4096, "prompt_lookup.min_ngram"),
            (4, 3, 4096, "prompt_lookup.max_ngram"),
            (1, 3, 3, "prompt_lookup.window"),
        ];
        for (min_ngram, max_ngram, window, key) in cases {
            let config = PromptLookupConfig {
                min_ngram,
                max_ngram,
                window,
            };
            let err = config.build().map(|_| ()).map_err(|e| e.to_string());
            assert!(
                matches!(&err, Err(msg) if msg.contains(key)),
                "{config:?}: {err:?}"
            );
        }
    }

    #[test]
    fn the_drafter_parses_in_snake_case() -> Result<(), serde_saphyr::Error> {
        let kind: DrafterKind = serde_saphyr::from_str("prompt_lookup")?;
        assert_eq!(
            (kind, kind.name()),
            (DrafterKind::PromptLookup, "prompt_lookup")
        );
        assert!(serde_saphyr::from_str::<DrafterKind>("model").is_err());
        Ok(())
    }
}
