//! The `inference.speculative` config section.

use anyhow::{anyhow, Result};
use boostr::inference::speculative::{DEFAULT_NUM_DRAFT_TOKENS, MAX_NUM_DRAFT_TOKENS};
use serde::{Deserialize, Serialize};

use super::speculative_drafter::{DrafterKind, DrafterSpec, PromptLookupConfig};

/// Speculative decoding config. It is off by default.
///
/// `enabled` drafts tokens for each eligible single-sequence request and
/// checks them in one target forward. Greedy output equals plain decoding.
/// A seeded sampled run is reproducible for a fixed seed, `num_draft_tokens`
/// and drafter. It differs from the same seed with speculation off.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeDecodingConfig {
    /// Speculative decoding runs for eligible requests. Default: off.
    #[serde(default)]
    pub enabled: bool,
    /// The drafter. Default: `prompt_lookup`.
    #[serde(default)]
    pub drafter: DrafterKind,
    /// Drafts proposed per step, in `1..=32`. Unset: boostr's default of 5.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_draft_tokens: Option<usize>,
    /// The old name of `num_draft_tokens`. `num_draft_tokens` wins when both are set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_speculative_tokens: Option<usize>,
    /// The n-gram search of the `prompt_lookup` drafter.
    #[serde(default)]
    pub prompt_lookup: PromptLookupConfig,
    /// The draft model of `drafter: model`: a GGUF file or a SafeTensors directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<String>,
    /// Accepted from old config files and ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adaptive_depth: Option<bool>,
}

impl SpeculativeDecodingConfig {
    /// Drafts proposed per step: `num_draft_tokens`, then its old name, then 5.
    pub fn draft_count(&self) -> usize {
        self.num_draft_tokens
            .or(self.num_speculative_tokens)
            .unwrap_or(DEFAULT_NUM_DRAFT_TOKENS)
    }

    /// The draft model path, when set and not empty.
    pub fn draft_model(&self) -> Option<&str> {
        self.draft_model.as_deref().filter(|path| !path.is_empty())
    }

    /// The configured drafter.
    ///
    /// # Errors
    ///
    /// An error naming the refused key: a `prompt_lookup` value boostr
    /// refuses, or `draft_model` unset or empty under `drafter: model`.
    pub fn drafter_spec(&self) -> Result<DrafterSpec> {
        match self.drafter {
            DrafterKind::PromptLookup => self.prompt_lookup.build().map(DrafterSpec::PromptLookup),
            DrafterKind::Model if self.draft_model().is_some() => Ok(DrafterSpec::Model),
            DrafterKind::Model => Err(anyhow!(
                "inference.speculative.draft_model is {:?}; drafter model needs a draft model. \
                 Set draft_model to a GGUF file or a SafeTensors directory",
                self.draft_model.as_deref().unwrap_or_default()
            )),
        }
    }

    /// Check every key, whether or not `enabled` is set.
    ///
    /// # Errors
    ///
    /// An error naming the key: a draft count outside `1..=32`, a
    /// `prompt_lookup` value boostr refuses, or a missing `draft_model`.
    pub fn check(&self) -> Result<()> {
        let key = if self.num_draft_tokens.is_none() && self.num_speculative_tokens.is_some() {
            "num_speculative_tokens"
        } else {
            "num_draft_tokens"
        };
        let count = self.draft_count();
        if !(1..=MAX_NUM_DRAFT_TOKENS).contains(&count) {
            return Err(anyhow!(
                "inference.speculative.{key} is {count}; set a value in 1..={MAX_NUM_DRAFT_TOKENS}"
            ));
        }
        self.prompt_lookup.build()?;
        self.drafter_spec().map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FileConfig, InferenceConfig};
    use boostr::inference::speculative::PromptLookup;
    use std::path::Path;

    fn section(yaml: &str) -> Result<SpeculativeDecodingConfig> {
        let cfg: InferenceConfig = serde_saphyr::from_str(yaml)?;
        cfg.speculative
            .ok_or_else(|| anyhow!("no speculative section in {yaml:?}"))
    }

    #[test]
    fn the_defaults_are_off_and_check() -> Result<()> {
        let spec = SpeculativeDecodingConfig::default();
        assert!(!spec.enabled);
        assert_eq!(spec.drafter, DrafterKind::PromptLookup);
        assert_eq!(spec.draft_count(), DEFAULT_NUM_DRAFT_TOKENS);
        assert_eq!(
            spec.drafter_spec()?,
            DrafterSpec::PromptLookup(PromptLookup::default())
        );
        assert_eq!(spec.draft_model(), None);
        spec.check()?;
        assert!(InferenceConfig::default().speculative.is_none());
        Ok(())
    }

    #[test]
    fn an_old_section_loads_with_its_keys() -> Result<()> {
        let old = "speculative:\n  draft_model: tiny\n  num_speculative_tokens: 7\n  \
                   adaptive_depth: true\n";
        let spec = section(old)?;
        assert_eq!((spec.enabled, spec.draft_count()), (false, 7));
        assert_eq!(
            (spec.draft_model(), spec.adaptive_depth),
            (Some("tiny"), Some(true))
        );
        let empty = section("speculative:\n  draft_model: \"\"\n")?;
        assert_eq!(empty.draft_model(), None);
        let both = section("speculative:\n  num_draft_tokens: 3\n  num_speculative_tokens: 7\n")?;
        assert_eq!(both.draft_count(), 3);
        Ok(())
    }

    #[test]
    fn old_keys_are_known_to_the_user_file() -> Result<()> {
        let text =
            "inference:\n  speculative:\n    enabled: true\n    num_speculative_tokens: 4\n    \
                    adaptive_depth: false\n    draft_model: \"\"\n    \
                    prompt_lookup:\n      max_ngram: 2\n";
        let parsed = FileConfig::parse(text, Path::new("/home/u/.blazr/config.yaml"))?;
        assert!(parsed.unknown_keys.is_empty(), "{:?}", parsed.unknown_keys);
        Ok(())
    }

    #[test]
    fn the_new_keys_parse() -> Result<()> {
        let spec = section(
            "speculative:\n  enabled: true\n  drafter: prompt_lookup\n  num_draft_tokens: 8\n  \
             prompt_lookup:\n    min_ngram: 2\n    window: 64\n",
        )?;
        assert!(spec.enabled);
        assert_eq!(spec.draft_count(), 8);
        let want = PromptLookupConfig {
            min_ngram: 2,
            max_ngram: 3,
            window: 64,
        };
        assert_eq!(spec.prompt_lookup, want);
        spec.check()
    }

    #[test]
    fn a_draft_count_outside_the_range_names_its_key() {
        let cases = [
            (Some(0), None, "num_draft_tokens"),
            (Some(MAX_NUM_DRAFT_TOKENS + 1), None, "num_draft_tokens"),
            (None, Some(0), "num_speculative_tokens"),
        ];
        for (new, old, key) in cases {
            let spec = SpeculativeDecodingConfig {
                num_draft_tokens: new,
                num_speculative_tokens: old,
                ..SpeculativeDecodingConfig::default()
            };
            let err = spec.check().map_err(|e| e.to_string());
            assert!(
                matches!(&err, Err(msg) if msg.contains(&format!("inference.speculative.{key}"))),
                "{spec:?}: {err:?}"
            );
        }
    }

    #[test]
    fn the_model_drafter_parses_with_its_path() -> Result<()> {
        let spec = section(
            "speculative:\n  enabled: true\n  drafter: model\n  draft_model: /models/draft.gguf\n",
        )?;
        assert_eq!(spec.drafter, DrafterKind::Model);
        assert_eq!(spec.draft_model(), Some("/models/draft.gguf"));
        assert_eq!(spec.drafter_spec()?, DrafterSpec::Model);
        spec.check()
    }

    #[test]
    fn the_model_drafter_without_a_path_names_draft_model() {
        for path in [None, Some(String::new())] {
            for enabled in [false, true] {
                let spec = SpeculativeDecodingConfig {
                    enabled,
                    drafter: DrafterKind::Model,
                    draft_model: path.clone(),
                    ..SpeculativeDecodingConfig::default()
                };
                let err = spec.check().map_err(|e| e.to_string());
                assert!(
                    matches!(&err, Err(msg) if msg.contains("inference.speculative.draft_model")),
                    "{spec:?}: {err:?}"
                );
            }
        }
    }

    #[test]
    fn a_bad_ngram_fails_the_check() {
        let spec = SpeculativeDecodingConfig {
            prompt_lookup: PromptLookupConfig {
                min_ngram: 0,
                ..PromptLookupConfig::default()
            },
            ..SpeculativeDecodingConfig::default()
        };
        let err = spec.check().map_err(|e| e.to_string());
        assert!(
            matches!(&err, Err(msg) if msg.contains("prompt_lookup.min_ngram")),
            "{err:?}"
        );
    }
}
