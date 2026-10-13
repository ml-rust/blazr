//! `inference.decision` of the user config file: how System One decision
//! models are detected, batched and bounded.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::model::decision_template::Family;

/// Shortest shared prefix, in tokens, that a request runs once with its
/// rows continuing it. Shorter prefixes run as full rows, merged with other
/// requests.
pub const DEFAULT_SHARED_MIN_TOKENS: usize = 768;
/// Lane-state budget of one forward pass, in MiB.
pub const DEFAULT_LANE_BUDGET_MB: usize = 8192;
/// Most prompt rows one request expands to.
pub const DEFAULT_MAX_ROWS: usize = 1024;
/// Tokens a row may hold past the family's state cap.
pub const ROW_TOKENS_OVER_STATE: usize = 4096;
/// Most tokens over all rows of one request.
pub const DEFAULT_MAX_REQUEST_TOKENS: usize = 1 << 20;
/// Most rows queued or running over all requests of one model.
pub const DEFAULT_MAX_QUEUE_ROWS: usize = 4096;

/// A decision model family, as the config and the command line name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum DecisionFamily {
    /// Jebadiah: AINode's prompt inside the chat template.
    Jebadiah,
    /// Mapika decider: the plain `Context:` / `Question:` layout.
    Decider,
}

impl DecisionFamily {
    /// The renderer family.
    pub fn family(self) -> Family {
        match self {
            Self::Jebadiah => Family::Jebadiah,
            Self::Decider => Family::Decider,
        }
    }
}

/// Decision model settings. Every key has a default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DecisionConfig {
    /// The family of the served model, over sidecar and GGUF detection.
    /// It applies to the start model (`--model`), else to every model.
    pub family: Option<DecisionFamily>,
    /// A sidecar file (`decider_config.json`, `prompt_contract.json` or
    /// `jebadiah.json`) for a model that ships none beside it, such as a
    /// bare `.gguf`. Same scope as `family`.
    pub config: Option<PathBuf>,
    /// Shortest shared prefix a request runs once. Shorter: full rows,
    /// merged with other queued requests.
    pub shared_min_tokens: usize,
    /// Lane-state budget of one forward pass in MiB. 0 sets no budget.
    pub lane_budget_mb: usize,
    /// Most prompt rows one request expands to.
    pub max_rows: usize,
    /// Most tokens in one row. Unset: the family's state cap plus 4096.
    pub max_row_tokens: Option<usize>,
    /// Most tokens over all rows of one request.
    pub max_request_tokens: usize,
    /// Most rows queued or running per model. More answers 503.
    pub max_queue_rows: usize,
    /// Milliseconds the worker waits after a request for more to merge.
    pub batch_wait_ms: u64,
    /// Serve a Jebadiah model whose prompt contract does not match.
    pub allow_contract_mismatch: bool,
    /// The start model. Set from `--model`, never from the file.
    #[serde(skip)]
    pub start_model: Option<String>,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            family: None,
            config: None,
            shared_min_tokens: DEFAULT_SHARED_MIN_TOKENS,
            lane_budget_mb: DEFAULT_LANE_BUDGET_MB,
            max_rows: DEFAULT_MAX_ROWS,
            max_row_tokens: None,
            max_request_tokens: DEFAULT_MAX_REQUEST_TOKENS,
            max_queue_rows: DEFAULT_MAX_QUEUE_ROWS,
            batch_wait_ms: 0,
            allow_contract_mismatch: false,
            start_model: None,
        }
    }
}

impl DecisionConfig {
    /// Whether `family` and `config` apply to the model `name`.
    pub fn overrides_apply_to(&self, name: &str) -> bool {
        self.start_model
            .as_deref()
            .is_none_or(|start| start == name)
    }

    /// The lane-state budget in bytes. `None` when `lane_budget_mb` is 0.
    pub fn max_lane_bytes(&self) -> Option<usize> {
        (self.lane_budget_mb > 0).then(|| self.lane_budget_mb.saturating_mul(1 << 20))
    }

    /// The row token cap for a family whose state cap is `state_cap`.
    pub fn row_token_cap(&self, state_cap: usize) -> usize {
        self.max_row_tokens
            .unwrap_or_else(|| state_cap.saturating_add(ROW_TOKENS_OVER_STATE))
    }

    /// Check that every limit admits at least one row and one token.
    ///
    /// # Errors
    ///
    /// The key that is 0, and the fix.
    pub fn check(&self) -> Result<(), String> {
        let zero = [
            ("max_rows", self.max_rows),
            ("max_row_tokens", self.max_row_tokens.unwrap_or(1)),
            ("max_request_tokens", self.max_request_tokens),
            ("max_queue_rows", self.max_queue_rows),
        ];
        match zero.iter().find(|(_, v)| *v == 0) {
            Some((key, _)) => Err(format!(
                "inference.decision.{key} is 0, which admits no request. Set it to 1 or more"
            )),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_decider_server_limits() {
        let cfg: DecisionConfig = serde_json::from_str("{}").expect("parses");
        assert_eq!(cfg, DecisionConfig::default());
        assert_eq!(cfg.row_token_cap(32768), 32768 + 4096);
        assert_eq!(cfg.max_lane_bytes(), Some(8192 << 20));
        assert!(cfg.check().is_ok());
    }

    #[test]
    fn keys_parse_and_zero_limits_are_refused() {
        let cfg: DecisionConfig =
            serde_json::from_str(r#"{"family": "decider", "max_rows": 0, "lane_budget_mb": 0}"#)
                .expect("parses");
        assert_eq!(cfg.family, Some(DecisionFamily::Decider));
        assert_eq!(cfg.max_lane_bytes(), None);
        let err = cfg.check().expect_err("refused");
        assert!(err.contains("inference.decision.max_rows"), "{err}");
    }

    #[test]
    fn overrides_follow_the_start_model() {
        let mut cfg = DecisionConfig::default();
        assert!(cfg.overrides_apply_to("any"));
        cfg.start_model = Some("decider-4b".into());
        assert!(cfg.overrides_apply_to("decider-4b"));
        assert!(!cfg.overrides_apply_to("other"));
    }
}
