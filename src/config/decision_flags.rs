//! `blazr serve`'s decision flags over the file's `inference.decision`.

use std::path::PathBuf;

use super::decision::{DecisionConfig, DecisionFamily};
use super::FileConfig;

/// The decision flags `blazr serve` was given. `None` keeps the file value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecisionFlags {
    pub family: Option<DecisionFamily>,
    pub config: Option<PathBuf>,
    pub shared_min_tokens: Option<usize>,
    pub lane_budget_mb: Option<usize>,
    pub prefix_cache_mb: Option<usize>,
    pub max_rows: Option<usize>,
    pub max_row_tokens: Option<usize>,
    pub max_request_tokens: Option<usize>,
    pub max_queue_rows: Option<usize>,
    pub batch_wait_ms: Option<u64>,
    /// `--allow-contract-mismatch`: true sets it, false keeps the file value.
    pub allow_contract_mismatch: bool,
    pub base: Option<String>,
    /// `--decision-truncate-states`: true sets it, false keeps the file value.
    pub truncate_states: bool,
}

/// The decision settings: each flag, else `inference.decision`, else the
/// default. `start_model` is the model `--model` loads.
///
/// # Errors
///
/// When `inference.decision` does not parse, or a limit is 0.
pub fn resolve_decision(
    flags: DecisionFlags,
    file: &FileConfig,
    start_model: Option<String>,
) -> Result<DecisionConfig, String> {
    let mut cfg = match file.inference.value("decision") {
        Some(value) => serde_json::from_value::<DecisionConfig>(value.clone())
            .map_err(|e| format!("inference.decision: {e}. Check the keys and their types"))?,
        None => DecisionConfig::default(),
    };
    cfg.family = flags.family.or(cfg.family);
    cfg.config = flags.config.or(cfg.config);
    cfg.shared_min_tokens = flags.shared_min_tokens.unwrap_or(cfg.shared_min_tokens);
    cfg.lane_budget_mb = flags.lane_budget_mb.unwrap_or(cfg.lane_budget_mb);
    cfg.prefix_cache_mb = flags.prefix_cache_mb.unwrap_or(cfg.prefix_cache_mb);
    cfg.max_rows = flags.max_rows.unwrap_or(cfg.max_rows);
    cfg.max_row_tokens = flags.max_row_tokens.or(cfg.max_row_tokens);
    cfg.max_request_tokens = flags.max_request_tokens.unwrap_or(cfg.max_request_tokens);
    cfg.max_queue_rows = flags.max_queue_rows.unwrap_or(cfg.max_queue_rows);
    cfg.batch_wait_ms = flags.batch_wait_ms.unwrap_or(cfg.batch_wait_ms);
    cfg.allow_contract_mismatch |= flags.allow_contract_mismatch;
    cfg.base = flags.base.or(cfg.base);
    cfg.truncate_states |= flags.truncate_states;
    cfg.start_model = start_model;
    cfg.check()?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn file(text: &str) -> FileConfig {
        FileConfig::parse(text, Path::new("config.yaml"))
            .expect("parses")
            .config
    }

    #[test]
    fn flag_then_file_then_default() {
        let from_file = file(
            "inference:\n  decision:\n    family: jebadiah\n    max_rows: 8\n    batch_wait_ms: 5\n",
        );
        let flags = DecisionFlags {
            family: Some(DecisionFamily::Decider),
            max_queue_rows: Some(9),
            prefix_cache_mb: Some(0),
            allow_contract_mismatch: true,
            ..Default::default()
        };
        let cfg = resolve_decision(flags, &from_file, Some("m".into())).expect("resolves");
        assert_eq!(cfg.family, Some(DecisionFamily::Decider));
        assert_eq!(cfg.prefix_cache_mb, 0, "the flag turns the cache off");
        assert_eq!(
            (cfg.max_rows, cfg.max_queue_rows, cfg.batch_wait_ms),
            (8, 9, 5)
        );
        assert!(cfg.allow_contract_mismatch);
        assert_eq!(cfg.start_model.as_deref(), Some("m"));

        let cfg = resolve_decision(DecisionFlags::default(), &file(""), None).expect("resolves");
        assert_eq!(cfg, DecisionConfig::default());
    }

    #[test]
    fn kev_base_and_truncation() {
        let from_file =
            file("inference:\n  decision:\n    base: qwen3.5-4b-base\n    truncate_states: true\n");
        let cfg = resolve_decision(DecisionFlags::default(), &from_file, None).expect("resolves");
        assert_eq!(cfg.base.as_deref(), Some("qwen3.5-4b-base"));
        assert!(cfg.truncate_states);
        let flags = DecisionFlags {
            base: Some("/models/base".into()),
            ..Default::default()
        };
        let cfg = resolve_decision(flags, &from_file, None).expect("resolves");
        assert_eq!(cfg.base.as_deref(), Some("/models/base"));
    }

    #[test]
    fn unknown_key_is_reported_and_zero_limit_refused() {
        let parsed = FileConfig::parse(
            "inference:\n  decision:\n    famly: decider\n",
            Path::new("config.yaml"),
        )
        .expect("parses");
        assert_eq!(parsed.unknown_keys, ["inference.decision.famly"]);

        let flags = DecisionFlags {
            max_request_tokens: Some(0),
            ..Default::default()
        };
        let err = resolve_decision(flags, &file(""), None).expect_err("refused");
        assert!(err.contains("max_request_tokens"), "{err}");
    }
}
