//! What `/v1/models` reports about a decision model, read from its sidecars
//! and GGUF header without loading weights or a tokenizer.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

use super::blazr_calibration::served_calibration;
use super::clef::{read_head_config, HEAD_DTYPE};
use super::detect::DecisionSource;
use super::family_files::{
    contract_failures, decider_config, empty_config, jebadiah_contract, model_template,
};
use super::kev::{head_temperature, read_kev_adapter};
use super::sidecar::{context_length, license_and_base, Calibration};
use crate::engine::decision::PrefixCacheReport;
use crate::model::decision_template::clef::MAX_LENGTH as CLEF_MAX_LENGTH;
use crate::model::decision_template::decider::DEFAULT_MAX_STATE_TOKENS;
use crate::model::decision_template::jebadiah::DEFAULT_MAX_TOKENS;
use crate::model::decision_template::kev::{SERVE_MAX_ROW, SERVE_MAX_STATE};
use crate::model::decision_template::kev_text::MAX_OPTIONS;
use crate::model::decision_template::labels::DECIDER_MAX_OPTIONS;
use crate::model::decision_template::{DeciderConfig, Family, Layout};

/// The prompt contract a Jebadiah model ships.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ContractInfo {
    /// The file it came from.
    pub source: String,
    /// The checks that need no tokenizer pass: the renderer hash, the chat
    /// template hash and the template variables. The label count is
    /// checked when the model loads.
    pub verified: bool,
    /// Each check that fails.
    pub problems: Vec<String>,
}

/// The temperatures a decision model ships.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationInfo {
    /// The file the temperatures came from: `blazr_calibration.json` when it
    /// sets any type, else the vendor sidecar. `None` when there is none.
    pub source: Option<String>,
    /// The temperature of each answer type the files set.
    pub temperatures: Option<BTreeMap<String, f64>>,
    /// The fit date of `blazr_calibration.json`, when it sets any type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fitted_at: Option<String>,
    /// The file each type's temperature came from, when
    /// `blazr_calibration.json` sets any type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<BTreeMap<String, String>>,
}

impl CalibrationInfo {
    fn of(calibration: Option<&Calibration>) -> Self {
        let fitted = calibration.filter(|c| c.fitted_at.is_some());
        Self {
            source: calibration.map(|c| c.source.clone()),
            temperatures: calibration.map(|c| c.temperatures.clone()),
            fitted_at: fitted.and_then(|c| c.fitted_at.clone()),
            sources: fitted.map(|c| c.sources.clone()),
        }
    }
}

/// The `decision` object of a `/v1/models` entry.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecisionInfo {
    /// `jebadiah`, `decider`, `kev` or `clef`.
    pub family: &'static str,
    /// How answers are read: `label_token`, one label token per option;
    /// `pointer`, Kev's decide-token query against each option's key; or
    /// `joint_schema`, Clef's head over every row of the record.
    pub readout: &'static str,
    /// What named the family.
    pub detected_by: String,
    /// The prompt layout: `plain` or `chat`.
    pub layout: &'static str,
    /// Decider: the most options one question takes. Jebadiah: AINode's
    /// single-token labels, the contract's `single_token_labels`; a wider
    /// question takes the extended alphabet with a warning. `None` when
    /// unknown before the tokenizer loads.
    pub max_options: Option<usize>,
    /// Decider: the `Context:` block cap a request may lower.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_state_tokens: Option<usize>,
    /// Jebadiah: the prompt-token cap of one row when the request names none.
    /// Kev: the token cap of one row, the state and one question's branch.
    /// Clef: the token cap of the whole record; a longer state is cut.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_prompt_tokens: Option<usize>,
    /// The positions the model addresses.
    pub context_length: Option<u64>,
    /// Jebadiah: the prompt contract, `None` when the model ships none.
    pub prompt_contract: Option<ContractInfo>,
    /// The fitted temperatures.
    pub calibration: CalibrationInfo,
    /// The license the model card or GGUF header names.
    pub license: Option<String>,
    /// The base model the model card or GGUF header names; for Kev, else
    /// the adapter's `base_model_name_or_path`.
    pub base_model: Option<String>,
    /// Kev: the base checkpoint the adapter applies to, when it resolves.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_path: Option<String>,
    /// Kev: the LoRA adapter's rank.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter_rank: Option<usize>,
    /// Kev: the one temperature `head.pt` was fitted with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Clef: the width of the joint-schema head.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_width: Option<usize>,
    /// Clef: the backbone hidden size the head reads.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_hidden_size: Option<usize>,
    /// Clef: the dtype the joint-schema head computes in.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_dtype: Option<&'static str>,
    /// The prefix-state cache of the model's decision worker. Absent until
    /// the worker runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache: Option<PrefixCacheReport>,
}

/// The `/v1/models` report of `source`.
///
/// # Errors
///
/// When a sidecar exists and does not parse, or its layout or temperatures
/// are invalid, `blazr_calibration.json` included.
pub fn decision_info(source: &DecisionSource) -> Result<DecisionInfo, String> {
    decision_info_with_base(source, None)
}

/// [`decision_info`], with the base checkpoint a Kev model applies to:
/// its context length and path. Other families take no base.
///
/// # Errors
///
/// As [`decision_info`], and for Kev when `adapter_config.json` or
/// `head.pt` does not read.
pub fn decision_info_with_base(
    source: &DecisionSource,
    base: Option<&Path>,
) -> Result<DecisionInfo, String> {
    let card_dir = source.model_dir.as_deref().unwrap_or(&source.sidecar_dir);
    let facts = source.gguf_facts.as_ref();
    let (license, base_model) = license_and_base(Some(card_dir), None);
    let mut info = DecisionInfo {
        family: source.family.name(),
        readout: source.family.readout().name(),
        detected_by: source.detected_by.clone(),
        layout: "plain",
        max_options: None,
        max_state_tokens: None,
        max_prompt_tokens: None,
        context_length: context_length(Some(card_dir), None)
            .or_else(|| facts.and_then(|f| f.context_length)),
        prompt_contract: None,
        calibration: CalibrationInfo::of(None),
        license: license.or_else(|| facts.and_then(|f| f.license.clone())),
        base_model: base_model.or_else(|| facts.and_then(|f| f.base_model.clone())),
        base_path: None,
        adapter_rank: None,
        temperature: None,
        head_width: None,
        head_hidden_size: None,
        head_dtype: None,
        prefix_cache: None,
    };
    match source.family {
        Family::Decider => {
            let value = decider_config(source)?.map_or_else(empty_config, |c| c.value);
            let config = DeciderConfig::from_json(&value).map_err(|e| e.to_string())?;
            info.layout = match config.layout {
                Layout::Plain => "plain",
                Layout::Chat => "chat",
            };
            info.max_options = Some(DECIDER_MAX_OPTIONS);
            info.max_state_tokens = Some(DEFAULT_MAX_STATE_TOKENS);
        }
        Family::Jebadiah => {
            info.layout = "chat";
            info.max_prompt_tokens = Some(DEFAULT_MAX_TOKENS);
            if let Some(contract) = jebadiah_contract(source)? {
                info.max_options = contract
                    .value
                    .get("single_token_labels")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|n| usize::try_from(n).ok());
                let template = model_template(source)?;
                let problems = contract_failures(&contract.value, None, template.as_ref());
                info.prompt_contract = Some(ContractInfo {
                    source: contract.name(),
                    verified: problems.is_empty(),
                    problems,
                });
            }
        }
        Family::Kev => {
            let adapter = read_kev_adapter(card_dir)?;
            info.max_options = Some(MAX_OPTIONS);
            info.max_state_tokens = Some(SERVE_MAX_STATE);
            info.max_prompt_tokens = Some(SERVE_MAX_ROW);
            info.adapter_rank = Some(adapter.rank);
            info.temperature = head_temperature(card_dir)?;
            info.base_model = info.base_model.or(Some(adapter.base));
            if let Some(base) = base {
                info.base_path = Some(base.display().to_string());
                info.context_length = context_length(Some(base), None).or(info.context_length);
            }
        }
        Family::Clef => {
            let head = read_head_config(&source.sidecar_dir)?;
            info.max_prompt_tokens = Some(CLEF_MAX_LENGTH);
            info.head_width = Some(head.width);
            info.head_hidden_size = Some(head.hidden_size);
            info.head_dtype = Some(HEAD_DTYPE);
        }
    }
    info.calibration = CalibrationInfo::of(served_calibration(source)?.as_ref());
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::decision::sidecar::{DECIDER_CONFIG, TEMPERATURES};

    fn source(dir: &std::path::Path, family: Family) -> DecisionSource {
        DecisionSource {
            family,
            detected_by: "test".into(),
            model_dir: Some(dir.to_path_buf()),
            gguf: None,
            gguf_facts: None,
            sidecar_dir: dir.to_path_buf(),
            config_file: None,
        }
    }

    #[test]
    fn decider_info_reads_its_config() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(
            dir.path().join(DECIDER_CONFIG),
            r#"{"temperature": 1.0, "temperature_by_type": {"noul": 1.5}, "layout": "plain"}"#,
        )
        .expect("write");
        let info = decision_info(&source(dir.path(), Family::Decider)).expect("info");
        assert_eq!((info.family, info.layout), ("decider", "plain"));
        assert_eq!(info.max_options, Some(255));
        let temps = info.calibration.temperatures.clone().expect("calibrated");
        assert_eq!((temps["noul"], temps["choice"]), (1.5, 1.0));
        let json = serde_json::to_value(&info).expect("serializes");
        assert!(json.get("max_prompt_tokens").is_none());
        assert_eq!(json["readout"], "label_token");
    }

    #[test]
    fn jebadiah_info_without_sidecars_is_uncalibrated() {
        let dir = tempfile::tempdir().expect("temp dir");
        let info = decision_info(&source(dir.path(), Family::Jebadiah)).expect("info");
        assert_eq!(info.prompt_contract, None);
        assert_eq!(info.calibration.source, None);
        std::fs::write(
            dir.path().join(TEMPERATURES),
            r#"{"temperatures": {"choice": 1.1, "noul": 1.3, "score": 0.8}}"#,
        )
        .expect("write");
        let info = decision_info(&source(dir.path(), Family::Jebadiah)).expect("info");
        assert_eq!(info.calibration.source.as_deref(), Some(TEMPERATURES));
        assert_eq!(info.max_prompt_tokens, Some(2048));
    }

    #[test]
    fn blazr_calibration_wins_per_type_and_a_bad_one_fails() {
        use crate::loader::decision::blazr_calibration::{BLAZR_CALIBRATION, CALIBRATION_SCHEMA};
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(
            dir.path().join(DECIDER_CONFIG),
            r#"{"temperature": 1.0, "temperature_by_type": {"noul": 1.5}}"#,
        )
        .expect("write");
        let file = serde_json::json!({"schema": CALIBRATION_SCHEMA, "family": "decider",
            "fitted_at": "2026-10-13", "model": "d", "temperatures": {"choice": 1.3},
            "fits": {}, "source": "blazr decision calibrate"});
        let path = dir.path().join(BLAZR_CALIBRATION);
        std::fs::write(&path, file.to_string()).expect("write");
        let info = decision_info(&source(dir.path(), Family::Decider)).expect("info");
        let cal = &info.calibration;
        assert_eq!(cal.source.as_deref(), Some(BLAZR_CALIBRATION));
        assert_eq!(cal.fitted_at.as_deref(), Some("2026-10-13"));
        let temps = cal.temperatures.clone().expect("calibrated");
        assert_eq!((temps["choice"], temps["noul"]), (1.3, 1.5));
        let sources = cal.sources.clone().expect("per type");
        assert_eq!(sources["noul"], DECIDER_CONFIG);
        let json = serde_json::to_value(&info).expect("serializes");
        assert_eq!(json["calibration"]["fitted_at"], "2026-10-13");

        let mut bad = file;
        bad["temperatures"]["choice"] = serde_json::json!(0.0);
        std::fs::write(&path, bad.to_string()).expect("write");
        let e = decision_info(&source(dir.path(), Family::Decider)).expect_err("refused");
        assert!(
            e.contains(BLAZR_CALIBRATION) && e.contains("temperatures"),
            "{e}"
        );
    }

    #[test]
    fn clef_info_reads_the_head_config() {
        use crate::loader::decision::clef::{HEAD_CONFIG, HEAD_WEIGHTS};
        let dir = tempfile::tempdir().expect("temp dir");
        let head = r#"{"hidden_size": 4096, "width": 1024, "routing_layers": 2,
            "layers": 4, "heads": 16, "feedforward": 4096}"#;
        std::fs::write(dir.path().join(HEAD_CONFIG), head).expect("write");
        std::fs::write(dir.path().join(HEAD_WEIGHTS), b"").expect("write");
        let info = decision_info(&source(dir.path(), Family::Clef)).expect("info");
        let json = serde_json::to_value(&info).expect("serializes");
        assert_eq!(json["family"], "clef");
        assert_eq!(json["readout"], "joint_schema");
        assert_eq!(json["head_width"], 1024);
        assert_eq!(json["head_hidden_size"], 4096);
        assert_eq!(json["head_dtype"], "f32");
        assert_eq!(json["max_prompt_tokens"], 16384);
        assert_eq!(json["calibration"]["source"], serde_json::Value::Null);
        assert!(json.get("temperature").is_none());
    }

    #[test]
    fn vendor_only_info_has_no_fit_date() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(
            dir.path().join(TEMPERATURES),
            r#"{"temperatures": {"choice": 1.1}}"#,
        )
        .expect("write");
        let info = decision_info(&source(dir.path(), Family::Jebadiah)).expect("info");
        let json = serde_json::to_value(&info).expect("serializes");
        assert!(json["calibration"].get("fitted_at").is_none());
        assert!(json["calibration"].get("sources").is_none());
    }
}
