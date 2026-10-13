//! The family files of a decision model: the decider config, the Jebadiah
//! prompt contract and its checks, and the chat template the contract pins.

use std::path::PathBuf;

use serde_json::{Map, Value};

use super::detect::DecisionSource;
use super::sidecar::{
    read_json_opt, template_sha256, DECIDER_CONFIG, JEBADIAH_CARD, PROMPT_CONTRACT,
};
use crate::model::chat_template::{ChatTemplate, ModelTemplate};
use crate::model::decision_template::jebadiah::contract_problems;

/// A sidecar's JSON and the file it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct SidecarJson {
    /// The parsed file, or the part of it that applies.
    pub value: Value,
    /// The file.
    pub path: PathBuf,
}

impl SidecarJson {
    /// The file name, as responses and `/v1/models` report the source.
    pub fn name(&self) -> String {
        self.path.file_name().map_or_else(
            || self.path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        )
    }
}

/// The decider config: the configured sidecar, else `decider_config.json`
/// in the sidecar directory. `None` when there is none.
///
/// # Errors
///
/// When the file exists and does not read or parse.
pub fn decider_config(source: &DecisionSource) -> Result<Option<SidecarJson>, String> {
    let path = match &source.config_file {
        Some(file) => file.clone(),
        None => source.sidecar_dir.join(DECIDER_CONFIG),
    };
    Ok(read_json_opt(&path)?.map(|value| SidecarJson { value, path }))
}

/// The Jebadiah prompt contract: the configured sidecar, else
/// `prompt_contract.json`, else `jebadiah.json`'s `prompt_contract`, in the
/// sidecar directory. `None` when there is none.
///
/// # Errors
///
/// When a file exists and does not read or parse.
pub fn jebadiah_contract(source: &DecisionSource) -> Result<Option<SidecarJson>, String> {
    let candidates = match &source.config_file {
        Some(file) => vec![file.clone()],
        None => vec![
            source.sidecar_dir.join(PROMPT_CONTRACT),
            source.sidecar_dir.join(JEBADIAH_CARD),
        ],
    };
    for path in candidates {
        let Some(value) = read_json_opt(&path)? else {
            continue;
        };
        let is_card = path.file_name().is_some_and(|n| n == JEBADIAH_CARD);
        let value = if is_card {
            match value.get("prompt_contract") {
                Some(contract @ Value::Object(_)) => contract.clone(),
                _ => continue,
            }
        } else {
            value
        };
        return Ok(Some(SidecarJson { value, path }));
    }
    Ok(None)
}

/// The chat template the model ships: the GGUF header's, else the
/// checkpoint directory's `chat_template.jinja` or `tokenizer_config.json`.
///
/// # Errors
///
/// When the checkpoint's template files do not read.
pub fn model_template(source: &DecisionSource) -> Result<Option<ModelTemplate>, String> {
    if let Some(template) = source
        .gguf_facts
        .as_ref()
        .and_then(|f| f.chat_template.clone())
    {
        return Ok(Some(template));
    }
    match &source.model_dir {
        Some(dir) => ChatTemplate::from_model(dir).map_err(|e| format!("{}: {e:#}", dir.display())),
        None => Ok(None),
    }
}

/// Every check of `contract` that fails: [`contract_problems`] with the
/// tokenizer's label count `labels` when known, and `chat_template_sha256`
/// against the template the model ships.
pub fn contract_failures(
    contract: &Value,
    labels: Option<usize>,
    template: Option<&ModelTemplate>,
) -> Vec<String> {
    let mut problems = contract_problems(contract, labels);
    if let Some(want) = contract.get("chat_template_sha256").and_then(Value::as_str) {
        match template {
            Some(t) if template_sha256(&t.source) == want => {}
            Some(_) => {
                problems.push("chat_template_sha256 differs from the model's template".into())
            }
            None => problems
                .push("chat_template_sha256 is pinned and the model ships no chat template".into()),
        }
    }
    problems
}

/// An empty JSON object, the decider config a model without one reads.
pub fn empty_config() -> Value {
    Value::Object(Map::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::chat_template::TemplateOrigin;
    use crate::model::decision_template::ainode::PROMPT_SOURCE_SHA256;
    use crate::model::decision_template::Family;
    use serde_json::json;

    fn source(dir: &std::path::Path) -> DecisionSource {
        DecisionSource {
            family: Family::Jebadiah,
            detected_by: String::new(),
            model_dir: Some(dir.to_path_buf()),
            gguf: None,
            gguf_facts: None,
            sidecar_dir: dir.to_path_buf(),
            config_file: None,
        }
    }

    #[test]
    fn contract_comes_from_the_card_when_alone() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert_eq!(jebadiah_contract(&source(dir.path())), Ok(None));
        std::fs::write(
            dir.path().join(JEBADIAH_CARD),
            r#"{"prompt_contract": {"single_token_labels": 68}}"#,
        )
        .expect("write");
        let got = jebadiah_contract(&source(dir.path()))
            .expect("reads")
            .expect("found");
        assert_eq!(got.value, json!({"single_token_labels": 68}));
    }

    #[test]
    fn template_hash_is_checked() {
        let template = ModelTemplate {
            source: "abc".into(),
            origin: TemplateOrigin::Gguf,
            bos_token: String::new(),
            eos_token: String::new(),
        };
        let contract = json!({"prompt_source_sha256": PROMPT_SOURCE_SHA256,
            "chat_template_sha256":
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"});
        assert!(contract_failures(&contract, None, Some(&template)).is_empty());
        let other = ModelTemplate {
            source: "abd".into(),
            ..template
        };
        assert_eq!(contract_failures(&contract, None, Some(&other)).len(), 1);
        assert_eq!(contract_failures(&contract, None, None).len(), 1);
    }
}
