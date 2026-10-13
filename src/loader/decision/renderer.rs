//! A decision model's renderer, calibration and standing warnings.
//!
//! The model file supplies the tokenizer and the chat template: a GGUF its
//! embedded vocabulary and `tokenizer.chat_template`, a checkpoint
//! directory its `tokenizer.json` and template files. The sidecars supply
//! the decider config, the prompt contract and the temperatures, each type
//! taken from `blazr_calibration.json` when that file sets it. A Kev model
//! encodes with its base's `tokenizer.json`, as Kev does, and reads its
//! temperature from `head.pt`. A Clef model writes its prompt without the
//! chat template and reads no temperature.

use std::path::{Path, PathBuf};

use boostr::format::{extract_gguf_vocab, Gguf};

use super::blazr_calibration::served_calibration;
use super::clef::{has_joint_head, HEAD_CONFIG, HEAD_WEIGHTS};
use super::detect::DecisionSource;
use super::family_files::{
    contract_failures, decider_config, empty_config, jebadiah_contract, model_template,
};
use super::info::{decision_info_with_base, ContractInfo, DecisionInfo};
use super::kev::HEAD_FILE;
use super::sidecar::{Calibration, DECIDER_CONFIG};
use crate::config::DecisionConfig;
use crate::model::chat_template::{JinjaTemplate, ModelTemplate};
use crate::model::decision_template::clef::ClefRenderer;
use crate::model::decision_template::family::load_tokenizer;
use crate::model::decision_template::{
    DeciderConfig, DeciderRenderer, DecisionRenderer, Family, JebadiahRenderer, KevRenderer, Layout,
};

/// A loaded decision model, weights aside.
#[derive(Debug)]
pub struct LoadedRenderer {
    /// The family's renderer over the model's tokenizer.
    pub renderer: DecisionRenderer,
    /// Kev: the `head.pt` the pointer readout loads. `None` for the other
    /// families.
    pub pointer_head: Option<PathBuf>,
    /// Clef: the directory holding `joint_head_config.json` and
    /// `joint_head.safetensors`, which the joint-schema readout loads.
    /// `None` for the other families.
    pub joint_head: Option<PathBuf>,
    /// The fitted temperatures. `None` serves raw.
    pub calibration: Option<Calibration>,
    /// What every response of this model carries in `warnings`.
    pub warnings: Vec<String>,
    /// The `/v1/models` report, the contract checked with the tokenizer.
    pub info: DecisionInfo,
}

/// The model's own tokenizer.
fn tokenizer(source: &DecisionSource) -> Result<splintr::AnyTokenizer, String> {
    if let Some(file) = &source.gguf {
        let gguf = Gguf::open(file)
            .map_err(|e| format!("{}: the GGUF does not open: {e}", file.display()))?;
        let vocab = extract_gguf_vocab(gguf.metadata())
            .map_err(|e| format!("{}: no tokenizer in the header: {e}", file.display()))?;
        return splintr::from_gguf_vocab(vocab)
            .map_err(|e| format!("{}: the tokenizer does not build: {e}", file.display()));
    }
    let dir = source
        .model_dir
        .as_deref()
        .ok_or_else(|| "the model has neither a GGUF nor a checkpoint directory".to_string())?;
    load_tokenizer(dir).map_err(|e| e.to_string())
}

/// The model's chat template, compiled.
fn compiled(template: Option<&ModelTemplate>) -> Result<JinjaTemplate, String> {
    let t = template.ok_or_else(|| {
        "the model ships no chat template (GGUF tokenizer.chat_template, chat_template.jinja \
         or tokenizer_config.json), and this family's prompt is built with it"
            .to_string()
    })?;
    JinjaTemplate::new(&t.source, t.origin, &t.bos_token, &t.eos_token).map_err(|e| e.to_string())
}

/// The Kev renderer of the Kev directory `dir`: over `base`'s
/// `tokenizer.json`, else the copy `dir` ships, with a warning.
fn kev_renderer(
    dir: &Path,
    base: Option<&Path>,
    cfg: &DecisionConfig,
    warnings: &mut Vec<String>,
) -> Result<KevRenderer, String> {
    let tokenizer_dir = match base {
        Some(base) if base.join("tokenizer.json").is_file() => base,
        _ => {
            warnings.push(format!(
                "the base model's tokenizer.json is not found, so prompts are encoded with \
                 the copy in {}, which Kev itself does not read",
                dir.display()
            ));
            dir
        }
    };
    KevRenderer::load(tokenizer_dir)
        .map(|r| r.with_truncation(cfg.truncate_states))
        .map_err(|e| e.to_string())
}

/// Load the renderer of `source`. A Jebadiah contract mismatch refuses the
/// model unless `cfg.allow_contract_mismatch`, which turns it into a
/// warning. `base` is the checkpoint a Kev model applies to; other
/// families ignore it.
///
/// # Errors
///
/// When the tokenizer, template or a sidecar does not load, the renderer
/// cannot frame the family's prompt, or the contract does not match.
pub fn load_renderer(
    source: &DecisionSource,
    cfg: &DecisionConfig,
    base: Option<&Path>,
) -> Result<LoadedRenderer, String> {
    let allow_contract_mismatch = cfg.allow_contract_mismatch;
    let mut info = decision_info_with_base(source, base)?;
    let mut warnings = Vec::new();
    let mut pointer_head = None;
    let mut joint_head = None;
    let renderer = match source.family {
        Family::Jebadiah => {
            let tokenizer = tokenizer(source)?;
            let template = model_template(source)?;
            let jinja = compiled(template.as_ref())?;
            let renderer = JebadiahRenderer::new(tokenizer, &jinja).map_err(|e| e.to_string())?;
            info.max_options = Some(renderer.max_options());
            match jebadiah_contract(source)? {
                Some(contract) => {
                    let labels = Some(renderer.max_options());
                    let problems = contract_failures(&contract.value, labels, template.as_ref());
                    let name = contract.name();
                    if !problems.is_empty() {
                        let msg = format!("prompt contract {name}: {}", problems.join("; "));
                        if !allow_contract_mismatch {
                            return Err(format!(
                                "{msg}. Pass --allow-contract-mismatch to serve it anyway"
                            ));
                        }
                        warnings.push(format!("{msg}; served under --allow-contract-mismatch"));
                    }
                    info.prompt_contract = Some(ContractInfo {
                        source: name,
                        verified: problems.is_empty(),
                        problems,
                    });
                }
                None => warnings.push(
                    "the model ships no prompt contract, so its prompt is not checked against \
                     the one it was trained on"
                        .to_string(),
                ),
            }
            DecisionRenderer::Jebadiah(renderer)
        }
        Family::Decider => {
            let tokenizer = tokenizer(source)?;
            let template = model_template(source)?;
            let cfg = decider_config(source)?;
            let value = cfg.as_ref().map_or_else(empty_config, |c| c.value.clone());
            let config = DeciderConfig::from_json(&value).map_err(|e| e.to_string())?;
            let jinja = match config.layout {
                Layout::Chat => Some(compiled(template.as_ref())?),
                Layout::Plain => None,
            };
            let renderer = DeciderRenderer::new(tokenizer, config, jinja.as_ref())
                .map_err(|e| e.to_string())?;
            if cfg.is_none() {
                warnings.push(format!(
                    "no {DECIDER_CONFIG} beside the model: the decider's defaults apply \
                     (neutralize_none on, isolated levels off). Pass --decision-config"
                ));
            }
            DecisionRenderer::Decider(renderer)
        }
        Family::Kev => {
            let dir = source
                .model_dir
                .as_deref()
                .ok_or_else(|| "a Kev model is a directory".to_string())?;
            pointer_head = Some(dir.join(HEAD_FILE));
            DecisionRenderer::Kev(kev_renderer(dir, base, cfg, &mut warnings)?)
        }
        Family::Clef => {
            let dir = &source.sidecar_dir;
            if !has_joint_head(dir) {
                return Err(format!(
                    "{} lacks {HEAD_CONFIG} or {HEAD_WEIGHTS}: a Clef model reads its \
                     answers with the joint-schema head they hold",
                    dir.display()
                ));
            }
            joint_head = Some(dir.clone());
            let renderer = ClefRenderer::new(tokenizer(source)?).map_err(|e| e.to_string())?;
            DecisionRenderer::Clef(renderer)
        }
    };
    let calibration = served_calibration(source)?;
    if calibration.is_none() {
        let uncalibrated = match source.family {
            Family::Kev => Some(format!(
                "{HEAD_FILE} sets no temperature, so every answer is uncalibrated \
                 (temperature 1). Fit one with blazr decision calibrate"
            )),
            Family::Jebadiah | Family::Decider => Some(
                "the model ships no fitted temperatures, so every answer is uncalibrated \
                 (temperature 1). Pass --decision-config with the model's sidecar, or fit \
                 them with blazr decision calibrate"
                    .to_string(),
            ),
            // The head's probabilities are a plain softmax of its logits.
            Family::Clef => None,
        };
        warnings.extend(uncalibrated);
    }
    Ok(LoadedRenderer {
        renderer,
        pointer_head,
        joint_head,
        calibration,
        warnings,
        info,
    })
}
