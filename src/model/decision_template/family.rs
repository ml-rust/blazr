//! The decision families, their shared error, the token-level output every
//! family renders into, and the model-directory files both families load.
//!
//! A [`RenderedDecision`] holds full prompt rows, the answer slot and label
//! tokens of each scoring row, and the question each scoring row answers.
//! Those map one to one onto boostr's decision inputs: `rows` and the
//! [`ScoringRow::slot`]s go to `plan_from_rows`, each scoring row becomes a
//! `LabelTokenItem` over its slot's gather, and each
//! [`DecisionQuestion::meta`] is the `QuestionMeta` the answer stage reads.

use std::fmt;
use std::ops::Range;
use std::path::Path;

use boostr::inference::decision::{QuestionMeta, RowSlot};
use serde_json::Value;

use super::decider::DeciderRenderer;
use super::jebadiah::JebadiahRenderer;
use super::jev::JevRequest;
use crate::model::chat_template::{ChatTemplate, JinjaTemplate, ModelTemplate};

/// A model family whose prompt blazr renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Jebadiah: AINode's `/v1/systemone` prompt inside the model's chat
    /// template, one row per question, the answer slot the last token.
    Jebadiah,
    /// Mapika decider: the plain `Context:` / `Question:` / `Answer: (`
    /// layout, rows per question or one packed row.
    Decider,
}

impl Family {
    /// The family of the model directory at `dir`: `decider_config.json`
    /// names a decider, `prompt_contract.json` names a Jebadiah model.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when the directory has neither file.
    pub fn detect(dir: &Path) -> Result<Self, DecisionError> {
        if dir.join("decider_config.json").is_file() {
            Ok(Family::Decider)
        } else if dir.join("prompt_contract.json").is_file() {
            Ok(Family::Jebadiah)
        } else {
            Err(DecisionError::Model(format!(
                "{} has neither decider_config.json (decider) nor prompt_contract.json \
                 (Jebadiah); blazr renders decision prompts for those two families only",
                dir.display()
            )))
        }
    }

    /// The family's name as the fixtures and logs spell it.
    pub fn name(self) -> &'static str {
        match self {
            Family::Jebadiah => "jebadiah",
            Family::Decider => "decider",
        }
    }
}

/// Why a decision prompt was not rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionError {
    /// The request is malformed. The message is the caller-facing text, in
    /// the family's own wording where its reference phrases it.
    Request(String),
    /// The model directory, its tokenizer or its template cannot render this
    /// family's prompt.
    Model(String),
}

impl fmt::Display for DecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecisionError::Request(m) | DecisionError::Model(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for DecisionError {}

/// One scoring row: where its label logits are read and which tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoringRow {
    /// The row, and the answer slot's position in the full row.
    pub slot: RowSlot,
    /// The label token of each option, in option order.
    pub labels: Vec<u32>,
    /// Index into [`RenderedDecision::questions`].
    pub question: usize,
}

/// One question of the request, in request order.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionQuestion {
    /// The question id the request used.
    pub id: String,
    /// Kind and option count, as boostr's answer stage takes them. A Noul
    /// question's `yes` is the family's: 0 for Jebadiah (`true` is listed
    /// first), 1 for the decider (options are `no`, `yes`).
    pub meta: QuestionMeta,
    /// What each option answers to on the wire, in option order: a choice
    /// criteria key as written, `true`/`false`, or a level index.
    pub names: Vec<String>,
    /// A Score question's level names as the family's wire `legend` shows
    /// them, in level order. Empty for the other kinds.
    pub legend: Vec<String>,
    /// The question's scoring rows: one, or one per level of an isolated
    /// Score question.
    pub scoring_rows: Range<usize>,
}

/// A request rendered to tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedDecision {
    /// The family that rendered it.
    pub family: Family,
    /// Every prompt row, in full: no shared prefix cut, no padding.
    pub rows: Vec<Vec<u32>>,
    /// Every scoring row, in the reference's order.
    pub scoring_rows: Vec<ScoringRow>,
    /// Every question, in request order.
    pub questions: Vec<DecisionQuestion>,
    /// Things the caller should know that did not stop the render.
    pub warnings: Vec<String>,
    /// Tokens the state takes. Decider: the `Context:\n` block every row
    /// starts with, after the state cap. Jebadiah: the state text encoded
    /// on its own.
    pub state_tokens: usize,
}

/// A loaded renderer of either family.
#[derive(Debug)]
pub enum DecisionRenderer {
    /// A Jebadiah model.
    Jebadiah(JebadiahRenderer),
    /// A decider model.
    Decider(DeciderRenderer),
}

impl DecisionRenderer {
    /// Load the renderer of the model directory at `dir`, its family
    /// detected by [`Family::detect`].
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] as the family's own `load`.
    pub fn load(dir: &Path) -> Result<Self, DecisionError> {
        Ok(match Family::detect(dir)? {
            Family::Jebadiah => DecisionRenderer::Jebadiah(JebadiahRenderer::load(dir)?),
            Family::Decider => DecisionRenderer::Decider(DeciderRenderer::load(dir)?),
        })
    }

    /// The family this renderer renders.
    pub fn family(&self) -> Family {
        match self {
            DecisionRenderer::Jebadiah(_) => Family::Jebadiah,
            DecisionRenderer::Decider(_) => Family::Decider,
        }
    }

    /// Render a request parsed with [`JevRequest::parse`] for this family.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Request`] for a request the family refuses,
    /// [`DecisionError::Model`] when the tokenizer cannot encode it.
    pub fn render(&self, request: &JevRequest) -> Result<RenderedDecision, DecisionError> {
        match self {
            DecisionRenderer::Jebadiah(r) => r.render(request),
            DecisionRenderer::Decider(r) => r.render(request),
        }
    }
}

/// A JSON file of the model directory.
pub(crate) fn read_json(path: &Path) -> Result<Value, DecisionError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| DecisionError::Model(format!("{} does not read: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| DecisionError::Model(format!("{} is not valid JSON: {e}", path.display())))
}

/// `tokenizer.json` of the model directory.
pub(crate) fn load_tokenizer(dir: &Path) -> Result<splintr::AnyTokenizer, DecisionError> {
    let path = dir.join("tokenizer.json");
    splintr::from_json_path(&path).map_err(|e| {
        DecisionError::Model(format!(
            "{} does not load as a tokenizer: {e}",
            path.display()
        ))
    })
}

/// The chat template the model ships.
pub(crate) fn load_template(dir: &Path) -> Result<JinjaTemplate, DecisionError> {
    let shipped: Option<ModelTemplate> = ChatTemplate::from_model(dir)
        .map_err(|e| DecisionError::Model(format!("{}: {e:#}", dir.display())))?;
    let shipped = shipped.ok_or_else(|| {
        DecisionError::Model(format!(
            "{} ships no chat template (chat_template.jinja or tokenizer_config.json)",
            dir.display()
        ))
    })?;
    JinjaTemplate::new(
        &shipped.source,
        shipped.origin,
        &shipped.bos_token,
        &shipped.eos_token,
    )
    .map_err(|e| DecisionError::Model(e.to_string()))
}
