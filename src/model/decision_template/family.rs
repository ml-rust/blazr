//! The decision families, their shared error, the token-level output every
//! family renders into, and the model-directory files the families load.
//!
//! A [`RenderedDecision`] holds full prompt rows, the positions each
//! scoring row is read at, and the question each scoring row answers.
//! Those map one to one onto boostr's decision inputs: `rows` and the
//! scoring rows' positions go to `plan_from_rows`. Under the label-token
//! readout each scoring row becomes a `LabelTokenItem` over its slot's
//! gather; under the pointer readout a `PointerItem` over its decide
//! slot's and option ends' gathers. Under the joint-schema readout the one
//! row is the whole record, read by `full_sequence_plan`, and
//! [`RenderedDecision::joint`] holds each field's `JointSchemaItem`. Each
//! [`DecisionQuestion::meta`] is the `QuestionMeta` the answer stage reads.

use std::fmt;
use std::ops::Range;
use std::path::Path;

use boostr::inference::decision::{QuestionMeta, RowSlot};
use boostr::model::decision::joint_schema::config::{CONFIG_FILE, WEIGHTS_FILE};
use serde_json::Value;

use super::clef::{ClefRenderer, JointLayout};
use super::decider::DeciderRenderer;
use super::jebadiah::JebadiahRenderer;
use super::jev::JevRequest;
use super::kev::KevRenderer;
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
    /// Kev: a LoRA adapter and a pointer head over a Qwen3.5 base, the
    /// state and each question's branch between Qwen's delimiter tokens,
    /// one row per question.
    Kev,
    /// Clef: a full Qwen3.5 checkpoint and a joint-schema head
    /// (`joint_head.safetensors`) that scores every field of one record at
    /// once from all its hidden rows.
    Clef,
}

/// How a family reads its answers off the hidden rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Readout {
    /// One label token per option, read at the answer slot.
    LabelToken,
    /// The decide token's query against each option's closing-token key.
    Pointer,
    /// Every hidden row of the record and each option's output rows, all
    /// fields scored jointly.
    JointSchema,
}

impl Readout {
    /// The readout's name as `/v1/models` reports it.
    pub fn name(self) -> &'static str {
        match self {
            Readout::LabelToken => "label_token",
            Readout::Pointer => "pointer",
            Readout::JointSchema => "joint_schema",
        }
    }
}

impl Family {
    /// The family of the model directory at `dir`: `joint_head_config.json`
    /// with `joint_head.safetensors` names Clef, `head.pt` with
    /// `adapter_config.json` Kev, `decider_config.json` a decider,
    /// `prompt_contract.json` a Jebadiah model.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when the directory has none of these files.
    pub fn detect(dir: &Path) -> Result<Self, DecisionError> {
        if dir.join(CONFIG_FILE).is_file() && dir.join(WEIGHTS_FILE).is_file() {
            Ok(Family::Clef)
        } else if dir.join("head.pt").is_file() && dir.join("adapter_config.json").is_file() {
            Ok(Family::Kev)
        } else if dir.join("decider_config.json").is_file() {
            Ok(Family::Decider)
        } else if dir.join("prompt_contract.json").is_file() {
            Ok(Family::Jebadiah)
        } else {
            Err(DecisionError::Model(format!(
                "{} has none of {CONFIG_FILE} with {WEIGHTS_FILE} (Clef), head.pt with \
                 adapter_config.json (Kev), decider_config.json (decider) or \
                 prompt_contract.json (Jebadiah); blazr renders decision prompts for those \
                 four families only",
                dir.display()
            )))
        }
    }

    /// The family's name as the fixtures and logs spell it.
    pub fn name(self) -> &'static str {
        match self {
            Family::Jebadiah => "jebadiah",
            Family::Decider => "decider",
            Family::Kev => "kev",
            Family::Clef => "clef",
        }
    }

    /// How the family's models read their answers.
    pub fn readout(self) -> Readout {
        match self {
            Family::Jebadiah | Family::Decider => Readout::LabelToken,
            Family::Kev => Readout::Pointer,
            Family::Clef => Readout::JointSchema,
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

/// One scoring row: where its option logits are read, and from what.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoringRow {
    /// The row, and the answer slot's position in the full row: the label
    /// slot, or under the pointer readout the decide token.
    pub slot: RowSlot,
    /// Label-token readout: the label token of each option, in option
    /// order. Empty under the pointer readout.
    pub labels: Vec<u32>,
    /// Index into [`RenderedDecision::questions`].
    pub question: usize,
    /// Pointer readout: the position in the slot's full row of each
    /// option's closing token, in option order. Empty under the
    /// label-token readout.
    pub option_ends: Vec<usize>,
}

/// One question of the request, in request order.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionQuestion {
    /// The question id the request used.
    pub id: String,
    /// Kind and option count, as boostr's answer stage takes them. A Noul
    /// question's `yes` is the family's: 0 for Jebadiah (`true` is listed
    /// first), 1 for the decider and Kev (options are `no`, `yes`). A Kev
    /// Score question of one level is a one-option Choice here, its
    /// `legend` set.
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
    /// on its own. Kev: the state delimiter and the whole state, before
    /// any cut. Clef: the state encoded on its own, before any cut.
    pub state_tokens: usize,
    /// Joint-schema readout: each field's spans in the one row, the whole
    /// record. `None` for the other readouts.
    pub joint: Option<JointLayout>,
}

/// A loaded renderer of any family.
#[derive(Debug)]
pub enum DecisionRenderer {
    /// A Jebadiah model.
    Jebadiah(JebadiahRenderer),
    /// A decider model.
    Decider(DeciderRenderer),
    /// A Kev model.
    Kev(KevRenderer),
    /// A Clef model.
    Clef(ClefRenderer),
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
            Family::Kev => DecisionRenderer::Kev(KevRenderer::load(dir)?),
            Family::Clef => DecisionRenderer::Clef(ClefRenderer::load(dir)?),
        })
    }

    /// The family this renderer renders.
    pub fn family(&self) -> Family {
        match self {
            DecisionRenderer::Jebadiah(_) => Family::Jebadiah,
            DecisionRenderer::Decider(_) => Family::Decider,
            DecisionRenderer::Kev(_) => Family::Kev,
            DecisionRenderer::Clef(_) => Family::Clef,
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
            DecisionRenderer::Kev(r) => r.render(request),
            DecisionRenderer::Clef(r) => r.render(request),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_names_each_family_and_its_readout() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert!(Family::detect(dir.path()).is_err());
        std::fs::write(dir.path().join("prompt_contract.json"), "{}").expect("write");
        assert_eq!(Family::detect(dir.path()), Ok(Family::Jebadiah));
        std::fs::write(dir.path().join("decider_config.json"), "{}").expect("write");
        assert_eq!(Family::detect(dir.path()), Ok(Family::Decider));
        std::fs::write(dir.path().join("head.pt"), b"").expect("write");
        assert_eq!(
            Family::detect(dir.path()),
            Ok(Family::Decider),
            "head.pt alone names no family"
        );
        std::fs::write(dir.path().join("adapter_config.json"), "{}").expect("write");
        assert_eq!(Family::detect(dir.path()), Ok(Family::Kev));

        assert_eq!(Family::Kev.name(), "kev");
        assert_eq!(Family::Kev.readout(), Readout::Pointer);
        assert_eq!(Family::Kev.readout().name(), "pointer");
        for family in [Family::Jebadiah, Family::Decider] {
            assert_eq!(family.readout().name(), "label_token");
        }

        std::fs::write(dir.path().join(CONFIG_FILE), "{}").expect("write");
        assert_eq!(
            Family::detect(dir.path()),
            Ok(Family::Kev),
            "the head config alone names no family"
        );
        std::fs::write(dir.path().join(WEIGHTS_FILE), b"").expect("write");
        assert_eq!(Family::detect(dir.path()), Ok(Family::Clef));
        assert_eq!(Family::Clef.name(), "clef");
        assert_eq!(Family::Clef.readout().name(), "joint_schema");
    }
}
