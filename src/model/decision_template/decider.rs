//! The Mapika decider prompt: `Context:\n<state>`, then per question
//! `\n\nQuestion: ...\nOptions:\n(A) ...\nAnswer: (`, the label read at the
//! final ` (` token.
//!
//! Ported from `decider/infer.py` (`Decider._system_one_items`,
//! `neutralize_options`), `decider/prompt.py` (`build`, `build_chat`) and
//! `decider/prompt_fast.py` (`build_rows`, `question_piece`). Every piece is encoded on its own and
//! the ids concatenated, exactly where the reference does so: the
//! `Context:` block, each question block, each wide option.
//!
//! Rendered: the state-first layout, plain or chat. The schema-first
//! layout is refused.

use std::fmt;
use std::path::Path;

use boostr::inference::decision::{QuestionKind, QuestionMeta, RowSlot};
use serde_json::{Map, Value};

use super::decider_layout::{chat_layout, ChatLayout, DeciderConfig, Layout, ANSWER, NARROW};
use super::encode::{DecisionEncoder, PromptText};
use super::family::{
    load_template, load_tokenizer, read_json, DecisionError, DecisionQuestion, Family,
    RenderedDecision, ScoringRow,
};
use super::jev::JevRequest;
use super::labels::{DeciderLabels, DECIDER_MAX_OPTIONS};
use super::py_text::{py_repr_str, py_strip};
use super::systemone::{plan_rows, render_question, render_state, PlanRow, S1Kind};
use crate::model::chat_template::JinjaTemplate;

/// The narrow labels.
pub(super) const LETTERS: [&str; NARROW] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
/// The `Context:` block cap when the request names none.
pub const DEFAULT_MAX_STATE_TOKENS: usize = 32768;
/// What a "none of the above" option is shown as.
pub const NEUTRAL_NONE: &str = "not listed here";

/// Renders decider prompts with one tokenizer.
pub struct DeciderRenderer {
    encoder: DecisionEncoder,
    labels: DeciderLabels,
    config: DeciderConfig,
    chat: Option<ChatLayout>,
}

impl fmt::Debug for DeciderRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeciderRenderer")
            .field("config", &self.config)
            .finish()
    }
}

impl DeciderRenderer {
    /// Load from a model directory: `decider_config.json` (absent is the
    /// plain layout with the defaults), `tokenizer.json`, and for the chat
    /// layout the chat template.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when a file does not load or the config or
    /// tokenizer cannot render the decider's prompt.
    pub fn load(dir: &Path) -> Result<Self, DecisionError> {
        let path = dir.join("decider_config.json");
        let cfg = if path.is_file() {
            read_json(&path)?
        } else {
            Value::Object(Map::new())
        };
        let config = DeciderConfig::from_json(&cfg)?;
        let template = match config.layout {
            Layout::Chat => Some(load_template(dir)?),
            Layout::Plain => None,
        };
        Self::new(load_tokenizer(dir)?, config, template.as_ref())
    }

    /// A renderer over `tokenizer`. `template` is the chat template, needed
    /// for the chat layout only.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when the label table cannot be built or the
    /// chat layout's checks fail.
    pub fn new(
        tokenizer: splintr::AnyTokenizer,
        config: DeciderConfig,
        template: Option<&JinjaTemplate>,
    ) -> Result<Self, DecisionError> {
        let encoder = DecisionEncoder::new(tokenizer)?;
        let labels = DeciderLabels::probe(&encoder)?;
        let chat = match config.layout {
            Layout::Plain => None,
            Layout::Chat => Some(chat_layout(&encoder, &labels, template)?),
        };
        Ok(Self {
            encoder,
            labels,
            config,
            chat,
        })
    }

    /// The config the renderer reads.
    pub fn config(&self) -> &DeciderConfig {
        &self.config
    }

    /// Render `request`, as `Decider.system_one` builds its rows.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Request`] with the decider's message for a question
    /// it refuses, or for the schema-first layout.
    pub fn render(&self, request: &JevRequest) -> Result<RenderedDecision, DecisionError> {
        let mut warnings = Vec::new();
        let default = if self.config.schema_first {
            "schema_first"
        } else {
            "state_first"
        };
        match request.layout.as_deref().unwrap_or(default) {
            "state_first" => {}
            "schema_first" => {
                return Err(DecisionError::Request(
                    "the schema-first layout is not rendered here; send \"layout\": \
                     \"state_first\""
                        .to_string(),
                ))
            }
            other => warnings.push(format!(
                "'layout' {} is read as \"state_first\", as the decider reads it",
                py_repr_str(other)
            )),
        }
        let independent = request.independent.unwrap_or(true);
        let isolated = request.isolated.unwrap_or(self.config.isolated_levels) && independent;
        let neutralize = request
            .neutralize_none
            .unwrap_or(self.config.neutralize_none);
        let max_state = request.max_state_tokens.unwrap_or(DEFAULT_MAX_STATE_TOKENS);

        let ctx = render_state(request.state.as_ref().unwrap_or(&Value::Null));
        let rqs = request
            .questions
            .iter()
            .map(|q| render_question(&q.spec))
            .collect::<Result<Vec<_>, _>>()
            .map_err(DecisionError::Request)?;
        let (mut flat, index) = plan_rows(&rqs, isolated);
        if neutralize {
            for row in &mut flat {
                let (options, changed) = neutralize_options(&row.options);
                if changed {
                    warnings.push(format!(
                        "an option reading as \"none of the above\" is shown as \
                         \"{NEUTRAL_NONE}\""
                    ));
                }
                row.options = options;
            }
        }

        let mut context = PromptText::template("Context:\n");
        context.push_caller(ctx);
        let mut ctx_ids = self.encoder.encode(&context)?;
        if ctx_ids.len() > max_state {
            warnings.push(format!(
                "the state was cut to {max_state} tokens (the \"Context:\" line included)"
            ));
            ctx_ids.truncate(max_state);
        }

        let packed: Vec<Vec<&PlanRow>> = if independent {
            flat.iter().map(|r| vec![r]).collect()
        } else if flat.is_empty() {
            Vec::new()
        } else {
            vec![flat.iter().collect()]
        };
        let mut rows = Vec::with_capacity(packed.len());
        let mut slots = Vec::with_capacity(flat.len());
        for (r, row) in packed.iter().enumerate() {
            let (ids, row_slots) = self.row(&ctx_ids, row)?;
            slots.extend(row_slots.into_iter().map(|pos| RowSlot { row: r, pos }));
            rows.push(ids);
        }

        let mut scoring_rows = Vec::with_capacity(flat.len());
        let mut questions = Vec::with_capacity(index.len());
        for entry in &index {
            let rq = &rqs[entry.question];
            let span = entry.first..entry.first + entry.count;
            for (slot, row) in slots[span.clone()].iter().zip(&flat[span.clone()]) {
                scoring_rows.push(ScoringRow {
                    slot: *slot,
                    labels: self.labels.ids(row.options.len()),
                    question: entry.question,
                    option_ends: Vec::new(),
                });
            }
            let (kind, options) = match (entry.isolated, rq.kind) {
                (true, _) => (
                    QuestionKind::Score {
                        values: None,
                        isolated: true,
                    },
                    entry.count,
                ),
                (false, S1Kind::Choice) => (QuestionKind::Choice, rq.options.len()),
                (false, S1Kind::Noul) => (QuestionKind::Noul { yes: 1 }, rq.options.len()),
                (false, S1Kind::Score) => (
                    QuestionKind::Score {
                        values: None,
                        isolated: false,
                    },
                    rq.options.len(),
                ),
            };
            questions.push(DecisionQuestion {
                id: request.questions[entry.question].id.clone(),
                meta: QuestionMeta { kind, options },
                names: rq.names.clone(),
                legend: rq.legend.clone(),
                scoring_rows: span,
            });
        }
        let head = self.chat.as_ref().map_or(0, |chat| chat.head.len());
        Ok(RenderedDecision {
            family: Family::Decider,
            rows,
            scoring_rows,
            questions,
            warnings,
            state_tokens: ctx_ids.len(),
            state_end: head + ctx_ids.len(),
            joint: None,
        })
    }

    /// One row after the `Context:` ids: its ids and its answer slots.
    fn row(
        &self,
        ctx_ids: &[u32],
        row: &[&PlanRow],
    ) -> Result<(Vec<u32>, Vec<usize>), DecisionError> {
        let multi = row.len() > 1;
        let number = |k: usize| {
            if multi {
                format!(" {}", k + 1)
            } else {
                String::new()
            }
        };
        let mut ids = Vec::new();
        if let Some(chat) = &self.chat {
            ids.extend_from_slice(&chat.head);
        }
        ids.extend_from_slice(ctx_ids);
        let mut slots = Vec::with_capacity(row.len());
        for (k, q) in row.iter().enumerate() {
            let n = q.options.len();
            if !(2..=DECIDER_MAX_OPTIONS).contains(&n) {
                return Err(DecisionError::Request(format!(
                    "2..{DECIDER_MAX_OPTIONS} options required"
                )));
            }
            let mut head = PromptText::template(format!("\n\nQuestion{}: ", number(k)));
            head.push_text(&q.question).push_template("\nOptions:");
            let tail = format!("\nAnswer{}: (", number(k));
            match (&self.chat, n <= NARROW) {
                (None, true) => {
                    let mut piece = head;
                    push_narrow(&mut piece, &q.options);
                    piece.push_template(tail);
                    ids.extend(self.encoder.encode(&piece)?);
                    slots.push(ids.len() - 1);
                }
                (None, false) => {
                    ids.extend(self.encoder.encode(&head)?);
                    self.push_wide(&mut ids, &q.options)?;
                    ids.extend(self.encoder.template(&tail)?);
                    slots.push(ids.len() - 1);
                }
                (Some(_), narrow) => {
                    ids.extend(self.encoder.encode(&head)?);
                    if narrow {
                        let mut lines = PromptText::new();
                        push_narrow(&mut lines, &q.options);
                        ids.extend(self.encoder.encode(&lines)?);
                    } else {
                        self.push_wide(&mut ids, &q.options)?;
                    }
                }
            }
        }
        if let Some(chat) = &self.chat {
            ids.extend_from_slice(&chat.tail);
            for k in 0..row.len() {
                let newline = if k == 0 { "" } else { "\n" };
                let piece = format!("{newline}{ANSWER}{}: (", number(k));
                ids.extend(self.encoder.template(&piece)?);
                slots.push(ids.len() - 1);
            }
        }
        Ok((ids, slots))
    }

    /// Wide options: `"\n("`, the label token, then `") <text>"` encoded on
    /// its own, for each option.
    fn push_wide(&self, ids: &mut Vec<u32>, options: &[PromptText]) -> Result<(), DecisionError> {
        for (label, option) in self.labels.labels.iter().zip(options) {
            ids.extend_from_slice(&self.labels.open_ids);
            ids.push(label.id);
            let mut text = PromptText::template(") ");
            text.push_text(option);
            ids.extend(self.encoder.encode(&text)?);
        }
        Ok(())
    }
}

/// Narrow options: `\n(A) text` lines.
fn push_narrow(text: &mut PromptText, options: &[PromptText]) {
    for (letter, option) in LETTERS.iter().zip(options) {
        text.push_template(format!("\n({letter}) "))
            .push_text(option);
    }
}

/// `neutralize_options`: an option that reads as "none of the above" is
/// shown as [`NEUTRAL_NONE`]. Returns the options and whether any changed.
pub fn neutralize_options(options: &[PromptText]) -> (Vec<PromptText>, bool) {
    let mut changed = false;
    let out = options
        .iter()
        .map(|o| {
            let key = py_strip(&o.text()).to_lowercase();
            if key.starts_with("none of the above")
                || ["none", "n/a", "none of these"].contains(&key.as_str())
            {
                changed = true;
                PromptText::template(NEUTRAL_NONE)
            } else {
                o.clone()
            }
        })
        .collect();
    (out, changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::chat_template::TemplateOrigin;
    use serde_json::json;

    /// The chat layout: template head, `Context:` block, question blocks,
    /// template tail, then one answer piece per question, each slot the
    /// last token of its piece.
    #[test]
    fn chat_layout_wraps_the_plain_pieces() {
        let template = JinjaTemplate::new(
            "{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n\
             {{ m.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}\
             <|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}",
            TemplateOrigin::Inline,
            "",
            "",
        )
        .expect("compiles");
        let config = DeciderConfig {
            layout: Layout::Chat,
            neutralize_none: false,
            isolated_levels: false,
            schema_first: false,
        };
        let tok = crate::tokenizer::from_pretrained("llama3").expect("bundled vocabulary loads");
        let r = DeciderRenderer::new(tok, config, Some(&template)).expect("chat layout");
        let body = json!({"state": "s", "independent": false, "questions": {
            "a": {"type": "noul", "instructions": "Q?"},
            "b": {"type": "choice", "instructions": "Pick", "criteria": {"x": null, "y": null}}}});
        let req = JevRequest::parse(&body, Family::Decider).expect("parses");
        let out = r.render(&req).expect("renders");
        assert_eq!(out.rows.len(), 1);
        let row = &out.rows[0];
        let enc = &r.encoder;
        let head = enc
            .template("<|start_header_id|>user<|end_header_id|>\n\n")
            .expect("encodes");
        assert!(row.starts_with(&head), "{row:?}");
        let chat = r.chat.as_ref().expect("the chat layout");
        assert_eq!(out.state_end, chat.head.len() + out.state_tokens);
        let question = enc.template("\n\nQuestion").expect("encodes");
        assert!(row[out.state_end..].starts_with(&question), "{row:?}");
        let first = enc.template("Answer 1: (").expect("encodes");
        let second = enc.template("\nAnswer 2: (").expect("encodes");
        assert!(row.ends_with(&[first, second.clone()].concat()), "{row:?}");
        assert_eq!(out.scoring_rows[1].slot.pos, row.len() - 1);
        assert_eq!(out.scoring_rows[0].slot.pos, row.len() - 1 - second.len());
        assert_eq!(out.questions[0].meta.kind, QuestionKind::Noul { yes: 1 });
        assert_eq!(out.scoring_rows[1].labels, r.labels.ids(2));
    }

    #[test]
    fn neutralize_rewrites_abstain_options() {
        let opts = [
            PromptText::caller(" None of the above: x"),
            PromptText::caller("N/A"),
            PromptText::caller("other"),
        ];
        let (out, changed) = neutralize_options(&opts);
        assert!(changed);
        let texts: Vec<String> = out.iter().map(PromptText::text).collect();
        assert_eq!(texts, [NEUTRAL_NONE, NEUTRAL_NONE, "other"]);
    }
}
