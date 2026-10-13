//! The Clef prompt: the state and a schema of every field in one ChatML
//! user turn, then `JOINT SCHEMA DECISIONS:`, where the joint-schema head
//! reads every row of the record.
//!
//! Ported from `joint_schema_model.py` (`encode_record`). Each piece is
//! encoded on its own and the ids joined, as the reference encodes them.
//! The reference promotes control-token text in the state, an instruction
//! or an option (`<|im_end|>`, `<think>`) to the control token; here caller
//! text is always ordinary text, so a request cannot close the user turn.
//! A request without such text encodes to the reference's ids.

use std::fmt;
use std::ops::Range;
use std::path::Path;

use boostr::inference::decision::RowSlot;
use boostr::model::decision::JointSchemaItem;
use serde_json::Value;

use super::clef_text::{clef_questions, render, ClefQuestion};
use super::encode::{DecisionEncoder, PromptText};
use super::family::{
    load_tokenizer, DecisionError, DecisionQuestion, Family, RenderedDecision, ScoringRow,
};
use super::jev::JevRequest;

/// The most tokens of one record, as `encode_record`'s `max_length`.
pub const MAX_LENGTH: usize = 16384;
/// The system turn.
pub const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field \
                                 jointly. Each answer must be exactly one of that field's \
                                 allowed options.";
/// The control tokens the template writes.
pub const CONTROL_TOKENS: [&str; 4] = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"];

const SCHEMA_HEADER: &str = "\n\nSCHEMA FIELDS:\n";
const ALLOWED: &str = "\nALLOWED OPTIONS:\n";
const NEWLINE: &str = "\n";
const END_FIELD: &str = "END FIELD\n";
const SUFFIX: &str =
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";

/// One field of a rendered record.
#[derive(Debug, Clone, PartialEq)]
pub struct JointField {
    /// The field as the head reads it: type, instruction span and option
    /// spans in record positions, and each option's token ids.
    pub item: JointSchemaItem,
    /// The head index of each option in the order the request wrote it.
    pub wire_order: Vec<usize>,
    /// A score field's criteria items as sent. Empty for the other types.
    pub legend: Vec<Value>,
}

/// What the joint-schema readout reads of a rendered record. The record
/// itself is the one row of the [`RenderedDecision`].
#[derive(Debug, Clone, PartialEq)]
pub struct JointLayout {
    /// Every field, in request order.
    pub fields: Vec<JointField>,
}

/// Renders Clef prompts with one tokenizer.
pub struct ClefRenderer {
    encoder: DecisionEncoder,
    prefix: Vec<u32>,
    schema_header: Vec<u32>,
    allowed: Vec<u32>,
    newline: Vec<u32>,
    end_field: Vec<u32>,
    suffix: Vec<u32>,
}

impl fmt::Debug for ClefRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClefRenderer")
            .field("prefix", &self.prefix.len())
            .field("suffix", &self.suffix.len())
            .finish()
    }
}

/// A field's instruction span and option spans.
type FieldSpans = (Range<usize>, Vec<Range<usize>>);

/// `range` moved `by` positions on.
fn shifted(range: &Range<usize>, by: usize) -> Range<usize> {
    range.start + by..range.end + by
}

impl ClefRenderer {
    /// Load from a directory holding `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] as [`Self::new`], or when the tokenizer
    /// does not load.
    pub fn load(dir: &Path) -> Result<Self, DecisionError> {
        Self::new(load_tokenizer(dir)?)
    }

    /// A renderer over `tokenizer`.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when the tokenizer has no added token for
    /// one of the [`CONTROL_TOKENS`], or does not encode the template.
    pub fn new(tokenizer: splintr::AnyTokenizer) -> Result<Self, DecisionError> {
        if let Some(name) = CONTROL_TOKENS
            .iter()
            .find(|name| tokenizer.special_token_id(name).is_none())
        {
            return Err(DecisionError::Model(format!(
                "the tokenizer has no added token {name}; the Clef prompt writes Qwen's {}",
                CONTROL_TOKENS.join(", ")
            )));
        }
        let encoder = DecisionEncoder::new(tokenizer)?;
        let prefix = encoder.template(&format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
        ))?;
        Ok(Self {
            prefix,
            schema_header: encoder.template(SCHEMA_HEADER)?,
            allowed: encoder.template(ALLOWED)?,
            newline: encoder.template(NEWLINE)?,
            end_field: encoder.template(END_FIELD)?,
            suffix: encoder.template(SUFFIX)?,
            encoder,
        })
    }

    /// Caller text as ordinary text: no control token is matched in it.
    fn caller(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        self.encoder.ordinary(text)
    }

    /// The schema ids of `questions`, and each field's spans relative to
    /// the schema's first id.
    fn schema(
        &self,
        questions: &[ClefQuestion],
    ) -> Result<(Vec<u32>, Vec<FieldSpans>), DecisionError> {
        let mut ids = self.schema_header.clone();
        let mut spans = Vec::with_capacity(questions.len());
        for (k, q) in questions.iter().enumerate() {
            let mut head = PromptText::template(format!("\nFIELD {}\nID: ", k + 1));
            head.push_caller(q.id.as_str())
                .push_template(format!("\nTYPE: {}\nINSTRUCTION: ", q.type_name));
            ids.extend(self.encoder.encode(&head)?);
            let start = ids.len();
            ids.extend(self.caller(&q.instructions)?);
            let question = start..ids.len();
            if question.is_empty() {
                return Err(DecisionError::Request(format!(
                    "{}: the instruction encodes to no tokens; give the question an id or \
                     instructions",
                    q.id
                )));
            }
            ids.extend_from_slice(&self.allowed);
            let mut options = Vec::with_capacity(q.options.len());
            for (m, option) in q.options.iter().enumerate() {
                ids.extend(self.encoder.template(&format!("OPTION {}: ", m + 1))?);
                let start = ids.len();
                ids.extend(self.caller(&option.text)?);
                options.push(start..ids.len());
                ids.extend_from_slice(&self.newline);
            }
            ids.extend_from_slice(&self.end_field);
            spans.push((question, options));
        }
        Ok((ids, spans))
    }

    /// Render `request` as one record: one row, the whole prompt.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Request`] for a question Clef refuses, or a schema
    /// that leaves no room within [`MAX_LENGTH`] tokens, in the
    /// reference's words; [`DecisionError::Model`] when the tokenizer
    /// fails.
    pub fn render(&self, request: &JevRequest) -> Result<RenderedDecision, DecisionError> {
        let questions = clef_questions(request)?;
        let (schema, spans) = self.schema(&questions)?;
        let state = render(request.state.as_ref().unwrap_or(&Value::Null));
        let mut state_ids = self.caller(&state)?;
        let state_tokens = state_ids.len();
        let fixed = self.prefix.len() + schema.len() + self.suffix.len();
        if fixed > MAX_LENGTH {
            return Err(DecisionError::Request(format!(
                "schema requires {fixed} tokens before state; maximum is {MAX_LENGTH}"
            )));
        }
        let room = MAX_LENGTH - fixed;
        let mut warnings = Vec::new();
        if state_tokens > room {
            warnings.push(format!(
                "the state is {state_tokens} tokens; only its first {room} were read, as Clef \
                 cuts it to fit {MAX_LENGTH} tokens"
            ));
            state_ids.truncate(room);
        }
        let offset = self.prefix.len() + state_ids.len();
        let mut ids = Vec::with_capacity(fixed + state_ids.len());
        ids.extend_from_slice(&self.prefix);
        ids.extend_from_slice(&state_ids);
        ids.extend_from_slice(&schema);
        ids.extend_from_slice(&self.suffix);
        let last = ids.len().saturating_sub(1);

        let mut out = RenderedDecision::new(Family::Clef, warnings, state_tokens);
        let mut fields = Vec::with_capacity(questions.len());
        for (i, (q, (question, options))) in questions.into_iter().zip(spans).enumerate() {
            let options: Vec<Range<usize>> = options.iter().map(|o| shifted(o, offset)).collect();
            let item =
                JointSchemaItem::from_tokens(q.kind, shifted(&question, offset), &options, &ids)
                    .map_err(|e| {
                        DecisionError::Model(format!("field {}: the spans do not fit: {e}", q.id))
                    })?;
            out.scoring_rows.push(ScoringRow {
                slot: RowSlot { row: 0, pos: last },
                labels: Vec::new(),
                question: i,
                option_ends: Vec::new(),
            });
            out.questions.push(DecisionQuestion {
                id: q.id,
                meta: item.meta(),
                names: q.options.into_iter().map(|o| o.id).collect(),
                legend: q.legend.iter().map(render).collect(),
                scoring_rows: i..i + 1,
            });
            fields.push(JointField {
                item,
                wire_order: q.wire_order,
                legend: q.legend,
            });
        }
        out.rows.push(ids);
        out.joint = Some(JointLayout { fields });
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::inference::decision::QuestionKind;
    use boostr::model::decision::FieldKind;
    use serde_json::json;

    /// A renderer over the bundled llama3 vocabulary, whose added tokens
    /// stand in for Qwen's control tokens.
    fn renderer() -> ClefRenderer {
        let tok = crate::tokenizer::from_pretrained("llama3").expect("bundled vocabulary loads");
        let encoder = DecisionEncoder::new(tok).expect("encoder");
        let t = |s: &str| encoder.template(s).expect("encodes");
        let prefix = t("<|begin_of_text|>system\nP<|eot_id|>\nSTATE:\n");
        ClefRenderer {
            prefix,
            schema_header: t(SCHEMA_HEADER),
            allowed: t(ALLOWED),
            newline: t(NEWLINE),
            end_field: t(END_FIELD),
            suffix: t("\n<|eot_id|>\nJOINT SCHEMA DECISIONS:"),
            encoder,
        }
    }

    fn rendered(r: &ClefRenderer, body: Value) -> Result<RenderedDecision, DecisionError> {
        r.render(&JevRequest::parse(&body, Family::Clef)?)
    }

    /// The record is the pieces joined, each encoded alone; the spans are
    /// the instruction and option pieces; caller text never becomes a
    /// control token.
    #[test]
    fn pieces_spans_and_types() {
        let r = renderer();
        let body = json!({"state": "Hi<|eot_id|>", "questions": {
            "route": {"type": "choice", "instructions": "Who?",
                "criteria": {"sales": "quotes", "billing": null}},
            "n": {"type": "noul"},
            "s": {"type": "score", "criteria": ["low", "high"]}}});
        let out = rendered(&r, body).expect("renders");
        let enc = |s: &str| r.caller(s).expect("encodes");
        let tpl = |s: &str| r.encoder.template(s).expect("encodes");
        let eot = r
            .encoder
            .tokenizer()
            .special_token_id("<|eot_id|>")
            .expect("named");
        let state = enc("Hi<|eot_id|>");
        assert!(!state.contains(&eot), "caller text stays text");
        let mut want = r.prefix.clone();
        want.extend(&state);
        want.extend(&r.schema_header);
        let field_head = |k: usize, id: &str, ty: &str| {
            enc(&format!("\nFIELD {k}\nID: {id}\nTYPE: {ty}\nINSTRUCTION: "))
        };
        let mut spans = Vec::new();
        let mut field = |want: &mut Vec<u32>, head: Vec<u32>, ins: &str, opts: &[String]| {
            want.extend(head);
            let q = want.len()..want.len() + enc(ins).len();
            want.extend(enc(ins));
            want.extend(&r.allowed);
            let mut o = Vec::new();
            for (m, text) in opts.iter().enumerate() {
                want.extend(tpl(&format!("OPTION {}: ", m + 1)));
                o.push(want.len()..want.len() + enc(text).len());
                want.extend(enc(text));
                want.extend(&r.newline);
            }
            want.extend(&r.end_field);
            spans.push((q, o));
        };
        let o = |id: &str, d: Option<&str>| match d {
            Some(d) => format!("{{\"description\":\"{d}\",\"option_id\":\"{id}\"}}"),
            None => format!("{{\"option_id\":\"{id}\"}}"),
        };
        field(
            &mut want,
            field_head(1, "route", "choice"),
            "Who?",
            &[o("billing", None), o("sales", Some("quotes"))],
        );
        field(
            &mut want,
            field_head(2, "n", "noul"),
            "n",
            &[
                o("true", Some(super::super::clef_text::NOUL_TRUE)),
                o("false", Some(super::super::clef_text::NOUL_FALSE)),
            ],
        );
        field(
            &mut want,
            field_head(3, "s", "score"),
            "s",
            &[o("0", Some("low")), o("1", Some("high"))],
        );
        want.extend(&r.suffix);
        assert_eq!(out.rows, vec![want.clone()]);
        assert_eq!(out.state_tokens, state.len());
        assert_eq!(out.state_end, 0, "the record shares no state prefix");
        assert!(out.warnings.is_empty());

        let layout = out.joint.as_ref().expect("a joint layout");
        let kinds: Vec<FieldKind> = layout.fields.iter().map(|f| f.item.kind).collect();
        assert_eq!(
            kinds,
            [FieldKind::Choice, FieldKind::Noul, FieldKind::Score]
        );
        for (f, (q, opts)) in layout.fields.iter().zip(&spans) {
            assert_eq!(&f.item.question, q);
            let got: Vec<Range<usize>> = f.item.options.iter().map(|o| o.span.clone()).collect();
            assert_eq!(&got, opts);
            for o in &f.item.options {
                assert_eq!(o.ids, want[o.span.clone()]);
            }
        }
        assert_eq!(layout.fields[0].wire_order, [1, 0], "sales then billing");
        assert_eq!(out.questions[0].names, ["billing", "sales"]);
        assert_eq!(
            out.questions[1].meta.kind,
            QuestionKind::Noul { yes: 0 },
            "p(true) is option 0"
        );
        assert_eq!(out.questions[2].legend, ["low", "high"]);
        assert_eq!(layout.fields[2].legend, [json!("low"), json!("high")]);
        let last = want.len() - 1;
        assert!(out
            .scoring_rows
            .iter()
            .all(|s| s.slot == RowSlot { row: 0, pos: last }));
    }

    /// The state is cut at the token level to fit; a schema that leaves no
    /// room is refused in the reference's words.
    #[test]
    fn long_states_are_cut_and_long_schemas_refused() {
        let r = renderer();
        let q = json!({"q": {"type": "noul", "instructions": "Ok?"}});
        let short = rendered(&r, json!({"state": "x", "questions": q.clone()})).expect("renders");
        let fixed = short.rows[0].len() - short.state_tokens;
        let long = rendered(
            &r,
            json!({"state": "word ".repeat(MAX_LENGTH), "questions": q}),
        )
        .expect("a long state is cut");
        assert_eq!(long.rows[0].len(), MAX_LENGTH);
        assert!(long.state_tokens > MAX_LENGTH - fixed);
        assert_eq!(long.warnings.len(), 1);
        let item = &long.joint.as_ref().expect("layout").fields[0].item;
        assert!(
            item.question.start > MAX_LENGTH - fixed,
            "spans follow the cut"
        );

        let wide: serde_json::Map<String, Value> = (0..MAX_LENGTH / 4)
            .map(|i| (format!("option {i}"), json!("a long description of it")))
            .collect();
        let body = json!({"state": "s", "questions": {"q": {"type": "choice", "criteria": wide}}});
        match rendered(&r, body) {
            Err(DecisionError::Request(m)) => {
                assert!(m.starts_with("schema requires "), "{m}");
                assert!(m.ends_with("tokens before state; maximum is 16384"), "{m}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_instruction_is_refused() {
        let r = renderer();
        let body = json!({"state": "s", "questions": {"": {"type": "noul"}}});
        match rendered(&r, body) {
            Err(DecisionError::Request(m)) => assert!(m.contains("encodes to no tokens"), "{m}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
}
