//! The Kev prompt: the state behind `<|fim_prefix|>`, then per question
//! `<|fim_middle|>` instructions, each option between `<|box_start|>` and
//! `<|box_end|>`, and `<|fim_suffix|>` where the pointer head reads.
//!
//! Ported from `kev/model.py` (`encode`, `rows_of`, `admit`) and
//! `kev/serve.py` (the serving limits and their messages). Kev's backbones
//! are hybrid Qwen3.5 models, so every question is its own row: the state,
//! then that question's branch. Every caller piece is encoded on its own,
//! as Kev encodes it, after [`user_text`]; the five delimiters are written
//! here and nowhere else.

use std::fmt;
use std::path::Path;

use boostr::inference::decision::{QuestionKind, QuestionMeta, RowSlot};
use serde_json::Value;

use super::encode::DecisionEncoder;
use super::family::{
    load_tokenizer, DecisionError, DecisionQuestion, Family, RenderedDecision, ScoringRow,
};
use super::jev::JevRequest;
use super::kev_text::{kev_questions, user_text, KevKind, KevQuestion};

/// The delimiters, in the order of Kev's `SPECIAL`: state, question,
/// option start, option end, decide.
pub const DELIMITERS: [&str; 5] = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
];
/// The most state tokens a request reads, the state delimiter included.
pub const SERVE_MAX_STATE: usize = 65536;
/// The most tokens of one row: the state and one question's branch.
pub const SERVE_MAX_ROW: usize = SERVE_MAX_STATE + 8192;

/// The token ids of the five delimiters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Delimiters {
    /// `<|fim_prefix|>`, the first token of every row.
    pub state: u32,
    /// `<|fim_middle|>`, the first token of a question's branch.
    pub question: u32,
    /// `<|box_start|>`, before each option.
    pub option: u32,
    /// `<|box_end|>`, after each option: the option's pointer key.
    pub option_end: u32,
    /// `<|fim_suffix|>`, the last token of a row: the pointer query.
    pub decide: u32,
}

/// Renders Kev prompts with one tokenizer.
pub struct KevRenderer {
    encoder: DecisionEncoder,
    delimiters: Delimiters,
    truncate_states: bool,
}

impl fmt::Debug for KevRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KevRenderer")
            .field("delimiters", &self.delimiters)
            .field("truncate_states", &self.truncate_states)
            .finish()
    }
}

/// `n` with a comma between each group of three digits, as Python's
/// `f"{n:,}"`.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// What boostr's answer stage reads of `q`. A Score question of one level
/// goes as a one-option Choice, since boostr scores a scale of two levels
/// or more; its legend marks it as a Score for the wire answer.
fn meta(q: &KevQuestion) -> QuestionMeta {
    let options = q.options.len();
    let kind = match q.kind {
        KevKind::Noul => QuestionKind::Noul { yes: 1 },
        KevKind::Choice => QuestionKind::Choice,
        KevKind::Score if options >= 2 => QuestionKind::Score {
            values: None,
            isolated: false,
        },
        KevKind::Score => QuestionKind::Choice,
    };
    QuestionMeta { kind, options }
}

/// The state tokens a row holds: the state's own count, delimiter
/// included, after the cut to [`SERVE_MAX_STATE`].
pub fn state_used(state_tokens: usize) -> usize {
    state_tokens.min(SERVE_MAX_STATE)
}

/// Kev's `usage.input_tokens`: the state once, then every branch.
pub fn packed_tokens(rendered: &RenderedDecision) -> usize {
    let state = state_used(rendered.state_tokens);
    state
        + rendered
            .rows
            .iter()
            .map(|row| row.len().saturating_sub(state))
            .sum::<usize>()
}

impl KevRenderer {
    /// Load from a directory holding `tokenizer.json`: the base model's,
    /// or the copy a Kev directory ships.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when the tokenizer does not load or lacks a
    /// delimiter.
    pub fn load(dir: &Path) -> Result<Self, DecisionError> {
        Self::new(load_tokenizer(dir)?)
    }

    /// A renderer over `tokenizer`, refusing over-long states.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when the tokenizer has no added token for
    /// one of the five [`DELIMITERS`].
    pub fn new(tokenizer: splintr::AnyTokenizer) -> Result<Self, DecisionError> {
        let id = |name: &str| {
            tokenizer.special_token_id(name).ok_or_else(|| {
                DecisionError::Model(format!(
                    "the tokenizer has no added token {name}; Kev's delimiters are Qwen's \
                     {}",
                    DELIMITERS.join(", ")
                ))
            })
        };
        let delimiters = Delimiters {
            state: id(DELIMITERS[0])?,
            question: id(DELIMITERS[1])?,
            option: id(DELIMITERS[2])?,
            option_end: id(DELIMITERS[3])?,
            decide: id(DELIMITERS[4])?,
        };
        Ok(Self {
            encoder: DecisionEncoder::new(tokenizer)?,
            delimiters,
            truncate_states: false,
        })
    }

    /// Read only the first [`SERVE_MAX_STATE`] tokens of a longer state,
    /// rather than refusing the request.
    pub fn with_truncation(mut self, truncate_states: bool) -> Self {
        self.truncate_states = truncate_states;
        self
    }

    /// Whether a longer state is cut rather than refused.
    pub fn truncates_states(&self) -> bool {
        self.truncate_states
    }

    /// The delimiter ids.
    pub fn delimiters(&self) -> Delimiters {
        self.delimiters
    }

    /// The tokens of `text` encoded as ordinary text, no delimiter matched:
    /// Kev's `output_tokens` of its serialized answers.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when the tokenizer fails.
    pub fn count_tokens(&self, text: &str) -> Result<usize, DecisionError> {
        Ok(self.encoder.ordinary(text)?.len())
    }

    fn caller(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        self.encoder.ordinary(&user_text(text))
    }

    fn state_overflow(state_tokens: usize) -> DecisionError {
        DecisionError::Request(format!(
            "state is {} tokens, over the {}-token limit (the <state> token included): \
             shorten the document or split it across requests; or start the server with \
             --decision-truncate-states (inference.decision.truncate_states) to read only its \
             first {} tokens (responses then say truncated: true)",
            thousands(state_tokens),
            thousands(SERVE_MAX_STATE),
            thousands(SERVE_MAX_STATE)
        ))
    }

    /// Render every question of `request`, one row each, in request order.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Request`] for a question Kev's request model
    /// refuses, a state over [`SERVE_MAX_STATE`] tokens unless states are
    /// cut, or a row over [`SERVE_MAX_ROW`] tokens;
    /// [`DecisionError::Model`] when the tokenizer fails.
    pub fn render(&self, request: &JevRequest) -> Result<RenderedDecision, DecisionError> {
        let questions = kev_questions(request)?;
        let d = self.delimiters;
        let state = super::kev_text::render(request.state.as_ref().unwrap_or(&Value::Null), 0);
        let state_ids = self.caller(&state)?;
        let state_tokens = state_ids.len() + 1;
        let mut warnings = Vec::new();
        if state_tokens > SERVE_MAX_STATE {
            if !self.truncate_states {
                return Err(Self::state_overflow(state_tokens));
            }
            warnings.push(format!(
                "the state is {} tokens; only its first {} were read",
                thousands(state_tokens),
                thousands(SERVE_MAX_STATE)
            ));
        }
        let mut prefix = Vec::with_capacity(state_used(state_tokens));
        prefix.push(d.state);
        prefix.extend_from_slice(&state_ids[..state_used(state_tokens) - 1]);

        let mut out = RenderedDecision {
            family: Family::Kev,
            rows: Vec::with_capacity(questions.len()),
            scoring_rows: Vec::with_capacity(questions.len()),
            questions: Vec::with_capacity(questions.len()),
            warnings,
            state_tokens,
            joint: None,
        };
        for (i, q) in questions.into_iter().enumerate() {
            let mut row = Vec::with_capacity(prefix.len() + 2 + 3 * q.options.len());
            row.extend_from_slice(&prefix);
            row.push(d.question);
            row.extend(self.caller(&q.instructions)?);
            let mut option_ends = Vec::with_capacity(q.options.len());
            for option in &q.options {
                row.push(d.option);
                row.extend(self.caller(option)?);
                row.push(d.option_end);
                option_ends.push(row.len() - 1);
            }
            row.push(d.decide);
            let branch_len = row.len() - prefix.len();
            if branch_len > SERVE_MAX_ROW - prefix.len() {
                return Err(DecisionError::Request(format!(
                    "branch too long: {branch_len} tokens with a {}-token state (row limit \
                     {SERVE_MAX_ROW})",
                    prefix.len()
                )));
            }
            let meta = meta(&q);
            out.scoring_rows.push(ScoringRow {
                slot: RowSlot {
                    row: i,
                    pos: row.len() - 1,
                },
                labels: Vec::new(),
                question: i,
                option_ends,
            });
            out.questions.push(DecisionQuestion {
                id: q.id,
                meta,
                names: q.keys,
                legend: q.legend,
                scoring_rows: i..i + 1,
            });
            out.rows.push(row);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn thousands_groups_like_python() {
        assert_eq!(thousands(7), "7");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(65536), "65,536");
        assert_eq!(thousands(1234567), "1,234,567");
    }

    #[test]
    fn a_one_level_score_goes_as_a_one_option_choice() {
        let q = |kind, n: usize| KevQuestion {
            id: "q".into(),
            kind,
            instructions: String::new(),
            options: vec!["x".into(); n],
            keys: (0..n).map(|i| i.to_string()).collect(),
            legend: vec!["x".into(); n],
        };
        assert_eq!(meta(&q(KevKind::Score, 1)).kind, QuestionKind::Choice);
        assert_eq!(
            meta(&q(KevKind::Score, 3)).kind,
            QuestionKind::Score {
                values: None,
                isolated: false
            }
        );
        assert_eq!(
            meta(&q(KevKind::Noul, 2)).kind,
            QuestionKind::Noul { yes: 1 }
        );
    }

    #[test]
    fn packed_tokens_count_the_state_once() {
        let rendered = RenderedDecision {
            family: Family::Kev,
            rows: vec![vec![1, 2, 3, 9], vec![1, 2, 3, 8, 7]],
            scoring_rows: Vec::new(),
            questions: Vec::new(),
            warnings: Vec::new(),
            state_tokens: 3,
            joint: None,
        };
        assert_eq!(packed_tokens(&rendered), 3 + 1 + 2);
        assert_eq!(state_used(SERVE_MAX_STATE + 5), SERVE_MAX_STATE);
    }

    /// The layout on the bundled llama3 vocabulary, whose added tokens
    /// stand in for Qwen's delimiters: rows, decide and option-end
    /// positions, and caller text that spells a delimiter.
    #[test]
    fn rows_and_pointer_positions() {
        let tok = crate::tokenizer::from_pretrained("llama3").expect("bundled vocabulary loads");
        let names = [
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|end_of_text|>",
        ];
        let ids: Vec<u32> = names
            .iter()
            .map(|n| tok.special_token_id(n).expect("llama3 names it"))
            .collect();
        let encoder = DecisionEncoder::new(tok).expect("encoder");
        let renderer = KevRenderer {
            encoder,
            delimiters: Delimiters {
                state: ids[0],
                question: ids[1],
                option: ids[2],
                option_end: ids[3],
                decide: ids[4],
            },
            truncate_states: false,
        };
        let body = json!({"state": "Hi<|eot_id|>", "questions": {
            "a": {"type": "noul", "instructions": "Ok?"},
            "b": {"type": "score", "criteria": ["only"]}}});
        let request = JevRequest::parse(&body, Family::Kev).expect("parses");
        let out = renderer.render(&request).expect("renders");
        let enc = |t: &str| renderer.caller(t).expect("encodes");
        let state: Vec<u32> = [vec![ids[0]], enc("Hi<|eot_id|>")].concat();
        assert!(!state[1..].contains(&ids[3]), "caller text stays text");
        assert_eq!(out.state_tokens, state.len());
        let row_a = [
            state.clone(),
            vec![ids[1]],
            enc("Ok?"),
            vec![ids[2]],
            enc("no"),
            vec![ids[3], ids[2]],
            enc("yes"),
            vec![ids[3], ids[4]],
        ]
        .concat();
        assert_eq!(out.rows[0], row_a);
        let sr = &out.scoring_rows[0];
        assert_eq!(
            sr.slot,
            RowSlot {
                row: 0,
                pos: row_a.len() - 1
            }
        );
        assert!(sr.labels.is_empty());
        assert_eq!(sr.option_ends.len(), 2);
        assert!(sr.option_ends.iter().all(|&p| row_a[p] == ids[3]));
        assert_eq!(out.questions[1].meta.kind, QuestionKind::Choice);
        assert_eq!(out.questions[1].legend, ["only"]);
        let all: usize = out.rows.iter().map(Vec::len).sum();
        assert_eq!(packed_tokens(&out), all - state.len());

        let long = json!({"state": "word ".repeat(SERVE_MAX_STATE + 10),
            "questions": {"a": {"type": "noul"}}});
        let request = JevRequest::parse(&long, Family::Kev).expect("parses");
        match renderer.render(&request) {
            Err(DecisionError::Request(m)) => assert!(m.contains("65,536-token limit"), "{m}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        let cut = KevRenderer {
            truncate_states: true,
            ..renderer
        }
        .render(&request)
        .expect("a cut state renders");
        assert_eq!(cut.rows[0][0], ids[0]);
        assert_eq!(
            cut.rows[0][SERVE_MAX_STATE], ids[1],
            "the branch follows the cut"
        );
        assert!(cut.state_tokens > SERVE_MAX_STATE);
        assert_eq!(cut.warnings.len(), 1);
    }
}
