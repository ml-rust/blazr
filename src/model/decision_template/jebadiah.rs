//! The Jebadiah prompt: AINode's `/v1/systemone` rendering inside the
//! model's chat template, thinking off, one row per question, the label
//! read at the last prompt token.
//!
//! Ported from the model repository's `scripts/jebadiah_prompt.py`
//! (`Renderer.render`, `Renderer.render_messages`, `wire_keys`,
//! `reorder_choice`) over `scripts/ainode_prompt_verbatim.py`, and from the
//! checks `jebadiah_server` runs before it renders (`translate_questions`,
//! `decide_questions`, the prompt contract).
//!
//! An over-long prompt is refused, as the served route refuses it. The
//! `Renderer` the model was trained with cut the state instead; that cut is
//! a training-time behaviour and is not reproduced.

use std::fmt;
use std::path::Path;

use boostr::inference::decision::{QuestionKind, QuestionMeta, RowSlot};
use serde_json::{Map, Value};

use super::ainode::{
    ainode_labels, check_options, criteria_pairs, option_text, question_kind, translate_one,
    user_message, Kind, Translated, MAX_CRITERIA, PROMPT_SOURCE_SHA256, SYSTEM_PROMPT,
};
use super::encode::{ChatFrame, DecisionEncoder, PromptText};
use super::family::{
    load_template, load_tokenizer, read_json, DecisionError, DecisionQuestion, Family,
    RenderedDecision, ScoringRow,
};
use super::jev::JevRequest;
use super::labels::{option_label, AinodeLabels};
use super::py_text::{py_str, py_strip};
use crate::model::chat_template::jinja_json::python_json;
use crate::model::chat_template::{JinjaTemplate, RenderOptions};

/// The prompt-token cap when the request names none: the `Renderer`'s
/// default.
pub const DEFAULT_MAX_TOKENS: usize = 2048;

/// The chat-template variables the model was trained with.
fn template_options() -> RenderOptions {
    let mut extra = Map::new();
    extra.insert("enable_thinking".into(), Value::Bool(false));
    extra.insert("thinking".into(), Value::Bool(false));
    RenderOptions {
        add_generation_prompt: true,
        tools: None,
        extra,
    }
}

/// `serialize_state`: a string verbatim, nothing for null, anything else
/// as compact JSON with sorted keys.
pub fn serialize_state(state: Option<&Value>) -> String {
    match state {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => python_json(other, None, Some((",", ":")), false, true),
    }
}

/// Renders Jebadiah prompts with one tokenizer and chat template.
pub struct JebadiahRenderer {
    encoder: DecisionEncoder,
    frame: ChatFrame,
    labels: AinodeLabels,
}

impl fmt::Debug for JebadiahRenderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JebadiahRenderer")
            .field("max_options", &self.max_options())
            .field("max_options_extended", &self.max_options_extended())
            .finish()
    }
}

/// One question rendered to a row.
struct Row {
    ids: Vec<u32>,
    labels: Vec<u32>,
    keys: Vec<String>,
    kind: Kind,
}

impl JebadiahRenderer {
    /// Load from a model directory: `tokenizer.json`, the chat template
    /// (`chat_template.jinja`, else `tokenizer_config.json`) and, when
    /// present, `prompt_contract.json`.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when a file is missing or does not load, or
    /// the contract names a prompt this renderer does not render.
    pub fn load(dir: &Path) -> Result<Self, DecisionError> {
        let tokenizer = load_tokenizer(dir)?;
        let template = load_template(dir)?;
        let renderer = Self::new(tokenizer, &template)?;
        let path = dir.join("prompt_contract.json");
        if path.is_file() {
            renderer.check_contract(&read_json(&path)?)?;
        }
        Ok(renderer)
    }

    /// A renderer over `tokenizer` and the model's chat `template`.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] when the template does not frame a user turn,
    /// or fewer than 26 labels are single tokens after it.
    pub fn new(
        tokenizer: splintr::AnyTokenizer,
        template: &JinjaTemplate,
    ) -> Result<Self, DecisionError> {
        let encoder = DecisionEncoder::new(tokenizer)?;
        let frame = ChatFrame::render(template, Some(SYSTEM_PROMPT), &template_options())?;
        let probe_options = [PromptText::template("o1"), PromptText::template("o2")];
        let mut probe = PromptText::template(&frame.head);
        probe
            .push_text(&user_message("x", "q?", &probe_options, &ainode_labels(2)))
            .push_template(&frame.tail);
        let base = encoder.encode(&probe)?;
        let labels = AinodeLabels::probe(&encoder, &base, |label| {
            let mut with = probe.clone();
            with.push_template(label);
            encoder.encode(&with)
        })?;
        Ok(Self {
            encoder,
            frame,
            labels,
        })
    }

    /// AINode labels that are one token at the answer slot.
    pub fn max_options(&self) -> usize {
        self.labels.ainode.len()
    }

    /// Labels of the extended alphabet.
    pub fn max_options_extended(&self) -> usize {
        self.labels.extended.len()
    }

    /// A prompt contract (`prompt_contract.json`, or `jebadiah.json`'s
    /// `prompt_contract`) must pass [`contract_problems`] with this
    /// tokenizer's label count.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Model`] naming every check that fails.
    pub fn check_contract(&self, contract: &Value) -> Result<(), DecisionError> {
        let problems = contract_problems(contract, Some(self.max_options()));
        if problems.is_empty() {
            Ok(())
        } else {
            Err(DecisionError::Model(format!(
                "prompt contract: {}",
                problems.join("; ")
            )))
        }
    }

    /// Render every question of `request`, one row each, in request order.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Request`] with AINode's message for a question it
    /// refuses, or when a prompt exceeds `max_tokens`.
    pub fn render(&self, request: &JevRequest) -> Result<RenderedDecision, DecisionError> {
        let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let mut warnings = Vec::new();
        let mut translated = Vec::with_capacity(request.questions.len());
        for q in &request.questions {
            let t = self.translate(&q.id, &q.spec, request.order_of(&q.id), &mut warnings)?;
            translated.push(t);
        }
        for (q, (t, _)) in request.questions.iter().zip(&translated) {
            check_options(&q.id, &t.options).map_err(DecisionError::Request)?;
        }
        let state = serialize_state(request.state.as_ref());
        let state_tokens = self.encoder.encode(&PromptText::caller(&state))?.len();

        let mut out = RenderedDecision {
            family: Family::Jebadiah,
            rows: Vec::with_capacity(translated.len()),
            scoring_rows: Vec::with_capacity(translated.len()),
            questions: Vec::with_capacity(translated.len()),
            warnings,
            state_tokens,
        };
        for (i, (q, (t, keys))) in request.questions.iter().zip(translated).enumerate() {
            let legend = match t.kind {
                Kind::Score => t.names.clone(),
                Kind::Choice | Kind::Noul => Vec::new(),
            };
            let row = self.row(&q.id, &state, t, keys, &mut out.warnings)?;
            if row.ids.len() > max_tokens {
                return Err(DecisionError::Request(format!(
                    "question '{}': the prompt is {} tokens, more than the {max_tokens} this \
                     server takes. The state is refused, not cut",
                    q.id,
                    row.ids.len()
                )));
            }
            let kind = match row.kind {
                Kind::Choice => QuestionKind::Choice,
                Kind::Noul => QuestionKind::Noul { yes: 0 },
                Kind::Score => QuestionKind::Score {
                    values: None,
                    isolated: false,
                },
            };
            out.scoring_rows.push(ScoringRow {
                slot: RowSlot {
                    row: i,
                    pos: row.ids.len().saturating_sub(1),
                },
                labels: row.labels,
                question: i,
            });
            out.questions.push(DecisionQuestion {
                id: q.id.clone(),
                meta: QuestionMeta {
                    kind,
                    options: row.keys.len(),
                },
                names: row.keys,
                legend,
                scoring_rows: i..i + 1,
            });
            out.rows.push(row.ids);
        }
        Ok(out)
    }

    /// `Renderer.render`'s translation: the question with its criteria in
    /// `order`, then AINode's `translate_one`, or for a choice wider than
    /// AINode's 20 criteria the same rendering without that cap. Returns the
    /// translation and the wire keys in display order.
    fn translate(
        &self,
        id: &str,
        spec: &Value,
        order: Option<&[String]>,
        warnings: &mut Vec<String>,
    ) -> Result<(Translated, Vec<String>), DecisionError> {
        let request = DecisionError::Request;
        let Value::Object(map) = spec else {
            return Err(request(format!("question '{id}' must be an object")));
        };
        let kind = question_kind(id, map).map_err(request)?;
        let mut map = map.clone();
        if kind == Kind::Choice {
            if let Some(order) = order {
                let reordered = reorder_choice(id, map.get("criteria"), order)?;
                map.insert("criteria".into(), reordered);
            }
        }
        let criteria = map.get("criteria");
        let t = match criteria {
            Some(Value::Object(crit)) if kind == Kind::Choice && crit.len() > MAX_CRITERIA => {
                warnings.push(format!(
                    "question '{id}': {} criteria is more than the {MAX_CRITERIA} AINode's \
                     served route takes; rendered as the model's local read renders it",
                    crit.len()
                ));
                let pairs = criteria_pairs(id, criteria, "choice").map_err(request)?;
                let instructions = match map.get("instructions") {
                    None => String::new(),
                    Some(v) => {
                        py_str(v, &format!("question '{id}': 'instructions'")).map_err(request)?
                    }
                };
                Translated {
                    kind,
                    question: py_strip(&instructions).to_string(),
                    options: pairs
                        .iter()
                        .map(|(n, d)| option_text(n, d.as_deref()))
                        .collect(),
                    names: pairs.into_iter().map(|(n, _)| n).collect(),
                }
            }
            _ => translate_one(id, &Value::Object(map.clone())).map_err(request)?,
        };
        let keys = match kind {
            Kind::Choice => match order {
                Some(order) => order.to_vec(),
                None => match map.get("criteria") {
                    Some(Value::Object(crit)) => crit.keys().cloned().collect(),
                    _ => t.names.clone(),
                },
            },
            Kind::Noul => vec!["true".to_string(), "false".to_string()],
            Kind::Score => (0..t.names.len()).map(|i| i.to_string()).collect(),
        };
        Ok((t, keys))
    }

    /// One question's row: labels, user message, chat template, ids.
    fn row(
        &self,
        id: &str,
        state: &str,
        t: Translated,
        keys: Vec<String>,
        warnings: &mut Vec<String>,
    ) -> Result<Row, DecisionError> {
        let n = t.options.len();
        let ainode = ainode_labels(n);
        let (letters, labels) = if n <= self.max_options() {
            let ids: Vec<u32> = self.labels.ainode[..n].iter().map(|l| l.id).collect();
            (ainode.clone(), ids)
        } else if n <= self.max_options_extended() {
            warnings.push(format!(
                "question '{id}': {n} options exceed the {} single-token AINode labels; the \
                 extended alphabet labels them",
                self.max_options()
            ));
            let chosen = &self.labels.extended[..n];
            (
                chosen.iter().map(|l| l.text.clone()).collect(),
                chosen.iter().map(|l| l.id).collect(),
            )
        } else {
            return Err(DecisionError::Request(format!(
                "{n} options exceed the {} single-token labels",
                self.max_options_extended()
            )));
        };
        let message = user_message(state, &t.question, &t.options, &letters);
        if letters != ainode {
            let swapped = swap_labels(
                &user_message(state, &t.question, &t.options, &ainode).text(),
                &letters,
            );
            if swapped.as_deref() != Some(message.text().as_str()) {
                return Err(DecisionError::Request(format!(
                    "question '{id}': the state or an option holds a line the label swap of the \
                     extended alphabet would rewrite; it is refused rather than rendered \
                     differently from the reference"
                )));
            }
        }
        let mut prompt = PromptText::template(&self.frame.head);
        prompt.push_text(&message).push_template(&self.frame.tail);
        Ok(Row {
            ids: self.encoder.encode(&prompt)?,
            labels,
            keys,
            kind: t.kind,
        })
    }
}

/// What a prompt contract says that this renderer does not render: another
/// AINode copy, another label count (`labels`, when the tokenizer is
/// known), or thinking on. The chat template hash is the caller's check:
/// the renderer keeps no template source. Empty when the contract matches.
pub fn contract_problems(contract: &Value, labels: Option<usize>) -> Vec<String> {
    let mut problems = Vec::new();
    if contract.get("prompt_source_sha256").and_then(Value::as_str) != Some(PROMPT_SOURCE_SHA256) {
        problems.push("prompt_source_sha256 differs from the bundled renderer".to_string());
    }
    match (contract.get("single_token_labels"), labels) {
        (None | Some(Value::Null), _) | (_, None) => {}
        (Some(n), Some(have)) if n.as_u64() == Some(have as u64) => {}
        (Some(n), Some(have)) => problems.push(format!(
            "single_token_labels is {n}, this tokenizer has {have}"
        )),
    }
    if let Some(kwargs) = contract.get("chat_template_kwargs") {
        let want = serde_json::json!({
            "add_generation_prompt": true, "enable_thinking": false, "thinking": false
        });
        if kwargs != &want {
            problems.push("chat_template_kwargs differ (thinking must be off)".to_string());
        }
    }
    problems
}

/// `reorder_choice`: the criteria in `order`, a list read as keys with no
/// description. `order` must hold every key once.
fn reorder_choice(
    id: &str,
    criteria: Option<&Value>,
    order: &[String],
) -> Result<Value, DecisionError> {
    let crit: Map<String, Value> = match criteria {
        Some(Value::Object(map)) => map.clone(),
        Some(Value::Array(items)) => {
            let mut map = Map::new();
            for item in items {
                let key = py_str(item, &format!("question '{id}': a 'criteria' entry"))
                    .map_err(DecisionError::Request)?;
                map.insert(key, Value::Null);
            }
            map
        }
        _ => Map::new(),
    };
    let mut sorted_order: Vec<&String> = order.iter().collect();
    sorted_order.sort();
    let mut sorted_keys: Vec<&String> = crit.keys().collect();
    sorted_keys.sort();
    if sorted_order != sorted_keys {
        return Err(DecisionError::Request(format!(
            "question '{id}': order must be a permutation of the option keys"
        )));
    }
    let mut out = Map::new();
    for key in order {
        out.insert(key.clone(), crit.get(key).cloned().unwrap_or(Value::Null));
    }
    Ok(Value::Object(out))
}

/// `Renderer.render_messages`' swap of AINode's labels for `letters` in
/// the option lines of `content`. `None` where the reference's assertion
/// fails.
fn swap_labels(content: &str, letters: &[String]) -> Option<String> {
    const SEP: &str = "\nOPTIONS:\n";
    let (head, rest) = content.split_once(SEP)?;
    let mut lines: Vec<String> = rest.split('\n').map(str::to_string).collect();
    for (i, letter) in letters.iter().enumerate() {
        let prefix = format!("{}. ", option_label(i));
        let line = lines.get_mut(i)?;
        let tail = line.strip_prefix(&prefix)?.to_string();
        *line = format!("{letter}. {tail}");
    }
    Some(format!("{head}{SEP}{}", lines.join("\n")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `serialize_state` is `json.dumps(state, separators=(",", ":"),
    /// sort_keys=True, ensure_ascii=False)`.
    #[test]
    fn state_is_compact_sorted_json() {
        let state = json!({"ticket": {"subject": "Invoice", "from": "dana", "n": 1.0}, "a": "é"});
        assert_eq!(
            serialize_state(Some(&state)),
            r#"{"a":"é","ticket":{"from":"dana","n":1.0,"subject":"Invoice"}}"#
        );
        assert_eq!(serialize_state(None), "");
        assert_eq!(serialize_state(Some(&Value::Null)), "");
        assert_eq!(serialize_state(Some(&json!("as is "))), "as is ");
    }

    #[test]
    fn label_swap_matches_the_reference() {
        let content = "STATE:\nx\n\nQUESTION: q\n\nOPTIONS:\nA. one\nB. two\n\nAnswer.";
        let letters = ["AA".to_string(), "AB".to_string()];
        assert_eq!(
            swap_labels(content, &letters).as_deref(),
            Some("STATE:\nx\n\nQUESTION: q\n\nOPTIONS:\nAA. one\nAB. two\n\nAnswer.")
        );
        let broken = "STATE:\n\nOPTIONS:\nnot a label\n\nQUESTION: q\n\nOPTIONS:\nA. one";
        assert_eq!(swap_labels(broken, &letters), None);
    }

    #[test]
    fn contract_problems_name_each_mismatch() {
        let good = json!({"prompt_source_sha256": PROMPT_SOURCE_SHA256, "single_token_labels": 68,
            "chat_template_kwargs": {"add_generation_prompt": true, "enable_thinking": false,
            "thinking": false}});
        assert!(contract_problems(&good, Some(68)).is_empty());
        assert!(contract_problems(&good, None).is_empty());
        let bad = json!({"prompt_source_sha256": "x", "single_token_labels": 26,
            "chat_template_kwargs": {"enable_thinking": true}});
        assert_eq!(contract_problems(&bad, None).len(), 2);
        assert_eq!(contract_problems(&bad, Some(68)).len(), 3);
    }

    #[test]
    fn order_must_be_a_permutation() {
        let crit = json!({"a": "x", "b": null});
        let order = ["b".to_string(), "a".to_string()];
        assert_eq!(
            reorder_choice("q", Some(&crit), &order).expect("reorders"),
            json!({"b": null, "a": "x"})
        );
        let short = ["b".to_string()];
        assert!(reorder_choice("q", Some(&crit), &short).is_err());
        let listed = json!(["a", 2]);
        let order = ["2".to_string(), "a".to_string()];
        assert_eq!(
            reorder_choice("q", Some(&listed), &order).expect("reorders"),
            json!({"2": null, "a": null})
        );
    }
}
