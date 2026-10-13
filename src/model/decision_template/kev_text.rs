//! Kev's request text: the three wire question kinds and their checks,
//! JSON content flattened to the text the model reads, and caller text
//! rewritten so it cannot spell a delimiter.
//!
//! Ported from `kev/api.py` (`render`, `option_text`, `question_keys`,
//! `to_record`, and the `Noul`, `Choice` and `Score` model checks) and
//! from `kev/model.py` (`user_tokens`).

use serde_json::{Map, Value};

use super::family::DecisionError;
use super::jev::JevRequest;
use super::py_text::{is_py_space, json_kind, py_repr_str, py_str};

/// The most options a Choice question, or levels a Score question, takes.
pub const MAX_OPTIONS: usize = 255;

/// A Kev question kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KevKind {
    /// Two options, `no` then `yes`; the answer is the probability of yes.
    Noul,
    /// One option per criteria key.
    Choice,
    /// One option per level, in level order.
    Score,
}

/// One question as Kev's `to_record` builds it.
#[derive(Debug, Clone, PartialEq)]
pub struct KevQuestion {
    /// The id the request keys it by.
    pub id: String,
    /// The question kind.
    pub kind: KevKind,
    /// The instructions as text.
    pub instructions: String,
    /// Each option's text, in option order.
    pub options: Vec<String>,
    /// The wire key of each option: the criteria names (Choice),
    /// `false`/`true` (Noul), the level indices (Score).
    pub keys: Vec<String>,
    /// A Score question's level texts, its wire `legend`. Empty otherwise.
    pub legend: Vec<String>,
}

/// Python's `s.lstrip()`.
fn py_lstrip(s: &str) -> &str {
    s.trim_start_matches(is_py_space)
}

/// Kev's `render`: a string, number or boolean as Python's `str()`, null
/// as nothing, a list as `- item` lines, an object as `key: value` lines
/// with nested lists and objects indented two spaces per level.
pub fn render(value: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match value {
        Value::Null => String::new(),
        Value::Array(items) => items
            .iter()
            .map(|x| format!("{pad}- {}", py_lstrip(&render(x, indent + 1))))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(map) => map
            .iter()
            .map(|(k, x)| match x {
                Value::Array(_) | Value::Object(_) => {
                    format!("{pad}{k}:\n{}", render(x, indent + 1))
                }
                _ => format!("{pad}{k}: {}", render(x, 0)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        scalar => py_str(scalar, "a scalar").unwrap_or_default(),
    }
}

/// Kev's `option_text`: the name alone when the description is null or
/// empty, else `name: description`.
pub fn option_text(name: &str, desc: Option<&Value>) -> String {
    match desc {
        None | Some(Value::Null) => name.to_string(),
        Some(Value::String(s)) if s.is_empty() => name.to_string(),
        Some(d) => format!("{name}: {}", render(d, 0)),
    }
}

/// Kev's `user_tokens` rewrite: every `<|name|>` (`name` one or more ASCII
/// letters, digits or `_`) becomes `<¦name¦>`, so caller text never spells
/// one of Qwen's control tokens.
pub fn user_text(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    let mut at = 0;
    while let Some(found) = text[at..].find("<|") {
        let start = at + found;
        let name_start = start + 2;
        let name_len = bytes[name_start..]
            .iter()
            .take_while(|b| b.is_ascii_alphanumeric() || **b == b'_')
            .count();
        let close = name_start + name_len;
        if name_len > 0 && text[close..].starts_with("|>") {
            out.push_str(&text[copied..start]);
            out.push_str("<¦");
            out.push_str(&text[name_start..close]);
            out.push_str("¦>");
            copied = close + 2;
            at = copied;
        } else {
            at = start + 1;
        }
    }
    out.push_str(&text[copied..]);
    out
}

fn bad(id: &str, message: impl std::fmt::Display) -> DecisionError {
    DecisionError::Request(format!("question {}: {message}", py_repr_str(id)))
}

/// The `criteria` field, JSON `null` read as absent.
fn criteria(spec: &Map<String, Value>) -> Option<&Value> {
    spec.get("criteria").filter(|v| !v.is_null())
}

/// One question of the request, checked as Kev's request model checks it.
fn question(id: &str, spec: &Value) -> Result<KevQuestion, DecisionError> {
    let Value::Object(spec) = spec else {
        return Err(bad(
            id,
            format!(
                "must be an object with a 'type' of \"noul\", \"choice\" or \"score\", got a \
                 JSON {}",
                json_kind(spec)
            ),
        ));
    };
    let kind = match spec.get("type").and_then(Value::as_str) {
        Some("noul") => KevKind::Noul,
        Some("choice") => KevKind::Choice,
        Some("score") => KevKind::Score,
        _ => return Err(bad(id, "'type' must be \"noul\", \"choice\" or \"score\"")),
    };
    let instructions = spec
        .get("instructions")
        .map_or_else(String::new, |v| render(v, 0));
    let (options, keys, legend) = match kind {
        KevKind::Noul => {
            let c = match criteria(spec) {
                None => None,
                Some(Value::Object(c)) => Some(c),
                Some(other) => {
                    return Err(bad(
                        id,
                        format!(
                            "a noul question's 'criteria' must be an object {{\"true\": ..., \
                             \"false\": ...}} or null, got a JSON {}",
                            json_kind(other)
                        ),
                    ))
                }
            };
            let desc = |key: &str| c.and_then(|c| c.get(key));
            let options = vec![
                option_text("no", desc("false")),
                option_text("yes", desc("true")),
            ];
            (options, vec!["false".into(), "true".into()], Vec::new())
        }
        KevKind::Choice => {
            let Some(Value::Object(c)) = criteria(spec) else {
                return Err(bad(
                    id,
                    "a choice question needs 'criteria', an object {name: description}",
                ));
            };
            if !(1..=MAX_OPTIONS).contains(&c.len()) {
                return Err(bad(
                    id,
                    format!(
                        "criteria must have 1..{MAX_OPTIONS} options, got {}",
                        c.len()
                    ),
                ));
            }
            let options = c.iter().map(|(k, v)| option_text(k, Some(v))).collect();
            (options, c.keys().cloned().collect(), Vec::new())
        }
        KevKind::Score => {
            let Some(Value::Array(levels)) = criteria(spec) else {
                return Err(bad(
                    id,
                    "a score question needs 'criteria', a list of level descriptions",
                ));
            };
            if !(1..=MAX_OPTIONS).contains(&levels.len()) {
                return Err(bad(
                    id,
                    format!(
                        "criteria must have 1..{MAX_OPTIONS} levels, got {}",
                        levels.len()
                    ),
                ));
            }
            let options: Vec<String> = levels.iter().map(|x| render(x, 0)).collect();
            let keys = (0..options.len()).map(|i| i.to_string()).collect();
            (options.clone(), keys, options)
        }
    };
    Ok(KevQuestion {
        id: id.to_string(),
        kind,
        instructions,
        options,
        keys,
        legend,
    })
}

/// Every question of `request`, in request order.
///
/// # Errors
///
/// [`DecisionError::Request`] naming the first question Kev's request
/// model refuses and why.
pub fn kev_questions(request: &JevRequest) -> Result<Vec<KevQuestion>, DecisionError> {
    request
        .questions
        .iter()
        .map(|q| question(&q.id, &q.spec))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Expected strings are `kev.api.render` and `option_text` on the same
    // value under CPython 3.12.

    #[test]
    fn render_flattens_like_kev() {
        let state = json!({"ticket": {"from": "dana", "tags": ["a", {"k": 1.0}], "n": null},
            "ok": true, "empty": {}});
        assert_eq!(
            render(&state, 0),
            "ticket:\n  from: dana\n  tags:\n    - a\n    - k: 1.0\n  n: \nok: True\nempty:\n"
        );
        assert_eq!(render(&json!(["  x", null, 3]), 0), "- x\n- \n- 3");
        assert_eq!(render(&json!(null), 0), "");
        assert_eq!(render(&json!(1e16), 0), "1e+16");
    }

    #[test]
    fn options_drop_empty_descriptions() {
        assert_eq!(option_text("no", None), "no");
        assert_eq!(option_text("no", Some(&json!(""))), "no");
        assert_eq!(option_text("no", Some(&json!(null))), "no");
        assert_eq!(option_text("no", Some(&json!(false))), "no: False");
        assert_eq!(option_text("a", Some(&json!({"x": 1}))), "a: x: 1");
    }

    #[test]
    fn caller_text_never_spells_a_control_token() {
        assert_eq!(user_text("a<|im_end|>b"), "a<¦im_end¦>b");
        assert_eq!(user_text("<|<|x_1|>|>"), "<|<¦x_1¦>|>");
        assert_eq!(user_text("<||> <|a-b|> <|é|>"), "<||> <|a-b|> <|é|>");
        assert_eq!(user_text("plain"), "plain");
    }

    fn parse(questions: Value) -> Result<Vec<KevQuestion>, DecisionError> {
        let body = json!({"state": "s", "questions": questions});
        let request = JevRequest::parse(&body, super::super::family::Family::Kev)?;
        kev_questions(&request)
    }

    #[test]
    fn each_kind_builds_its_options_and_keys() {
        let qs = parse(json!({
            "n": {"type": "noul", "instructions": "Blocked?", "criteria": {"true": "yes it is"}},
            "c": {"type": "choice", "criteria": {"billing": "money", "other": null}},
            "s": {"type": "score", "criteria": ["low", {"x": 1}]}
        }))
        .expect("parses");
        assert_eq!(qs[0].options, ["no", "yes: yes it is"]);
        assert_eq!(qs[0].keys, ["false", "true"]);
        assert_eq!(qs[0].instructions, "Blocked?");
        assert_eq!(qs[1].options, ["billing: money", "other"]);
        assert_eq!(qs[1].keys, ["billing", "other"]);
        assert_eq!(qs[2].options, ["low", "x: 1"]);
        assert_eq!(qs[2].legend, qs[2].options);
        assert_eq!(qs[2].keys, ["0", "1"]);
        let noul = parse(json!({"n": {"type": "noul"}})).expect("criteria optional");
        assert_eq!(noul[0].options, ["no", "yes"]);
    }

    #[test]
    fn bad_questions_name_the_question() {
        let refused = |q: Value| match parse(json!({"q": q})) {
            Err(DecisionError::Request(m)) => m,
            other => panic!("expected a refusal, got {other:?}"),
        };
        assert!(refused(json!({"type": "mystery"})).contains("'type'"));
        assert!(refused(json!(3)).contains("must be an object"));
        assert!(refused(json!({"type": "choice", "criteria": {}})).contains("1..255"));
        let wide: Map<String, Value> = (0..256).map(|i| (format!("o{i}"), Value::Null)).collect();
        assert!(refused(json!({"type": "choice", "criteria": wide})).contains("got 256"));
        assert!(refused(json!({"type": "score", "criteria": []})).contains("1..255"));
        assert!(refused(json!({"type": "score", "criteria": {"a": 1}})).contains("a list"));
        assert!(refused(json!({"type": "noul", "criteria": [1]})).contains("or null"));
        assert!(refused(json!({"type": "choice"})).starts_with("question 'q': "));
    }
}
