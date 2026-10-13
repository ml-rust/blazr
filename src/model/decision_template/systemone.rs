//! The decider's Jev questions, ported from `decider/systemone.py`
//! (`render_state`, `annotate_indices`, `render_question`, `plan_rows`,
//! `isolated_rows`, `strip_level_number`).
//!
//! Error messages are the decider's, word for word. It names no question
//! id in them, and neither does this port.

use serde_json::{Map, Value};

use super::encode::PromptText;
use super::py_text::{is_py_space, nd_digit, py_float, py_repr, py_repr_str, py_str, py_truthy};
use crate::model::chat_template::jinja_json::python_json;

/// The widest choice question.
pub const MAX_CHOICE: usize = 255;
/// The most levels of a score question.
pub const MAX_LEVELS: usize = 10;
/// Arrays this long or longer get each element's position written in.
pub const ANNOTATE_MIN: usize = 8;
/// The question text of a noul question sent without instructions.
pub const NOUL_WITHOUT_INSTRUCTIONS: &str = "Which answer fits the context?";

/// A decider answer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S1Kind {
    /// One of the criteria keys.
    Choice,
    /// No or yes.
    Noul,
    /// One level of an ordered scale.
    Score,
}

/// `render_question`'s output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S1Question {
    /// The question line.
    pub question: PromptText,
    /// One option line each.
    pub options: Vec<PromptText>,
    /// The answer type.
    pub kind: S1Kind,
    /// What each option answers to: criteria keys, `false`/`true`, or level
    /// indices.
    pub names: Vec<String>,
    /// A score question's level descriptions, in level order.
    pub legend: Vec<String>,
    /// The question's own `isolated` flag (default true).
    pub isolated: bool,
}

/// One scoring row of [`plan_rows`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanRow {
    /// The question line of the row.
    pub question: PromptText,
    /// Its option lines.
    pub options: Vec<PromptText>,
}

/// Where a question's scoring rows sit in the [`plan_rows`] order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanEntry {
    /// Index of the question.
    pub question: usize,
    /// One `[no, yes]` row per level.
    pub isolated: bool,
    /// The first row.
    pub first: usize,
    /// The number of rows.
    pub count: usize,
}

/// `_txt`: a string as is, any other value as `json.dumps(v,
/// ensure_ascii=False)`.
fn txt(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => python_json(other, None, None, false, false),
    }
}

/// `v in (None, "")`.
fn blank(value: Option<&Value>) -> bool {
    matches!(value, None | Some(Value::Null)) || value.and_then(Value::as_str) == Some("")
}

/// `spec.get(field)`, with null read as Python's `None`.
fn get<'a>(spec: &'a Map<String, Value>, field: &str) -> Option<&'a Value> {
    spec.get(field).filter(|v| !v.is_null())
}

/// `render_state`: a string as is, anything else as `json.dumps` of it
/// with long arrays annotated.
pub fn render_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => python_json(&annotate_indices(other), None, None, false, false),
    }
}

/// `annotate_indices`: each element of an array of [`ANNOTATE_MIN`] or more
/// elements carries its position, `{"_index": i, ...}` for an object and
/// `{"_index": i, "value": v}` for anything else.
pub fn annotate_indices(value: &Value) -> Value {
    match value {
        Value::Array(items) if items.len() >= ANNOTATE_MIN => Value::Array(
            items
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let mut out = Map::new();
                    out.insert("_index".into(), Value::from(i));
                    match annotate_indices(v) {
                        // `{"_index": i, **v}`: a key `_index` of `v` keeps
                        // the first position with its own value.
                        Value::Object(inner) if v.is_object() => out.extend(inner),
                        other => {
                            out.insert("value".into(), other);
                        }
                    }
                    Value::Object(out)
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(annotate_indices).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), annotate_indices(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `strip_level_number`: `"2: somewhat"` is `"somewhat"`. The prefix is
/// the regex `^\s*-?\d+\s*:\s*` with Python's Unicode `\s` and `\d`.
pub fn strip_level_number(text: &str) -> &str {
    let rest = text.trim_start_matches(is_py_space);
    let rest = rest.strip_prefix('-').unwrap_or(rest);
    let digits = rest.trim_start_matches(|c| nd_digit(c).is_some());
    if digits.len() == rest.len() {
        return text;
    }
    match digits.trim_start_matches(is_py_space).strip_prefix(':') {
        Some(after) => after.trim_start_matches(is_py_space),
        None => text,
    }
}

/// `render_question`.
///
/// # Errors
///
/// The decider's message for a question it refuses.
pub fn render_question(spec: &Value) -> Result<S1Question, String> {
    let Value::Object(spec) = spec else {
        return Err("a question must be an object of {type, instructions, criteria}".into());
    };
    let kind_value = spec.get("type").cloned().unwrap_or_else(|| "choice".into());
    let crit = match spec.get("criteria") {
        Some(v) => Some(v).filter(|v| !v.is_null()),
        None => get(spec, "options"),
    };
    // `spec.get("instructions", spec.get("question", ""))`; `None` is the `""`.
    let raw = spec.get("instructions").or_else(|| spec.get("question"));
    let is = |name: &str| kind_value.as_str() == Some(name);
    let noul_like = is("noul") || is("bool");
    let question = if noul_like && blank(raw) {
        let described = match crit {
            Some(Value::Object(c)) => [get(c, "true"), get(c, "false")]
                .into_iter()
                .any(|d| !blank(d)),
            _ => false,
        };
        if !described && matches!(crit, None | Some(Value::Object(_))) {
            return Err(
                "noul question without instructions: criteria must describe true or false".into(),
            );
        }
        PromptText::template(NOUL_WITHOUT_INSTRUCTIONS)
    } else {
        let text = match raw {
            None => String::new(),
            Some(v) => txt(v),
        };
        PromptText::caller(text)
    };
    if question.text().is_empty() {
        return Err("question without instructions".into());
    }
    let (kind, options, names, legend) = if is("choice") {
        let (options, names) = choice_options(crit)?;
        (S1Kind::Choice, options, names, Vec::new())
    } else if is("score") {
        let levels = score_levels(crit)?;
        let options = levels
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut o = PromptText::template(format!("{i}: "));
                o.push_caller(txt(c));
                o
            })
            .collect();
        let names = (0..levels.len()).map(|i| i.to_string()).collect();
        (
            S1Kind::Score,
            options,
            names,
            levels.iter().map(txt).collect(),
        )
    } else if noul_like {
        let c = match crit {
            None => None,
            Some(Value::Object(c)) => Some(c),
            Some(_) => {
                return Err("noul criteria: a map of optional true/false descriptions".into())
            }
        };
        let side = |label: &str, key: &str| {
            let d = c.and_then(|c| get(c, key));
            let mut o = PromptText::template(label);
            if !blank(d) {
                o.push_template(": ")
                    .push_caller(d.map(txt).unwrap_or_default());
            }
            o
        };
        let options = vec![side("no", "false"), side("yes", "true")];
        let names = vec!["false".to_string(), "true".to_string()];
        (S1Kind::Noul, options, names, Vec::new())
    } else {
        return Err(format!("unknown question type {}", py_repr(&kind_value)));
    };
    Ok(S1Question {
        question,
        options,
        kind,
        names,
        legend,
        isolated: spec.get("isolated").is_none_or(py_truthy),
    })
}

fn choice_options(crit: Option<&Value>) -> Result<(Vec<PromptText>, Vec<String>), String> {
    let refused = || format!("choice criteria: a map of 2..{MAX_CHOICE} options");
    let map: Map<String, Value> = match crit {
        Some(Value::Array(items)) => {
            let mut map = Map::new();
            for item in items {
                map.insert(py_str(item, "a choice criteria entry")?, Value::Null);
            }
            map
        }
        Some(Value::Object(map)) => map.clone(),
        _ => return Err(refused()),
    };
    if !(2..=MAX_CHOICE).contains(&map.len()) {
        return Err(refused());
    }
    let options = map
        .iter()
        .map(|(name, d)| {
            let mut o = PromptText::caller(name);
            if !blank(Some(d)) {
                o.push_template(": ").push_caller(txt(d));
            }
            o
        })
        .collect();
    Ok((options, map.keys().cloned().collect()))
}

/// A score question's levels: a list as is, a `{level number: description}`
/// map in the order of its keys read as floats.
fn score_levels(crit: Option<&Value>) -> Result<Vec<Value>, String> {
    let levels = match crit {
        Some(Value::Object(map)) => {
            let mut keyed = Vec::with_capacity(map.len());
            for (k, v) in map {
                let f = py_float(k)?;
                if f.is_nan() {
                    return Err(format!(
                        "score criteria: the level key {} reads as NaN, which has no order",
                        py_repr_str(k)
                    ));
                }
                keyed.push((f, v.clone()));
            }
            // Stable, and -0.0 equals 0.0, as in Python's `sorted`.
            keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            Some(keyed.into_iter().map(|(_, v)| v).collect::<Vec<_>>())
        }
        Some(Value::Array(items)) => Some(items.clone()),
        _ => None,
    };
    match levels {
        Some(levels) if (2..=MAX_LEVELS).contains(&levels.len()) => Ok(levels),
        _ => Err(format!(
            "score criteria: an ordered list of 2..{MAX_LEVELS} level descriptions"
        )),
    }
}

/// `plan_rows`: one scoring row per question, or one `[no, yes]` row per
/// level of a Score question when `isolated` (and the question's own flag)
/// says so.
pub fn plan_rows(questions: &[S1Question], isolated: bool) -> (Vec<PlanRow>, Vec<PlanEntry>) {
    let mut rows = Vec::new();
    let mut index = Vec::with_capacity(questions.len());
    for (q, rq) in questions.iter().enumerate() {
        if isolated && rq.kind == S1Kind::Score && rq.isolated {
            index.push(PlanEntry {
                question: q,
                isolated: true,
                first: rows.len(),
                count: rq.legend.len(),
            });
            for level in &rq.legend {
                let mut question = rq.question.clone();
                question
                    .push_template("\nProposed answer: ")
                    .push_caller(strip_level_number(level))
                    .push_template("\nDoes the proposed answer fit?");
                rows.push(PlanRow {
                    question,
                    options: vec![PromptText::template("no"), PromptText::template("yes")],
                });
            }
        } else {
            index.push(PlanEntry {
                question: q,
                isolated: false,
                first: rows.len(),
                count: 1,
            });
            rows.push(PlanRow {
                question: rq.question.clone(),
                options: rq.options.clone(),
            });
        }
    }
    (rows, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn texts(options: &[PromptText]) -> Vec<String> {
        options.iter().map(PromptText::text).collect()
    }

    // Expected strings are CPython 3.12's output of the decider's own code.

    #[test]
    fn strip_level_number_matches_re_sub() {
        for (text, want) in [
            ("2: somewhat", "somewhat"),
            (" -12 : x", "x"),
            ("\u{ff11}\u{ff12}: y", "y"),
            ("3:", ""),
            ("a: b", "a: b"),
            ("4 :  ", ""),
            ("-: x", "-: x"),
        ] {
            assert_eq!(strip_level_number(text), want, "{text:?}");
        }
    }

    #[test]
    fn state_json_is_python_default_with_annotated_arrays() {
        let state = json!({"b": 1.0, "a": [1, 2], "long": [1, {"x": 1}, {"_index": "own"}, 4, 5, 6, 7, "é"]});
        assert_eq!(
            render_state(&state),
            "{\"b\": 1.0, \"a\": [1, 2], \"long\": [{\"_index\": 0, \"value\": 1}, \
             {\"_index\": 1, \"x\": 1}, {\"_index\": \"own\"}, {\"_index\": 3, \"value\": 4}, \
             {\"_index\": 4, \"value\": 5}, {\"_index\": 5, \"value\": 6}, \
             {\"_index\": 6, \"value\": 7}, {\"_index\": 7, \"value\": \"é\"}]}"
        );
        assert_eq!(render_state(&json!("plain")), "plain");
        assert_eq!(render_state(&Value::Null), "null");
    }

    #[test]
    fn choice_and_noul_options() {
        let q = render_question(&json!({"instructions": {"a": 1},
            "criteria": {"billing": "x", "support": null, "sales": "", "other": [1, 2]}}))
        .expect("renders");
        assert_eq!(q.question.text(), "{\"a\": 1}");
        assert_eq!(
            texts(&q.options),
            ["billing: x", "support", "sales", "other: [1, 2]"]
        );
        let q = render_question(&json!({"type": "noul", "instructions": "Blocked?",
            "criteria": {"true": "stopped", "false": "it can wait"}}))
        .expect("renders");
        assert_eq!(texts(&q.options), ["no: it can wait", "yes: stopped"]);
        let q =
            render_question(&json!({"type": "bool", "criteria": {"true": "x"}})).expect("renders");
        assert_eq!(q.question.text(), NOUL_WITHOUT_INSTRUCTIONS);
        assert_eq!(q.kind, S1Kind::Noul);
    }

    #[test]
    fn score_legend_map_sorts_by_float() {
        let q = render_question(&json!({"type": "score", "instructions": "How?",
            "criteria": {"10": "ten", "2": "two", "-1": "minus"}}))
        .expect("renders");
        assert_eq!(texts(&q.options), ["0: minus", "1: two", "2: ten"]);
        assert_eq!(q.legend, ["minus", "two", "ten"]);
        assert_eq!(
            render_question(
                &json!({"type": "score", "instructions": "x", "criteria": {"a": 1, "b": 2}})
            ),
            Err("could not convert string to float: 'a'".into())
        );
    }

    #[test]
    fn decider_errors() {
        for (spec, msg) in [
            (
                json!({"type": "choice", "instructions": "x", "criteria": {"a": 1}}),
                "choice criteria: a map of 2..255 options",
            ),
            (
                json!({"type": "score", "instructions": "x", "criteria": ["a"]}),
                "score criteria: an ordered list of 2..10 level descriptions",
            ),
            (
                json!({"type": "noul", "instructions": "x", "criteria": [1]}),
                "noul criteria: a map of optional true/false descriptions",
            ),
            (
                json!({"type": "noul", "criteria": {"true": ""}}),
                "noul question without instructions: criteria must describe true or false",
            ),
            (
                json!({"type": "choice", "instructions": ""}),
                "question without instructions",
            ),
            (
                json!({"type": "maybe", "instructions": "x"}),
                "unknown question type 'maybe'",
            ),
            (
                json!({"type": null, "instructions": "x"}),
                "unknown question type None",
            ),
        ] {
            assert_eq!(render_question(&spec), Err(msg.to_string()), "{spec}");
        }
    }

    #[test]
    fn isolated_rows_strip_level_numbers() {
        let q = render_question(&json!({"type": "score", "instructions": "How urgent?",
            "criteria": ["0: none", "high"]}))
        .expect("renders");
        let (rows, index) = plan_rows(std::slice::from_ref(&q), true);
        assert_eq!(
            index,
            [PlanEntry {
                question: 0,
                isolated: true,
                first: 0,
                count: 2
            }]
        );
        assert_eq!(
            rows[0].question.text(),
            "How urgent?\nProposed answer: none\nDoes the proposed answer fit?"
        );
        let (rows, index) = plan_rows(&[q], false);
        assert_eq!((rows.len(), index[0].isolated), (1, false));
    }
}
