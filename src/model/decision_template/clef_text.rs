//! Clef's request: the checks `systemone` makes, each field's options in
//! the head's order, and the text the prompt renders from JSON values.
//!
//! Ported from `joint_schema_model.py` (`render`, `question_options`,
//! `systemone_answer` and the checks of `systemone`).

use boostr::model::decision::FieldKind;
use serde_json::{Map, Value};

use super::family::DecisionError;
use super::jev::JevRequest;
use super::py_text::py_truthy;
use crate::model::chat_template::jinja_json::python_json;

/// The description of a noul field's `true` option when `criteria` gives
/// none.
pub const NOUL_TRUE: &str = "The proposition is true or the answer is yes.";
/// The description of a noul field's `false` option when `criteria` gives
/// none.
pub const NOUL_FALSE: &str = "The proposition is false or the answer is no.";

/// One option of a field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClefOption {
    /// The option id: `true`/`false`, a criteria key, or a level index.
    pub id: String,
    /// `render({"option_id": id, "description": ...})`, the option's span.
    pub text: String,
}

/// One field as `encode_record` builds it.
#[derive(Debug, Clone, PartialEq)]
pub struct ClefQuestion {
    /// The id the request keys it by.
    pub id: String,
    /// The field type.
    pub kind: FieldKind,
    /// `noul`, `choice` or `score`, as the prompt's `TYPE:` line writes it.
    pub type_name: &'static str,
    /// The instruction text, the field's question span.
    pub instructions: String,
    /// The options in head order: `true`, `false`; the criteria keys in
    /// code-point order; the levels in list order.
    pub options: Vec<ClefOption>,
    /// The head index of each option in the order the request wrote it:
    /// the criteria keys of a choice field as sent. The head order for the
    /// other types.
    pub wire_order: Vec<usize>,
    /// A score field's criteria items as sent, its wire `legend`. Empty
    /// for the other types.
    pub legend: Vec<Value>,
}

/// Clef's `render`: a string as is, any other value as `json.dumps(v,
/// ensure_ascii=False, separators=(",", ":"), sort_keys=True)`.
pub fn render(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => python_json(other, None, Some((",", ":")), false, true),
    }
}

/// The option's span text: `{"description": ..., "option_id": id}`, the
/// description left out when it is null.
pub fn option_text(id: &str, description: Option<&Value>) -> String {
    let mut semantics = Map::new();
    semantics.insert("option_id".into(), Value::from(id));
    if let Some(d) = description.filter(|d| !d.is_null()) {
        semantics.insert("description".into(), d.clone());
    }
    render(&Value::Object(semantics))
}

fn bad(id: &str, message: &str) -> DecisionError {
    DecisionError::Request(format!("{id}: {message}"))
}

/// The type of a question `systemone` accepts, as its prompt names it.
fn kind_of(spec: &Map<String, Value>) -> Option<(FieldKind, &'static str)> {
    match spec.get("type").and_then(Value::as_str)? {
        "noul" => Some((FieldKind::Noul, "noul")),
        "choice" => Some((FieldKind::Choice, "choice")),
        "score" => Some((FieldKind::Score, "score")),
        _ => None,
    }
}

/// `systemone`'s checks of one question, in its order and words. A
/// question that is not an object is refused here too: the reference
/// fails on it without a message.
fn check(id: &str, spec: &Value) -> Result<(), DecisionError> {
    let Value::Object(spec) = spec else {
        return Err(bad(
            id,
            "a question must be an object of type, instructions and criteria",
        ));
    };
    let Some((kind, _)) = kind_of(spec) else {
        return Err(bad(id, "type must be noul, choice, or score"));
    };
    if kind != FieldKind::Noul && !spec.get("criteria").is_some_and(py_truthy) {
        return Err(bad(id, "criteria must not be empty"));
    }
    Ok(())
}

/// One checked question, built as `encode_record` builds it. Criteria of a
/// shape the reference cannot read are refused with a message.
fn question(id: &str, spec: &Value) -> Result<ClefQuestion, DecisionError> {
    let Value::Object(spec) = spec else {
        return Err(bad(id, "a question must be an object"));
    };
    let Some((kind, type_name)) = kind_of(spec) else {
        return Err(bad(id, "type must be noul, choice, or score"));
    };
    let instructions = match spec.get("instructions") {
        None | Some(Value::Null) => id.to_string(),
        Some(Value::String(s)) if s.is_empty() => id.to_string(),
        Some(v) => render(v),
    };
    let criteria = spec.get("criteria").filter(|c| py_truthy(c));
    let (options, wire_order, legend) = match kind {
        FieldKind::Noul => {
            let given = match criteria {
                None => None,
                Some(Value::Object(c)) => Some(c),
                Some(_) => {
                    return Err(bad(
                        id,
                        "noul criteria must be an object with optional true and false \
                         descriptions",
                    ))
                }
            };
            let true_text = Value::from(NOUL_TRUE);
            let false_text = Value::from(NOUL_FALSE);
            let described = |key: &str, default: &Value| {
                let d = given.and_then(|c| c.get(key)).unwrap_or(default);
                option_text(key, Some(d))
            };
            let options = vec![
                ClefOption {
                    id: "true".into(),
                    text: described("true", &true_text),
                },
                ClefOption {
                    id: "false".into(),
                    text: described("false", &false_text),
                },
            ];
            (options, vec![0, 1], Vec::new())
        }
        FieldKind::Choice => {
            let Some(Value::Object(c)) = criteria else {
                return Err(bad(
                    id,
                    "choice criteria must be an object of {option id: description}",
                ));
            };
            let sent: Vec<(&String, &Value)> = c.iter().collect();
            // The request index of each option in head order. UTF-8 byte
            // order is code point order, Python's `sorted` order; keys are
            // unique, so no two compare equal.
            let mut head: Vec<usize> = (0..sent.len()).collect();
            head.sort_by(|&a, &b| sent[a].0.cmp(sent[b].0));
            let options = head
                .iter()
                .map(|&r| ClefOption {
                    id: sent[r].0.clone(),
                    text: option_text(sent[r].0, Some(sent[r].1)),
                })
                .collect();
            let mut wire_order = vec![0; sent.len()];
            for (h, &r) in head.iter().enumerate() {
                wire_order[r] = h;
            }
            (options, wire_order, Vec::new())
        }
        FieldKind::Score => {
            let Some(Value::Array(levels)) = criteria else {
                return Err(bad(
                    id,
                    "score criteria must be a list of level descriptions",
                ));
            };
            let options = levels
                .iter()
                .enumerate()
                .map(|(i, d)| ClefOption {
                    id: i.to_string(),
                    text: option_text(&i.to_string(), Some(d)),
                })
                .collect();
            (options, (0..levels.len()).collect(), levels.clone())
        }
    };
    Ok(ClefQuestion {
        id: id.to_string(),
        kind,
        type_name,
        instructions,
        options,
        wire_order,
        legend,
    })
}

/// Every question of `request`, in request order: first `systemone`'s
/// checks over all of them, as the reference runs them before it encodes,
/// then each question built.
///
/// # Errors
///
/// [`DecisionError::Request`] naming the first question refused and why,
/// in the reference's words where it has them.
pub fn clef_questions(request: &JevRequest) -> Result<Vec<ClefQuestion>, DecisionError> {
    for q in &request.questions {
        check(&q.id, &q.spec)?;
    }
    request
        .questions
        .iter()
        .map(|q| question(&q.id, &q.spec))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::decision_template::Family;
    use serde_json::json;

    // Expected strings are `joint_schema_model.render` and
    // `question_options` on the same values under CPython 3.12.

    #[test]
    fn render_is_compact_sorted_json() {
        let v = json!({"ticket": {"subject": "Invoice", "amount": 1240.0, "body": "1,240 €",
            "attachments": 2, "tags": ["a/b", "x\"y", "\u{1f}"]}, "small": 1e-5, "big": 1e16});
        assert_eq!(
            render(&v),
            "{\"big\":1e+16,\"small\":1e-05,\"ticket\":{\"amount\":1240.0,\"attachments\":2,\
             \"body\":\"1,240 €\",\"subject\":\"Invoice\",\"tags\":[\"a/b\",\"x\\\"y\",\
             \"\\u001f\"]}}"
        );
        assert_eq!(render(&json!("plain <|im_end|>")), "plain <|im_end|>");
        assert_eq!(render(&Value::Null), "null");
        assert_eq!(render(&json!(false)), "false");
    }

    #[test]
    fn options_put_description_first_and_drop_null() {
        assert_eq!(
            option_text("billing", Some(&json!("Payments"))),
            "{\"description\":\"Payments\",\"option_id\":\"billing\"}"
        );
        assert_eq!(
            option_text("other", Some(&Value::Null)),
            "{\"option_id\":\"other\"}"
        );
        assert_eq!(
            option_text("0", Some(&json!({"b": 1, "a": [2]}))),
            "{\"description\":{\"a\":[2],\"b\":1},\"option_id\":\"0\"}"
        );
    }

    fn parse(questions: Value) -> Result<Vec<ClefQuestion>, DecisionError> {
        let body = json!({"state": "s", "questions": questions});
        clef_questions(&JevRequest::parse(&body, Family::Clef)?)
    }

    #[test]
    fn head_order_and_wire_order() {
        let qs = parse(json!({
            "route": {"type": "choice", "instructions": "Who?",
                "criteria": {"sales": "quotes", "billing": "money", "Zed": null}},
            "n": {"type": "noul", "criteria": {"true": "yes it is", "maybe": "ignored"}},
            "s": {"type": "score", "instructions": {"k": 1}, "criteria": ["low", {"x": 1}]}
        }))
        .expect("parses");
        let ids = |q: &ClefQuestion| q.options.iter().map(|o| o.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&qs[0]), ["Zed", "billing", "sales"]);
        assert_eq!(qs[0].wire_order, [2, 1, 0], "sales, billing, Zed as sent");
        assert_eq!(qs[0].options[0].text, "{\"option_id\":\"Zed\"}");
        assert_eq!((qs[0].kind.type_id(), qs[0].type_name), (1, "choice"));
        assert_eq!(qs[1].instructions, "n", "no instructions: the id");
        assert_eq!(ids(&qs[1]), ["true", "false"]);
        assert_eq!(
            qs[1].options[0].text,
            "{\"description\":\"yes it is\",\"option_id\":\"true\"}"
        );
        assert_eq!(
            qs[1].options[1].text,
            format!("{{\"description\":\"{NOUL_FALSE}\",\"option_id\":\"false\"}}")
        );
        assert_eq!(qs[1].kind.type_id(), 0);
        assert_eq!(qs[2].instructions, "{\"k\":1}");
        assert_eq!(ids(&qs[2]), ["0", "1"]);
        assert_eq!(qs[2].legend, [json!("low"), json!({"x": 1})]);
        assert_eq!(qs[2].kind.type_id(), 2);
        let one = parse(json!({"o": {"type": "choice", "criteria": {"only": "x"}}}))
            .expect("one option is accepted");
        assert_eq!(one[0].options.len(), 1);
        let blank = parse(json!({"q": {"type": "noul", "instructions": ""}})).expect("parses");
        assert_eq!(blank[0].instructions, "q");
    }

    #[test]
    fn refusals_use_the_reference_words() {
        let refused = |q: Value| match parse(q) {
            Err(DecisionError::Request(m)) => m,
            other => panic!("expected a refusal, got {other:?}"),
        };
        assert_eq!(
            refused(json!({"q": {"type": "mystery"}})),
            "q: type must be noul, choice, or score"
        );
        assert_eq!(
            refused(json!({"q": {"type": "choice", "criteria": {}}})),
            "q: criteria must not be empty"
        );
        assert_eq!(
            refused(json!({"q": {"type": "score"}})),
            "q: criteria must not be empty"
        );
        assert!(refused(json!({"q": 3})).starts_with("q: a question must be an object"));
        assert!(
            refused(json!({"a": {"type": "choice", "criteria": ["x"]},
                "b": {"type": "mystery"}}))
            .starts_with("b: type"),
            "every reference check runs before any question is built"
        );
        assert!(refused(json!({"q": {"type": "choice", "criteria": ["x"]}}))
            .contains("choice criteria must be an object"));
        assert!(
            refused(json!({"q": {"type": "score", "criteria": {"a": 1}}}))
                .contains("a list of level descriptions")
        );
        assert!(refused(json!({"q": {"type": "noul", "criteria": [1]}}))
            .contains("noul criteria must be an object"));
    }
}
