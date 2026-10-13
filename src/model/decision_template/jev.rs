//! The Jev request every family takes:
//! `{state, questions: {id: {type, instructions, criteria}}, model?, calibration?}`
//! plus each family's own fields.
//!
//! Questions keep the order the caller wrote them; blazr's `serde_json` is
//! built with `preserve_order`, and a repeated key keeps its first position
//! with its last value, as Python's `json.loads` does. Each question stays a
//! JSON value: the families read its fields differently (the decider takes
//! `options` for `criteria`, `question` for `instructions`, any JSON value
//! as text, and `choice` as the default type; Kev flattens any JSON value
//! to text), so each family's renderer reads them.

use serde_json::{Map, Value};

use super::family::{DecisionError, Family};
use super::py_text::{py_repr, py_strip};

/// Whether the model's fitted temperatures apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Calibration {
    /// The fitted temperatures apply (no `calibration` field).
    #[default]
    Fitted,
    /// `"calibration": "raw"`: temperature 1.
    Raw,
}

/// One question off the wire.
#[derive(Debug, Clone, PartialEq)]
pub struct JevQuestion {
    /// The id the request keys it by.
    pub id: String,
    /// The question as sent.
    pub spec: Value,
}

/// A parsed request.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct JevRequest {
    /// The state; `None` when the field is absent.
    pub state: Option<Value>,
    /// The questions, in request order.
    pub questions: Vec<JevQuestion>,
    /// The model the caller asked for.
    pub model: Option<String>,
    /// Whether the fitted temperatures apply.
    pub calibration: Calibration,
    /// Decider: one row per question (`true`, the default) or every question
    /// packed into one row.
    pub independent: Option<bool>,
    /// Decider: the prompt layout; only `state_first` is rendered.
    pub layout: Option<String>,
    /// Decider: read each Score level in its own row; `None` takes
    /// `decider_config.json`'s `isolated_levels`.
    pub isolated: Option<bool>,
    /// Decider: rewrite "none of the above" options; `None` takes
    /// `decider_config.json`'s `neutralize_none`.
    pub neutralize_none: Option<bool>,
    /// Decider: cap on the tokens of the `Context:` block.
    pub max_state_tokens: Option<usize>,
    /// Jebadiah: cap on the tokens of each prompt row.
    pub max_tokens: Option<usize>,
    /// Jebadiah: the option order of a choice question, by question id.
    pub order: Vec<(String, Vec<String>)>,
}

fn bad(message: impl Into<String>) -> DecisionError {
    DecisionError::Request(message.into())
}

/// `field`, with JSON `null` read as absent.
fn present<'a>(body: &'a Map<String, Value>, field: &str) -> Option<&'a Value> {
    body.get(field).filter(|v| !v.is_null())
}

fn bool_field(body: &Map<String, Value>, field: &str) -> Result<Option<bool>, DecisionError> {
    match present(body, field) {
        None => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(bad(format!("'{field}' must be true or false when given"))),
    }
}

fn count_field(body: &Map<String, Value>, field: &str) -> Result<Option<usize>, DecisionError> {
    match present(body, field) {
        None => Ok(None),
        Some(v) => match v.as_u64().and_then(|n| usize::try_from(n).ok()) {
            Some(n) if n > 0 => Ok(Some(n)),
            _ => Err(bad(format!(
                "'{field}' must be a positive integer when given"
            ))),
        },
    }
}

impl JevRequest {
    /// Parse a request body for `family`.
    ///
    /// Fields the family does not read are ignored, as its reference does.
    ///
    /// # Errors
    ///
    /// [`DecisionError::Request`] naming the field, in the family's wording
    /// where its reference has one.
    pub fn parse(body: &Value, family: Family) -> Result<Self, DecisionError> {
        let Value::Object(body) = body else {
            return Err(bad("body must be a JSON object"));
        };
        let mut request = JevRequest {
            state: body.get("state").cloned(),
            ..Self::default()
        };
        request.model = match present(body, "model") {
            None => None,
            Some(Value::String(s)) if family != Family::Jebadiah || !py_strip(s).is_empty() => {
                Some(py_strip(s).to_string())
            }
            Some(_) => {
                return Err(bad(match family {
                    Family::Jebadiah => "'model' must be a non-empty string when given",
                    Family::Decider | Family::Kev => "'model' must be a string when given",
                }))
            }
        };
        request.questions = parse_questions(body.get("questions"), family)?;
        request.calibration = match body.get("calibration") {
            None | Some(Value::Null) => Calibration::Fitted,
            Some(Value::String(s)) if s == "raw" => Calibration::Raw,
            Some(other) => {
                return Err(bad(format!(
                    "'calibration' must be \"raw\" when given (got {})",
                    py_repr(other)
                )))
            }
        };
        match family {
            Family::Jebadiah => {
                request.max_tokens = count_field(body, "max_tokens")?;
                request.order = parse_order(present(body, "order"))?;
            }
            Family::Decider => {
                if request.state.is_none() {
                    return Err(bad("'state' is required"));
                }
                request.independent = bool_field(body, "independent")?;
                request.isolated = bool_field(body, "isolated")?;
                request.neutralize_none = bool_field(body, "neutralize_none")?;
                request.max_state_tokens = count_field(body, "max_state_tokens")?;
                request.layout = match present(body, "layout") {
                    None => None,
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(_) => return Err(bad("'layout' must be a string when given")),
                };
            }
            Family::Kev => {
                if !body.contains_key("state") {
                    return Err(bad("'state' is required (it may be null)"));
                }
            }
        }
        Ok(request)
    }

    /// The order the caller gave for question `id`.
    pub fn order_of(&self, id: &str) -> Option<&[String]> {
        self.order
            .iter()
            .find(|(q, _)| q == id)
            .map(|(_, keys)| keys.as_slice())
    }
}

fn parse_questions(raw: Option<&Value>, family: Family) -> Result<Vec<JevQuestion>, DecisionError> {
    let map = match (raw, family) {
        (Some(Value::Object(map)), Family::Jebadiah | Family::Kev) if !map.is_empty() => map,
        (_, Family::Jebadiah | Family::Kev) => {
            return Err(bad(
                "'questions' must be a non-empty object of {id: question}",
            ))
        }
        (Some(Value::Object(map)), Family::Decider) => map,
        (_, Family::Decider) => return Err(bad("'questions' must be an object of {id: question}")),
    };
    let mut questions = Vec::with_capacity(map.len());
    for (id, spec) in map {
        if family == Family::Jebadiah && py_strip(id).is_empty() {
            return Err(bad("every question id must be a non-empty string"));
        }
        questions.push(JevQuestion {
            id: id.clone(),
            spec: spec.clone(),
        });
    }
    Ok(questions)
}

/// `order`: `{question id: [criteria key, ...]}`.
fn parse_order(raw: Option<&Value>) -> Result<Vec<(String, Vec<String>)>, DecisionError> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    let shape = || bad("'order' must be an object of {question id: [criteria key, ...]}");
    let Value::Object(map) = raw else {
        return Err(shape());
    };
    let mut out = Vec::with_capacity(map.len());
    for (id, keys) in map {
        let Value::Array(keys) = keys else {
            return Err(shape());
        };
        let keys = keys
            .iter()
            .map(|k| k.as_str().map(str::to_string).ok_or_else(shape))
            .collect::<Result<Vec<_>, _>>()?;
        out.push((id.clone(), keys));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn questions_keep_request_order() {
        let body = json!({"state": "s", "questions": {"z": {}, "a": {}, "m": {}}});
        let req = JevRequest::parse(&body, Family::Decider).expect("parses");
        let ids: Vec<&str> = req.questions.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["z", "a", "m"]);
    }

    #[test]
    fn repeated_key_keeps_first_position_and_last_value() {
        let body: Value =
            serde_json::from_str(r#"{"state": 1, "questions": {"a": 1, "b": 2, "a": 3}}"#)
                .expect("json");
        let req = JevRequest::parse(&body, Family::Decider).expect("parses");
        assert_eq!(
            req.questions[0],
            JevQuestion {
                id: "a".into(),
                spec: json!(3)
            }
        );
        assert_eq!(req.questions[1].id, "b");
    }

    #[test]
    fn family_wording() {
        let err = |body: Value, family| JevRequest::parse(&body, family).expect_err("refused");
        assert_eq!(
            err(json!({"questions": {}}), Family::Jebadiah),
            bad("'questions' must be a non-empty object of {id: question}")
        );
        assert_eq!(
            err(json!({"questions": {" ": {}}}), Family::Jebadiah),
            bad("every question id must be a non-empty string")
        );
        assert_eq!(
            err(
                json!({"questions": {"a": {}}, "calibration": "cold"}),
                Family::Jebadiah
            ),
            bad("'calibration' must be \"raw\" when given (got 'cold')")
        );
        assert_eq!(
            err(json!({"questions": {}}), Family::Decider),
            bad("'state' is required")
        );
        assert_eq!(
            err(
                json!({"state": "s", "questions": {}, "independent": 1}),
                Family::Decider
            ),
            bad("'independent' must be true or false when given")
        );
    }

    #[test]
    fn kev_wording() {
        let err = |body: Value| JevRequest::parse(&body, Family::Kev).expect_err("refused");
        assert_eq!(
            err(json!({"state": "s", "questions": {}})),
            bad("'questions' must be a non-empty object of {id: question}")
        );
        assert_eq!(
            err(json!({"questions": {"q": {}}})),
            bad("'state' is required (it may be null)")
        );
        assert_eq!(
            err(json!({"state": 1, "questions": {"q": {}}, "model": 2})),
            bad("'model' must be a string when given")
        );
        let ok = JevRequest::parse(
            &json!({"state": null, "model": "", "questions": {" ": {}}, "max_tokens": 0}),
            Family::Kev,
        )
        .expect("Kev takes any string id and model, and ignores other fields");
        assert_eq!(ok.state, Some(Value::Null));
        assert_eq!(ok.model.as_deref(), Some(""));
        assert_eq!(ok.max_tokens, None);
    }

    #[test]
    fn family_fields() {
        let body = json!({"state": {"a": 1}, "questions": {"q": {}}, "max_tokens": 2048,
            "order": {"q": ["b", "a"]}, "independent": false, "isolated": null});
        let jeb = JevRequest::parse(&body, Family::Jebadiah).expect("parses");
        assert_eq!(jeb.max_tokens, Some(2048));
        assert_eq!(
            jeb.order_of("q"),
            Some(&["b".to_string(), "a".to_string()][..])
        );
        assert_eq!(jeb.independent, None, "Jebadiah ignores it");
        let dec = JevRequest::parse(&body, Family::Decider).expect("parses");
        assert_eq!(dec.independent, Some(false));
        assert_eq!(dec.isolated, None);
        assert!(dec.order.is_empty());
    }
}
