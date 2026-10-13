//! The label file of `blazr decision calibrate`: JSON lines, each a
//! labelled request or a precomputed record.
//!
//! - Request: `{"state": ..., "questions": {id: {type, instructions,
//!   criteria}}, "labels": {id: gold}}`, plus any request field of the
//!   model's family (`isolated`, `independent`, ...). It runs at
//!   temperature 1.
//! - Record, as `decider/calibrate.py` reads them: `{"type": "choice" |
//!   "noul" | "score", "logits": [...], "gold": index}` or `{"type":
//!   "score", "level_logits": [[no, yes], ...], "gold": level}`.
//!   `probs` / `level_probs` stand in for the logits as their logarithms.
//!   A noul record's options are `[false, true]` in the model's option
//!   order, its gold 1 for true.
//!
//! Blank lines are skipped. Line numbers count from 1.

use serde_json::{Map, Value};

/// An answer type: the key of a temperature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AnswerType {
    /// A choice question.
    Choice,
    /// A yes/no question.
    Noul,
    /// A score question, listed or isolated.
    Score,
}

impl AnswerType {
    /// Every type, in report order.
    pub const ALL: [AnswerType; 3] = [AnswerType::Choice, AnswerType::Noul, AnswerType::Score];

    /// The name a temperature map keys it by.
    pub fn name(self) -> &'static str {
        match self {
            AnswerType::Choice => "choice",
            AnswerType::Noul => "noul",
            AnswerType::Score => "score",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name() == name)
    }
}

/// A request with gold answers for some of its questions.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelledRequest {
    /// The line it was read from.
    pub line: usize,
    /// The request body: `labels` and `model` removed, `calibration` set to
    /// `"raw"`.
    pub body: Value,
    /// The gold answer of each labelled question, by question id.
    pub labels: Map<String, Value>,
}

/// The logits of a precomputed record at temperature 1.
#[derive(Debug, Clone, PartialEq)]
pub enum RecordLogits {
    /// One logit per option.
    Listed(Vec<f64>),
    /// One `[no, yes]` pair per level of an isolated Score answer.
    Isolated(Vec<[f64; 2]>),
}

/// A precomputed answer with its gold.
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// The line it was read from.
    pub line: usize,
    /// Its answer type.
    pub kind: AnswerType,
    /// Its logits.
    pub logits: RecordLogits,
    /// The gold option or level index.
    pub gold: usize,
}

/// One line of a label file.
#[derive(Debug, Clone, PartialEq)]
pub enum LabelLine {
    /// A request to run.
    Request(LabelledRequest),
    /// A precomputed record.
    Record(Record),
}

/// Parse a label file.
///
/// # Errors
///
/// `line N: ...` naming the field of the first line that does not parse.
pub fn parse_label_lines(text: &str) -> Result<Vec<LabelLine>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        if raw.trim().is_empty() {
            continue;
        }
        let value: Value =
            serde_json::from_str(raw).map_err(|e| format!("line {line}: not valid JSON: {e}"))?;
        out.push(parse_line(line, value)?);
    }
    Ok(out)
}

fn parse_line(line: usize, value: Value) -> Result<LabelLine, String> {
    let Value::Object(mut body) = value else {
        return Err(format!("line {line}: not a JSON object"));
    };
    match (body.contains_key("questions"), body.contains_key("type")) {
        (true, true) => Err(format!(
            "line {line}: has both 'questions' (a labelled request) and 'type' (a record); \
             give one"
        )),
        (true, false) => {
            let labels = match body.remove("labels") {
                Some(Value::Object(labels)) => labels,
                Some(_) => {
                    return Err(format!(
                        "line {line}: 'labels' must be an object of {{question id: gold}}"
                    ))
                }
                None => {
                    return Err(format!(
                        "line {line}: 'labels' is missing; a request line needs \
                         {{question id: gold}}"
                    ))
                }
            };
            body.remove("model");
            body.insert("calibration".into(), Value::String("raw".into()));
            Ok(LabelLine::Request(LabelledRequest {
                line,
                body: Value::Object(body),
                labels,
            }))
        }
        (false, true) => parse_record(line, &body).map(LabelLine::Record),
        (false, false) => Err(format!(
            "line {line}: neither 'questions' (a labelled request) nor 'type' (a record)"
        )),
    }
}

/// A row of numbers: logits as given, probabilities as their logarithms.
fn numbers(line: usize, key: &str, value: &Value, probs: bool) -> Result<Vec<f64>, String> {
    let Value::Array(items) = value else {
        return Err(format!("line {line}: '{key}' must be an array of numbers"));
    };
    items
        .iter()
        .map(|v| {
            let x = v
                .as_f64()
                .ok_or_else(|| format!("line {line}: '{key}' holds {v}, not a number"))?;
            if probs {
                if x > 0.0 && x <= 1.0 {
                    Ok(x.ln())
                } else {
                    Err(format!(
                        "line {line}: '{key}' holds {x}; a probability must be in (0, 1], a 0 \
                         has no finite logit (give logits instead)"
                    ))
                }
            } else if x.is_finite() {
                Ok(x)
            } else {
                Err(format!(
                    "line {line}: '{key}' holds {x}, not a finite number"
                ))
            }
        })
        .collect()
}

fn parse_record(line: usize, body: &Map<String, Value>) -> Result<Record, String> {
    let kind = match body.get("type") {
        Some(Value::String(s)) => AnswerType::parse(s),
        _ => None,
    }
    .ok_or_else(|| {
        format!(
            "line {line}: 'type' is {}, expected \"choice\", \"noul\" or \"score\"",
            body.get("type")
                .map_or("missing".to_string(), Value::to_string)
        )
    })?;
    let keys = ["logits", "probs", "level_logits", "level_probs"];
    let given: Vec<&str> = keys.into_iter().filter(|k| body.contains_key(*k)).collect();
    let [key] = given[..] else {
        return Err(format!(
            "line {line}: give exactly one of 'logits', 'probs', 'level_logits', 'level_probs'"
        ));
    };
    let probs = key.ends_with("probs");
    let logits = if key.starts_with("level_") {
        if kind != AnswerType::Score {
            return Err(format!(
                "line {line}: '{key}' (isolated levels) belongs to a score record, not {}",
                kind.name()
            ));
        }
        let Some(Value::Array(rows)) = body.get(key) else {
            return Err(format!(
                "line {line}: '{key}' must be an array of [no, yes] pairs"
            ));
        };
        let mut levels = Vec::with_capacity(rows.len());
        for row in rows {
            match numbers(line, key, row, probs)?[..] {
                [no, yes] => levels.push([no, yes]),
                _ => {
                    return Err(format!(
                        "line {line}: '{key}' must hold one [no, yes] pair per level"
                    ))
                }
            }
        }
        if levels.len() < 2 {
            return Err(format!("line {line}: '{key}' needs at least two levels"));
        }
        RecordLogits::Isolated(levels)
    } else {
        let row = body
            .get(key)
            .map_or(Ok(Vec::new()), |v| numbers(line, key, v, probs))?;
        if row.len() < 2 {
            return Err(format!("line {line}: '{key}' needs at least two options"));
        }
        if kind == AnswerType::Noul && row.len() != 2 {
            return Err(format!(
                "line {line}: a noul record has exactly two options (false, true); '{key}' has {}",
                row.len()
            ));
        }
        RecordLogits::Listed(row)
    };
    let options = match &logits {
        RecordLogits::Listed(row) => row.len(),
        RecordLogits::Isolated(levels) => levels.len(),
    };
    let gold = match body.get("gold") {
        Some(Value::Number(n)) => n.as_u64().and_then(|g| usize::try_from(g).ok()),
        _ => None,
    }
    .filter(|&g| g < options)
    .ok_or_else(|| {
        format!(
            "line {line}: 'gold' must be an integer index in 0..{}, got {}",
            options - 1,
            body.get("gold")
                .map_or("nothing".to_string(), Value::to_string)
        )
    })?;
    Ok(Record {
        line,
        kind,
        logits,
        gold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn one(text: &str) -> Result<LabelLine, String> {
        parse_label_lines(text).map(|mut v| v.remove(0))
    }

    #[test]
    fn requests_run_raw_and_keep_their_family_fields() {
        let text = "\n{\"state\": {\"a\": 1}, \"questions\": {\"q\": {\"type\": \"noul\"}}, \
                    \"labels\": {\"q\": true}, \"model\": \"x\", \"isolated\": false}\n";
        let lines = parse_label_lines(text).expect("parses");
        let LabelLine::Request(r) = &lines[0] else {
            panic!("a request")
        };
        assert_eq!(r.line, 2);
        assert_eq!(r.labels["q"], json!(true));
        assert_eq!(r.body["calibration"], "raw");
        assert_eq!(r.body["isolated"], json!(false));
        assert!(r.body.get("labels").is_none() && r.body.get("model").is_none());
    }

    #[test]
    fn records_of_each_shape() {
        let rec = |t: &str| match one(t).expect("parses") {
            LabelLine::Record(r) => r,
            other => panic!("a record, got {other:?}"),
        };
        let r = rec(r#"{"type": "choice", "logits": [1.0, 2, -0.5], "gold": 2}"#);
        assert_eq!(
            (r.kind, r.gold, r.logits),
            (
                AnswerType::Choice,
                2,
                RecordLogits::Listed(vec![1.0, 2.0, -0.5])
            )
        );
        let r = rec(r#"{"type": "score", "level_logits": [[1, 2], [3, 0]], "gold": 1}"#);
        assert_eq!(
            r.logits,
            RecordLogits::Isolated(vec![[1.0, 2.0], [3.0, 0.0]])
        );
        let r = rec(r#"{"type": "noul", "probs": [0.25, 0.75], "gold": 1}"#);
        assert_eq!(
            r.logits,
            RecordLogits::Listed(vec![0.25f64.ln(), 0.75f64.ln()])
        );
    }

    #[test]
    fn every_rejection_names_its_line_and_field() {
        let cases = [
            ("[1]", "not a JSON object"),
            ("{bad", "not valid JSON"),
            ("{}", "neither 'questions'"),
            (r#"{"questions": {}, "type": "choice"}"#, "both 'questions'"),
            (r#"{"questions": {}}"#, "'labels' is missing"),
            (
                r#"{"questions": {}, "labels": [1]}"#,
                "'labels' must be an object",
            ),
            (
                r#"{"type": "bool", "logits": [1, 2], "gold": 0}"#,
                "'type' is \"bool\"",
            ),
            (r#"{"type": "choice", "gold": 0}"#, "exactly one of"),
            (
                r#"{"type": "choice", "logits": [1, 2], "probs": [0.5, 0.5], "gold": 0}"#,
                "exactly one of",
            ),
            (
                r#"{"type": "choice", "logits": [1], "gold": 0}"#,
                "at least two options",
            ),
            (
                r#"{"type": "choice", "logits": [1, "x"], "gold": 0}"#,
                "'logits' holds \"x\"",
            ),
            (
                r#"{"type": "noul", "logits": [1, 2, 3], "gold": 0}"#,
                "exactly two options",
            ),
            (
                r#"{"type": "choice", "logits": [1, 2], "gold": 2}"#,
                "'gold' must be",
            ),
            (
                r#"{"type": "choice", "logits": [1, 2], "gold": true}"#,
                "'gold' must be",
            ),
            (
                r#"{"type": "choice", "logits": [1, 2], "gold": 1.0}"#,
                "'gold' must be",
            ),
            (
                r#"{"type": "choice", "probs": [0.0, 1.0], "gold": 1}"#,
                "a probability",
            ),
            (
                r#"{"type": "noul", "level_logits": [[1, 2], [2, 1]], "gold": 0}"#,
                "belongs to a score",
            ),
            (
                r#"{"type": "score", "level_logits": [[1, 2]], "gold": 0}"#,
                "at least two levels",
            ),
            (
                r#"{"type": "score", "level_logits": [[1, 2, 3], [1, 2]], "gold": 0}"#,
                "[no, yes] pair",
            ),
        ];
        for (text, want) in cases {
            let e = one(&format!("\n\n{text}")).expect_err(text);
            assert!(e.starts_with("line 3:"), "{text}: {e}");
            assert!(e.contains(want), "{text}: {e}");
        }
    }
}
