//! The `/v1/systemone` response around the answers: model, usage, latency,
//! calibration and warnings.

use serde_json::{json, Map, Value};

use crate::loader::decision::Calibration;

/// Everything a response reports besides the answers.
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseFacts<'a> {
    /// The model that answered.
    pub model: &'a str,
    /// Prompt tokens, the shared prefix of the rows counted once.
    pub input_tokens: usize,
    /// Prompt tokens the run did not run: the prefix whose state the
    /// worker kept from an earlier request. 0 on a miss or with no cache.
    pub cached_input_tokens: usize,
    /// Milliseconds from the request's arrival to its answers.
    pub latency_ms: f64,
    /// The model's fitted temperatures.
    pub calibration: Option<&'a Calibration>,
    /// `"calibration": "raw"` was asked for.
    pub raw: bool,
    /// The model's standing warnings, then the request's.
    pub warnings: Vec<String>,
}

/// The `calibration` object: whether the fitted temperatures applied, and
/// which.
fn calibration_json(calibration: Option<&Calibration>, raw: bool) -> Value {
    json!({
        "applied": calibration.is_some() && !raw,
        "temperatures": calibration.map(|c| &c.temperatures),
        "source": calibration.map(|c| c.source.as_str())
    })
}

/// The response body. `usage.cached_tokens` and
/// `usage.cached_input_tokens` both count the prompt tokens a kept prefix
/// state saved.
pub fn response_body(facts: ResponseFacts<'_>, answers: Map<String, Value>) -> Value {
    json!({
        "model": facts.model,
        "answers": answers,
        "usage": {
            "input_tokens": facts.input_tokens,
            "output_tokens": 0,
            "cached_tokens": facts.cached_input_tokens,
            "cached_input_tokens": facts.cached_input_tokens
        },
        "latency_ms": (facts.latency_ms * 10.0).round() / 10.0,
        "calibration": calibration_json(facts.calibration, facts.raw),
        "warnings": facts.warnings
    })
}

/// What a Kev response reports besides the answers and [`ResponseFacts`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KevUsage {
    /// Kev's `output_tokens`: the tokens of the answers serialized as
    /// Python's `json.dumps` writes them.
    pub output_tokens: usize,
    /// When the server cuts long states: the state's tokens and the tokens
    /// read, both with the state delimiter. `None` when it refuses them.
    pub truncation: Option<(usize, usize)>,
}

/// The Kev response body, as `kev.serve` writes it: `model`, `answers`,
/// `usage` with `input_tokens` (the state once, then every branch) and
/// `output_tokens`, and `latency_ms`. A server that cuts long states adds
/// `usage.state_tokens`, `usage.state_tokens_used` and `truncated`. The
/// blazr `calibration` and `warnings` follow; Kev's clients ignore fields
/// they do not model. `usage` holds Kev's fields only, so a kept prefix
/// state shows in `/v1/models` and the metrics, not here.
pub fn kev_response_body(
    facts: ResponseFacts<'_>,
    kev: KevUsage,
    answers: Map<String, Value>,
) -> Value {
    let mut usage = Map::new();
    usage.insert("input_tokens".into(), facts.input_tokens.into());
    usage.insert("output_tokens".into(), kev.output_tokens.into());
    if let Some((state, used)) = kev.truncation {
        usage.insert("state_tokens".into(), state.into());
        usage.insert("state_tokens_used".into(), used.into());
    }
    let mut body = Map::new();
    body.insert("model".into(), facts.model.into());
    body.insert("answers".into(), Value::Object(answers));
    body.insert("usage".into(), Value::Object(usage));
    let latency = (facts.latency_ms * 10.0).round() / 10.0;
    body.insert("latency_ms".into(), latency.into());
    if let Some((state, used)) = kev.truncation {
        body.insert("truncated".into(), (state > used).into());
    }
    body.insert(
        "calibration".into(),
        calibration_json(facts.calibration, facts.raw),
    );
    body.insert("warnings".into(), facts.warnings.into());
    Value::Object(body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn calibration_applies_unless_raw_or_absent() {
        let calibration = Calibration::from_file(
            "temperatures.json",
            BTreeMap::from([("choice".to_string(), 1.1)]),
        );
        let facts = ResponseFacts {
            model: "m",
            input_tokens: 12,
            cached_input_tokens: 0,
            latency_ms: 3.12,
            calibration: Some(&calibration),
            raw: false,
            warnings: vec!["w".into()],
        };
        let body = response_body(facts.clone(), Map::new());
        assert_eq!(
            body,
            json!({"model": "m", "answers": {},
                "usage": {"input_tokens": 12, "output_tokens": 0, "cached_tokens": 0,
                    "cached_input_tokens": 0},
                "latency_ms": 3.1,
                "calibration": {"applied": true, "temperatures": {"choice": 1.1},
                    "source": "temperatures.json"},
                "warnings": ["w"]})
        );
        let raw = response_body(
            ResponseFacts {
                raw: true,
                ..facts.clone()
            },
            Map::new(),
        );
        assert_eq!(raw["calibration"]["applied"], json!(false));
        let none = response_body(
            ResponseFacts {
                calibration: None,
                ..facts.clone()
            },
            Map::new(),
        );
        assert_eq!(
            none["calibration"],
            json!({"applied": false, "temperatures": null, "source": null})
        );
        let hit = response_body(
            ResponseFacts {
                cached_input_tokens: 9,
                ..facts
            },
            Map::new(),
        );
        let usage = json!({"input_tokens": 12, "output_tokens": 0, "cached_tokens": 9,
            "cached_input_tokens": 9});
        assert_eq!(hit["usage"], usage);
    }

    #[test]
    fn kev_usage_and_truncation_marks() {
        let facts = ResponseFacts {
            model: "kev-4b",
            input_tokens: 119,
            cached_input_tokens: 80,
            latency_ms: 12.34,
            calibration: None,
            raw: false,
            warnings: Vec::new(),
        };
        let refusing = KevUsage {
            output_tokens: 40,
            truncation: None,
        };
        let body = kev_response_body(facts.clone(), refusing, Map::new());
        assert_eq!(
            body["usage"],
            json!({"input_tokens": 119, "output_tokens": 40})
        );
        assert_eq!(body["latency_ms"], json!(12.3));
        assert!(body.get("truncated").is_none());
        let keys: Vec<&String> = body.as_object().expect("object").keys().collect();
        assert_eq!(
            keys,
            [
                "model",
                "answers",
                "usage",
                "latency_ms",
                "calibration",
                "warnings"
            ]
        );

        let cutting = KevUsage {
            output_tokens: 40,
            truncation: Some((70000, 65536)),
        };
        let body = kev_response_body(facts.clone(), cutting, Map::new());
        assert_eq!(body["usage"]["state_tokens"], json!(70000));
        assert_eq!(body["usage"]["state_tokens_used"], json!(65536));
        assert_eq!(body["truncated"], json!(true));
        let whole = KevUsage {
            output_tokens: 40,
            truncation: Some((72, 72)),
        };
        assert_eq!(
            kev_response_body(facts, whole, Map::new())["truncated"],
            json!(false)
        );
    }
}
