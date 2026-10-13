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
    /// Milliseconds from the request's arrival to its answers.
    pub latency_ms: f64,
    /// The model's fitted temperatures.
    pub calibration: Option<&'a Calibration>,
    /// `"calibration": "raw"` was asked for.
    pub raw: bool,
    /// The model's standing warnings, then the request's.
    pub warnings: Vec<String>,
}

/// The response body.
pub fn response_body(facts: ResponseFacts<'_>, answers: Map<String, Value>) -> Value {
    let applied = facts.calibration.is_some() && !facts.raw;
    json!({
        "model": facts.model,
        "answers": answers,
        "usage": {
            "input_tokens": facts.input_tokens,
            "output_tokens": 0,
            "cached_tokens": 0
        },
        "latency_ms": (facts.latency_ms * 10.0).round() / 10.0,
        "calibration": {
            "applied": applied,
            "temperatures": facts.calibration.map(|c| &c.temperatures),
            "source": facts.calibration.map(|c| c.source.as_str())
        },
        "warnings": facts.warnings
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn calibration_applies_unless_raw_or_absent() {
        let calibration = Calibration {
            source: "temperatures.json".into(),
            temperatures: BTreeMap::from([("choice".to_string(), 1.1)]),
        };
        let facts = ResponseFacts {
            model: "m",
            input_tokens: 12,
            latency_ms: 3.12,
            calibration: Some(&calibration),
            raw: false,
            warnings: vec!["w".into()],
        };
        let body = response_body(facts.clone(), Map::new());
        assert_eq!(
            body,
            json!({"model": "m", "answers": {},
                "usage": {"input_tokens": 12, "output_tokens": 0, "cached_tokens": 0},
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
                ..facts
            },
            Map::new(),
        );
        assert_eq!(
            none["calibration"],
            json!({"applied": false, "temperatures": null, "source": null})
        );
    }
}
