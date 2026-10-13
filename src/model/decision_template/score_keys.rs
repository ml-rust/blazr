//! The decider's score levels keyed by level number: the key order its
//! renderer shows the levels in, so a level named by its key resolves to
//! the level index the answer stage reports.

use serde_json::{Map, Value};

use super::py_text::{py_float, py_repr_str};

/// `criteria`, else `options` when `criteria` is absent; null is absent.
pub(crate) fn criteria_of(spec: &Map<String, Value>) -> Option<&Value> {
    match spec.get("criteria") {
        Some(v) => Some(v).filter(|v| !v.is_null()),
        None => spec.get("options").filter(|v| !v.is_null()),
    }
}

/// The `(key, description)` pairs of a `{level number: description}` map,
/// in the order of the keys read as floats.
///
/// # Errors
///
/// When a key does not read as a float, or reads as NaN.
pub(crate) fn keyed_levels(map: &Map<String, Value>) -> Result<Vec<(&String, &Value)>, String> {
    let mut keyed = Vec::with_capacity(map.len());
    for (k, v) in map {
        let f = py_float(k)?;
        if f.is_nan() {
            return Err(format!(
                "score criteria: the level key {} reads as NaN, which has no order",
                py_repr_str(k)
            ));
        }
        keyed.push((f, k, v));
    }
    // Stable, and -0.0 equals 0.0, as in Python's `sorted`.
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(keyed.into_iter().map(|(_, k, v)| (k, v)).collect())
}

/// The keys of a decider score question's level map, in the level order
/// `systemone::render_question` gives the levels. `None` when the levels
/// are a list or a key does not read as a number.
pub fn score_level_keys(spec: &Value) -> Option<Vec<String>> {
    let Some(Value::Object(map)) = criteria_of(spec.as_object()?) else {
        return None;
    };
    let keyed = keyed_levels(map).ok()?;
    Some(keyed.into_iter().map(|(k, _)| k.clone()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keys_follow_the_numeric_level_order() {
        let spec = json!({"type": "score", "criteria": {"10": "high", "2": "low", "5.5": "mid"}});
        assert_eq!(
            score_level_keys(&spec),
            Some(vec!["2".to_string(), "5.5".to_string(), "10".to_string()])
        );
        let options = json!({"type": "score", "options": {"1": "a", "0": "b"}});
        assert_eq!(
            score_level_keys(&options),
            Some(vec!["0".to_string(), "1".to_string()])
        );
        assert_eq!(
            score_level_keys(&json!({"criteria": ["a", "b"]})),
            None,
            "a list has no keys"
        );
        assert_eq!(score_level_keys(&json!({"criteria": {"x": "a"}})), None);
        assert_eq!(score_level_keys(&json!("not an object")), None);
    }

    #[test]
    fn nan_keys_have_no_order() {
        let map = json!({"nan": "a", "1": "b"});
        let err = keyed_levels(map.as_object().expect("object")).expect_err("refused");
        assert!(err.contains("NaN"), "{err}");
    }
}
