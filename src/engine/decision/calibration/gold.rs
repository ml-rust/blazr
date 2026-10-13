//! A labelled question's gold answer as the index of the option or level
//! the rendered question reports.
//!
//! - Choice: the option key, as the request spells it.
//! - Noul: `true` or `false` (also `1` or `0`).
//! - Score: a level index (an integer), or the level as the request spells
//!   it (a string). The decider names a level by its key in a `{level
//!   number: description}` map, in the numeric order it renders the levels
//!   in, or by its text in a list. Jebadiah names a level by its name in
//!   the list or map, stripped as it renders it.

use boostr::inference::decision::QuestionKind;
use serde_json::Value;

use crate::model::decision_template::py_text::py_strip;
use crate::model::decision_template::score_keys::score_level_keys;
use crate::model::decision_template::{DecisionQuestion, Family};

fn listed(names: &[String]) -> String {
    names
        .iter()
        .map(|n| format!("'{n}'"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The index of `gold` among `question`'s options or levels. `spec` is the
/// question as the request sent it.
///
/// # Errors
///
/// What `gold` is and the answers the question takes, when it is not one
/// of them.
pub fn gold_index(
    family: Family,
    question: &DecisionQuestion,
    spec: &Value,
    gold: &Value,
) -> Result<usize, String> {
    let names = &question.names;
    match &question.meta.kind {
        QuestionKind::Choice => {
            let key = gold
                .as_str()
                .ok_or_else(|| format!("a choice gold is an option key string, got {gold}"))?;
            names.iter().position(|n| n == key).ok_or_else(|| {
                format!(
                    "'{key}' is not an option; the options are {}",
                    listed(names)
                )
            })
        }
        QuestionKind::Noul { .. } => {
            let side = match gold {
                Value::Bool(b) => *b,
                Value::Number(n) if n.as_u64() == Some(1) => true,
                Value::Number(n) if n.as_u64() == Some(0) => false,
                _ => return Err(format!("a noul gold is true or false, got {gold}")),
            };
            let name = if side { "true" } else { "false" };
            names
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| format!("the question has no '{name}' option"))
        }
        QuestionKind::Score { .. } => {
            let levels = names.len();
            match gold {
                Value::Number(n) => n
                    .as_u64()
                    .and_then(|g| usize::try_from(g).ok())
                    .filter(|&g| g < levels)
                    .ok_or_else(|| {
                        format!(
                            "level {n} is not a level index; the question has levels 0..{}",
                            levels.saturating_sub(1)
                        )
                    }),
                Value::String(s) => score_by_name(family, question, spec, s),
                _ => Err(format!(
                    "a score gold is a level index or the level as the request spells it, \
                     got {gold}"
                )),
            }
        }
        _ => Err(format!(
            "question '{}' has a kind calibration does not read",
            question.id
        )),
    }
}

fn score_by_name(
    family: Family,
    question: &DecisionQuestion,
    spec: &Value,
    name: &str,
) -> Result<usize, String> {
    let (spelled, wanted) = match family {
        Family::Decider => match score_level_keys(spec) {
            Some(keys) => (keys, name),
            None => (question.legend.clone(), name),
        },
        Family::Jebadiah => (question.legend.clone(), py_strip(name)),
    };
    spelled.iter().position(|l| l == wanted).ok_or_else(|| {
        format!(
            "'{name}' is not a level; the levels are {}, or a level index",
            listed(&spelled)
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::inference::decision::QuestionMeta;
    use serde_json::json;

    fn question(kind: QuestionKind, names: &[&str], legend: &[&str]) -> DecisionQuestion {
        DecisionQuestion {
            id: "q".into(),
            meta: QuestionMeta {
                kind,
                options: names.len(),
            },
            names: names.iter().map(|s| s.to_string()).collect(),
            legend: legend.iter().map(|s| s.to_string()).collect(),
            scoring_rows: 0..1,
        }
    }

    fn score(isolated: bool) -> QuestionKind {
        QuestionKind::Score {
            values: None,
            isolated,
        }
    }

    #[test]
    fn choice_and_noul() {
        let q = question(QuestionKind::Choice, &["billing", "support"], &[]);
        let at = |g: Value| gold_index(Family::Decider, &q, &json!({}), &g);
        assert_eq!(at(json!("support")), Ok(1));
        assert!(at(json!("sales"))
            .expect_err("absent")
            .contains("'billing', 'support'"));
        assert!(at(json!(0)).is_err());
        let decider = question(QuestionKind::Noul { yes: 1 }, &["false", "true"], &[]);
        let jebadiah = question(QuestionKind::Noul { yes: 0 }, &["true", "false"], &[]);
        let noul = |q: &DecisionQuestion, g: Value| gold_index(Family::Decider, q, &json!({}), &g);
        assert_eq!(noul(&decider, json!(true)), Ok(1));
        assert_eq!(noul(&jebadiah, json!(true)), Ok(0));
        assert_eq!(noul(&decider, json!(0)), Ok(0));
        assert!(noul(&decider, json!("yes")).is_err());
        assert!(noul(&decider, json!(2)).is_err());
    }

    #[test]
    fn decider_scores_by_index_key_or_text() {
        let spec = json!({"type": "score", "criteria": {"10": "high", "1": "low", "5": "mid"}});
        let q = question(score(true), &["0", "1", "2"], &["low", "mid", "high"]);
        let at = |g: Value| gold_index(Family::Decider, &q, &spec, &g);
        assert_eq!(at(json!(2)), Ok(2));
        assert_eq!(at(json!("10")), Ok(2), "the key, in numeric order");
        assert_eq!(at(json!("1")), Ok(0));
        assert!(at(json!("high")).is_err(), "a map names levels by key");
        assert!(at(json!(3)).expect_err("range").contains("0..2"));
        assert!(at(json!(-1)).is_err());
        let list = json!({"type": "score", "criteria": ["low", "high"]});
        let q = question(score(false), &["0", "1"], &["low", "high"]);
        assert_eq!(
            gold_index(Family::Decider, &q, &list, &json!("high")),
            Ok(1)
        );
    }

    #[test]
    fn jebadiah_scores_by_stripped_name() {
        let spec = json!({"type": "score", "criteria": [" low ", "high"]});
        let q = question(score(false), &["0", "1"], &["low", "high"]);
        let at = |g: Value| gold_index(Family::Jebadiah, &q, &spec, &g);
        assert_eq!(at(json!(" low")), Ok(0));
        assert_eq!(at(json!(1)), Ok(1));
        assert!(at(json!("medium"))
            .expect_err("absent")
            .contains("'low', 'high'"));
    }
}
