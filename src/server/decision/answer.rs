//! Each family's wire answer from boostr's answers, as the family's own
//! server builds it.
//!
//! - Decider (`systemone.format_answer` and `assemble`): the distribution
//!   renormalized, then 4 digits; `score` to 2 digits; `x_p_max`,
//!   `certainty`; the Score `confidence` by level distance; an isolated
//!   Score adds `level_fit` and `fit_mass`.
//! - Jebadiah (`decision_entry`, then AINode's `answer_from_decision`): the
//!   distribution to 6 digits first, and everything else from the rounded
//!   values; the Score `confidence` chance-corrected.
//! - Kev (`to_answers`): see [`super::kev_answer`].

use boostr::inference::decision::confidence::first_argmax;
use boostr::inference::decision::{
    certainty, choice_confidence, score_confidence, Answer, DecisionResult, ScoreConfidence,
};
use serde_json::{Map, Value};

use super::kev_answer::kev_answer;
use crate::model::decision_template::{DecisionQuestion, Family};

/// Python's `round(x, digits)`: the correctly rounded decimal, ties to
/// even, read back as the nearest double.
pub fn py_round(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

pub(super) fn object(pairs: Vec<(&str, Value)>) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

pub(super) fn keyed(
    keys: impl Iterator<Item = String>,
    values: impl Iterator<Item = f64>,
) -> Value {
    Value::Object(keys.zip(values.map(Value::from)).collect())
}

pub(super) fn level_keys(n: usize) -> impl Iterator<Item = String> {
    (0..n).map(|i| i.to_string())
}

/// The `legend` of a Score question: level index to level name.
pub(super) fn legend(q: &DecisionQuestion) -> Value {
    Value::Object(
        q.legend
            .iter()
            .enumerate()
            .map(|(i, name)| (i.to_string(), Value::from(name.as_str())))
            .collect(),
    )
}

pub(super) fn name(q: &DecisionQuestion, index: usize) -> Result<Value, String> {
    q.names
        .get(index)
        .map(|n| Value::from(n.as_str()))
        .ok_or_else(|| format!("question '{}' has no option {index}", q.id))
}

/// `p / sum(p)`, a zero sum left as it is.
fn renormalized(probs: &[f64]) -> Vec<f64> {
    let total: f64 = probs.iter().sum();
    let total = if total == 0.0 { 1.0 } else { total };
    probs.iter().map(|p| p / total).collect()
}

/// The decider's answer to `q`.
fn decider(q: &DecisionQuestion, answer: &Answer) -> Result<Value, String> {
    let r4 = |x: f64| Value::from(py_round(x, 4));
    let summary = |p: &[f64]| {
        let j = first_argmax(p);
        (j, r4(p[j]), r4(certainty(p)))
    };
    match answer {
        Answer::Noul { p_yes, .. } => {
            Ok(object(vec![("type", "noul".into()), ("noul", r4(*p_yes))]))
        }
        Answer::Choice { probs, .. } => {
            let p = renormalized(probs);
            let (j, p_max, sure) = summary(&p);
            Ok(object(vec![
                ("type", "choice".into()),
                ("choice", name(q, j)?),
                ("confidence", r4(choice_confidence(&p))),
                ("x_p_max", p_max),
                ("certainty", sure),
                (
                    "probabilities",
                    keyed(q.names.iter().cloned(), p.iter().map(|&x| py_round(x, 4))),
                ),
            ]))
        }
        Answer::Score {
            probs, fit_mass, ..
        } => {
            let p = renormalized(probs);
            let (_, p_max, sure) = summary(&p);
            let expected: f64 = p.iter().enumerate().map(|(i, x)| i as f64 * x).sum();
            let mut pairs = vec![
                ("type", "score".into()),
                ("score", Value::from(py_round(expected, 2))),
                (
                    "confidence",
                    r4(score_confidence(&p, ScoreConfidence::Distance)),
                ),
                ("x_p_max", p_max),
                ("certainty", sure),
                ("legend", legend(q)),
                (
                    "probabilities",
                    keyed(level_keys(p.len()), p.iter().map(|&x| py_round(x, 4))),
                ),
            ];
            if let Some(mass) = fit_mass {
                let fit = probs.iter().map(|&x| py_round(x * mass, 4));
                pairs.push(("level_fit", keyed(level_keys(probs.len()), fit)));
                pairs.push(("fit_mass", r4(*mass)));
            }
            Ok(object(pairs))
        }
        other => Err(format!(
            "question '{}': no decider answer for {other:?}",
            q.id
        )),
    }
}

/// AINode's `probability`: a finite value clipped to `[0, 1]`, else 0.
fn probability(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Jebadiah's answer to `q`.
fn jebadiah(q: &DecisionQuestion, answer: &Answer) -> Result<Value, String> {
    let rounded = |probs: &[f64]| -> Vec<f64> {
        probs.iter().map(|&p| probability(py_round(p, 6))).collect()
    };
    match answer {
        Answer::Noul { p_yes, .. } => Ok(object(vec![
            ("type", "noul".into()),
            ("noul", Value::from(probability(py_round(*p_yes, 6)))),
        ])),
        Answer::Choice { probs, .. } => {
            let dist = rounded(probs);
            let picked = first_argmax(&dist);
            Ok(object(vec![
                ("type", "choice".into()),
                ("choice", name(q, picked)?),
                (
                    "confidence",
                    Value::from(py_round(choice_confidence(&dist), 6)),
                ),
                (
                    "probabilities",
                    keyed(q.names.iter().cloned(), dist.into_iter()),
                ),
            ]))
        }
        Answer::Score { probs, .. } => {
            let dist = rounded(probs);
            let expected: f64 = dist.iter().enumerate().map(|(i, x)| i as f64 * x).sum();
            let confidence = score_confidence(&dist, ScoreConfidence::ChanceCorrected);
            Ok(object(vec![
                ("type", "score".into()),
                ("score", Value::from(py_round(expected, 6))),
                ("confidence", Value::from(py_round(confidence, 6))),
                ("legend", legend(q)),
                (
                    "probabilities",
                    keyed(level_keys(dist.len()), dist.into_iter()),
                ),
            ]))
        }
        other => Err(format!(
            "question '{}': no Jebadiah answer for {other:?}",
            q.id
        )),
    }
}

/// The `answers` object: each question's id to its family's wire answer,
/// in question order.
///
/// # Errors
///
/// When `result` does not hold one answer per question of its kind.
pub fn answers_json(
    family: Family,
    questions: &[DecisionQuestion],
    result: &DecisionResult,
) -> Result<Map<String, Value>, String> {
    if result.answers.len() != questions.len() {
        return Err(format!(
            "the run returned {} answers for {} questions",
            result.answers.len(),
            questions.len()
        ));
    }
    questions
        .iter()
        .zip(&result.answers)
        .map(|(q, a)| {
            let value = match family {
                Family::Decider => decider(q, a)?,
                Family::Jebadiah => jebadiah(q, a)?,
                Family::Kev => kev_answer(q, a)?,
            };
            Ok((q.id.clone(), value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::inference::decision::{QuestionKind, QuestionMeta};
    use serde_json::json;

    fn question(names: &[&str], legend: &[&str], kind: QuestionKind) -> DecisionQuestion {
        DecisionQuestion {
            id: "q".into(),
            meta: QuestionMeta {
                kind,
                options: names.len(),
            },
            names: names.iter().map(|n| n.to_string()).collect(),
            legend: legend.iter().map(|n| n.to_string()).collect(),
            scoring_rows: 0..1,
        }
    }

    #[test]
    fn py_round_rounds_like_python() {
        assert_eq!(py_round(0.98893333, 4), 0.9889);
        // 2.675 is stored just below 2.675, so Python gives 2.67.
        assert_eq!(py_round(2.675, 2), 2.67);
        // 0.125 is exact: a tie, to even.
        assert_eq!(py_round(0.125, 2), 0.12);
        assert_eq!(py_round(1e-9, 6), 0.0);
    }

    #[test]
    fn decider_choice_and_isolated_score() {
        let q = question(&["a", "b"], &[], QuestionKind::Choice);
        let a = Answer::Choice {
            index: 0,
            confidence: 0.0,
            probs: vec![0.75, 0.25],
        };
        let h = -(0.75f64 * 0.75f64.ln() + 0.25f64 * 0.25f64.ln());
        assert_eq!(
            decider(&q, &a).expect("answers"),
            json!({"type": "choice", "choice": "a", "confidence": 0.5, "x_p_max": 0.75,
                "certainty": py_round(1.0 - h / 2f64.ln(), 4),
                "probabilities": {"a": 0.75, "b": 0.25}})
        );

        let kind = QuestionKind::Score {
            values: None,
            isolated: true,
        };
        let q = question(&["0", "1", "2"], &["low", "mid", "high"], kind);
        let a = Answer::Score {
            expected: 0.0,
            normalized: 0.0,
            confidence: 0.0,
            probs: vec![0.5, 0.25, 0.25],
            fit_mass: Some(1.2),
        };
        let got = decider(&q, &a).expect("answers");
        let keys: Vec<&str> = got
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "type",
                "score",
                "confidence",
                "x_p_max",
                "certainty",
                "legend",
                "probabilities",
                "level_fit",
                "fit_mass"
            ]
        );
        assert_eq!(got["score"], json!(0.75));
        assert_eq!(got["legend"], json!({"0": "low", "1": "mid", "2": "high"}));
        assert_eq!(got["level_fit"], json!({"0": 0.6, "1": 0.3, "2": 0.3}));
        assert_eq!(got["fit_mass"], json!(1.2));
    }

    #[test]
    fn jebadiah_rounds_first_and_picks_from_the_rounded_values() {
        let q = question(&["x", "y", "z"], &[], QuestionKind::Choice);
        let a = Answer::Choice {
            index: 1,
            confidence: 0.0,
            probs: vec![0.4000001, 0.4000004, 0.2],
        };
        let got = jebadiah(&q, &a).expect("answers");
        assert_eq!(
            got["choice"],
            json!("x"),
            "a tie after rounding goes to the first"
        );
        assert_eq!(got["probabilities"], json!({"x": 0.4, "y": 0.4, "z": 0.2}));

        let q = question(&["true", "false"], &[], QuestionKind::Noul { yes: 0 });
        let a = Answer::Noul {
            p_yes: 0.97928912,
            decided: Some(true),
        };
        assert_eq!(
            jebadiah(&q, &a).expect("answers"),
            json!({"type": "noul", "noul": 0.979289})
        );

        let kind = QuestionKind::Score {
            values: None,
            isolated: false,
        };
        let q = question(&["0", "1"], &["bad", "good"], kind);
        let a = Answer::Score {
            expected: 0.0,
            normalized: 0.0,
            confidence: 0.0,
            probs: vec![0.2, 0.8],
            fit_mass: None,
        };
        assert_eq!(
            jebadiah(&q, &a).expect("answers"),
            json!({"type": "score", "score": 0.8, "confidence": 0.6,
                "legend": {"0": "bad", "1": "good"}, "probabilities": {"0": 0.2, "1": 0.8}})
        );
    }

    #[test]
    fn answer_count_must_match() {
        let q = question(&["a", "b"], &[], QuestionKind::Choice);
        let result = DecisionResult {
            answers: Vec::new(),
            raw_logits: Vec::new(),
            temperatures: Vec::new(),
        };
        assert!(answers_json(Family::Decider, &[q], &result).is_err());
    }
}
