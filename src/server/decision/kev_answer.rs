//! Kev's wire answers, as `kev/api.py` `to_answers` builds them: every
//! probability and derived scalar rounded to 4 digits, the distribution as
//! the model gave it, and the confidences over the distribution normalized
//! to sum 1 (all zeros read as uniform).
//!
//! - Noul: `noul` is p(yes).
//! - Choice: the first most likely key, its chance-corrected confidence,
//!   and the probability of each key.
//! - Score: the expected level `sum(i * p_i)`, the legend, the probability
//!   of each level, and the confidence by level distance.
//!
//! A Score question of one level reaches the answer stage as a one-option
//! Choice, since boostr scores two levels or more. Its legend marks it, and
//! it is written as a Score: score 0, confidence 1, its level's
//! probability.

use boostr::inference::decision::{choice_confidence, score_confidence, Answer, ScoreConfidence};
use serde_json::Value;

use super::answer::{keyed, legend, level_keys, name, object, py_round, round4 as round_prob};
use crate::model::decision_template::DecisionQuestion;

/// Kev's `_normalize`: each entry over the sum, a zero sum read as uniform.
fn normalize(p: &[f64]) -> Vec<f64> {
    let total: f64 = p.iter().sum();
    if total == 0.0 {
        vec![1.0 / p.len() as f64; p.len()]
    } else {
        p.iter().map(|x| x / total).collect()
    }
}

/// The Score answer over the level distribution `p`.
fn score(q: &DecisionQuestion, p: &[f64]) -> Value {
    let expected: f64 = p.iter().enumerate().map(|(i, x)| i as f64 * x).sum();
    let confidence = score_confidence(&normalize(p), ScoreConfidence::Distance);
    object(vec![
        ("type", "score".into()),
        ("score", round_prob(expected)),
        ("legend", legend(q)),
        (
            "probabilities",
            keyed(level_keys(p.len()), p.iter().map(|&x| py_round(x, 4))),
        ),
        ("confidence", round_prob(confidence)),
    ])
}

/// Kev's answer to `q`.
///
/// # Errors
///
/// When the answer is of a kind Kev does not answer, or names an option
/// the question does not have.
pub fn kev_answer(q: &DecisionQuestion, answer: &Answer) -> Result<Value, String> {
    match answer {
        Answer::Noul { p_yes, .. } => Ok(object(vec![
            ("type", "noul".into()),
            ("noul", round_prob(*p_yes)),
        ])),
        Answer::Choice { probs, .. } if !q.legend.is_empty() => Ok(score(q, probs)),
        Answer::Choice { index, probs, .. } => Ok(object(vec![
            ("type", "choice".into()),
            ("choice", name(q, *index)?),
            (
                "confidence",
                round_prob(choice_confidence(&normalize(probs))),
            ),
            (
                "probabilities",
                keyed(
                    q.names.iter().cloned(),
                    probs.iter().map(|&x| py_round(x, 4)),
                ),
            ),
        ])),
        Answer::Score { probs, .. } => Ok(score(q, probs)),
        other => Err(format!("question '{}': no Kev answer for {other:?}", q.id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::inference::decision::QuestionKind;
    use boostr::inference::decision::{answer_from_probs, AnswerOptions, QuestionMeta};
    use serde_json::json;

    fn question(kind: QuestionKind, names: &[&str], legend: &[&str]) -> DecisionQuestion {
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

    fn answer(q: &DecisionQuestion, probs: Vec<f64>) -> Value {
        let a = answer_from_probs(&q.meta, probs, &AnswerOptions::default()).expect("answers");
        kev_answer(q, &a).expect("formats")
    }

    // Expected values are `kev.api.to_answers` on the same distributions.

    #[test]
    fn noul_is_p_yes() {
        let q = question(QuestionKind::Noul { yes: 1 }, &["false", "true"], &[]);
        assert_eq!(
            answer(&q, vec![0.031_643_927_097_320_56, 0.968_356_072_902_679_4]),
            json!({"type": "noul", "noul": 0.9684})
        );
    }

    #[test]
    fn choice_takes_the_first_argmax_and_chance_corrects() {
        let q = question(
            QuestionKind::Choice,
            &["billing", "support", "sales", "security"],
            &[],
        );
        let got = answer(&q, vec![0.8987, 0.0376, 0.0624, 0.0013]);
        assert_eq!(
            got,
            json!({"type": "choice", "choice": "billing", "confidence": 0.8649,
                "probabilities": {"billing": 0.8987, "support": 0.0376, "sales": 0.0624,
                    "security": 0.0013}})
        );
        let tie = question(QuestionKind::Choice, &["a", "b"], &[]);
        assert_eq!(answer(&tie, vec![0.5, 0.5])["choice"], json!("a"));
        let one = question(QuestionKind::Choice, &["only"], &[]);
        assert_eq!(
            answer(&one, vec![1.0]),
            json!({"type": "choice", "choice": "only", "confidence": 1.0,
                "probabilities": {"only": 1.0}})
        );
    }

    #[test]
    fn score_is_the_expected_level_with_its_legend() {
        let kind = QuestionKind::Score {
            values: None,
            isolated: false,
        };
        let levels = [
            "not urgent at all",
            "low",
            "medium",
            "high",
            "critical, act now",
        ];
        let q = question(kind, &["0", "1", "2", "3", "4"], &levels);
        let p = vec![0.014, 0.018, 0.1519, 0.3229, 0.4932];
        let got = answer(&q, p.clone());
        let expected: f64 = p.iter().enumerate().map(|(i, x)| i as f64 * x).sum();
        assert_eq!(got["score"], json!(py_round(expected, 4)));
        assert_eq!(got["legend"]["4"], json!("critical, act now"));
        assert_eq!(got["probabilities"]["2"], json!(0.1519));
        // D = (2 + 1 + 0 + 1 + 2) / 5; E|i - 4| = 0.014*4 + 0.018*3 + 0.1519*2 + 0.3229
        let spread = 0.014 * 4.0 + 0.018 * 3.0 + 0.1519 * 2.0 + 0.3229;
        assert_eq!(got["confidence"], json!(py_round(1.0 - spread / 1.2, 4)));
        let keys: Vec<&String> = got.as_object().expect("object").keys().collect();
        assert_eq!(
            keys,
            ["type", "score", "legend", "probabilities", "confidence"]
        );
    }

    /// Kev answers a Score of one level; boostr's answer stage takes it as
    /// a one-option Choice, and the legend turns it back into a Score.
    #[test]
    fn a_one_level_score_is_score_zero_confidence_one() {
        let q = question(QuestionKind::Choice, &["0"], &["the only level"]);
        assert_eq!(
            answer(&q, vec![1.0]),
            json!({"type": "score", "score": 0.0, "legend": {"0": "the only level"},
                "probabilities": {"0": 1.0}, "confidence": 1.0})
        );
    }

    #[test]
    fn zero_mass_normalizes_to_uniform() {
        assert_eq!(normalize(&[0.0, 0.0]), vec![0.5, 0.5]);
        assert_eq!(normalize(&[1.0, 3.0]), vec![0.25, 0.75]);
    }
}
