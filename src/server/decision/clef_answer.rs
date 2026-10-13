//! Clef's wire answers, as `joint_schema_model.py` `systemone_answer`
//! builds them: every value rounded to 4 digits from the head's plain
//! softmax.
//!
//! - Noul: `noul` is p(true).
//! - Choice: the probability of each criteria key in the order the request
//!   wrote them; `choice` the first most likely key in that order;
//!   `confidence` its probability, not chance-corrected.
//! - Score: `score` the expected level index `sum(i * p_i)`, `confidence`
//!   the largest level probability, `legend` the criteria items as sent,
//!   and the probability of each level.
//!
//! A Score field of one level reaches the answer stage as a one-option
//! Choice, since boostr scores a scale of two levels or more; its field
//! type turns it back into a Score: score 0, confidence 1.

use boostr::inference::decision::{Answer, DecisionResult};
use boostr::model::decision::FieldKind;
use serde_json::{Map, Value};

use super::answer::{keyed, level_keys, name, object, py_round, round4 as r4};
use crate::model::decision_template::clef::JointField;
use crate::model::decision_template::{DecisionQuestion, RenderedDecision};

/// The probability of head option `h`.
fn at(q: &DecisionQuestion, probs: &[f64], h: usize) -> Result<f64, String> {
    probs
        .get(h)
        .copied()
        .ok_or_else(|| format!("question '{}' has no option {h}", q.id))
}

/// The Choice answer: probabilities and the pick in request order.
fn choice(q: &DecisionQuestion, field: &JointField, probs: &[f64]) -> Result<Value, String> {
    let mut best: Option<(usize, f64)> = None;
    let mut by_key = Map::with_capacity(field.wire_order.len());
    for &h in &field.wire_order {
        let p = at(q, probs, h)?;
        // Python's `max` keeps the first of equal values.
        if best.is_none_or(|(_, top)| p > top) {
            best = Some((h, p));
        }
        let key = q
            .names
            .get(h)
            .ok_or_else(|| format!("question '{}' has no option {h}", q.id))?;
        by_key.insert(key.clone(), r4(p));
    }
    let (h, p) = best.ok_or_else(|| format!("question '{}' has no options", q.id))?;
    Ok(object(vec![
        ("type", "choice".into()),
        ("choice", name(q, h)?),
        ("confidence", r4(p)),
        ("probabilities", Value::Object(by_key)),
    ]))
}

/// The Score answer over the level distribution `p`.
fn score(field: &JointField, p: &[f64]) -> Value {
    let expected: f64 = p.iter().enumerate().map(|(i, x)| i as f64 * x).sum();
    let top = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let legend = Value::Object(
        level_keys(field.legend.len())
            .zip(field.legend.iter().cloned())
            .collect(),
    );
    object(vec![
        ("type", "score".into()),
        ("score", r4(expected)),
        ("confidence", r4(top)),
        ("legend", legend),
        (
            "probabilities",
            keyed(level_keys(p.len()), p.iter().map(|&x| py_round(x, 4))),
        ),
    ])
}

/// Clef's answer to `q`, whose field is `field`.
///
/// # Errors
///
/// When the answer's kind does not fit the field's type, or names an
/// option the question does not have.
pub fn clef_answer(
    q: &DecisionQuestion,
    field: &JointField,
    answer: &Answer,
) -> Result<Value, String> {
    match (field.item.kind, answer) {
        (FieldKind::Noul, Answer::Noul { p_yes, .. }) => {
            Ok(object(vec![("type", "noul".into()), ("noul", r4(*p_yes))]))
        }
        (FieldKind::Choice, Answer::Choice { probs, .. }) => choice(q, field, probs),
        (FieldKind::Score, Answer::Score { probs, .. } | Answer::Choice { probs, .. }) => {
            Ok(score(field, probs))
        }
        (kind, other) => Err(format!(
            "question '{}': no Clef answer for a {kind:?} field from {other:?}",
            q.id
        )),
    }
}

/// The `answers` object of a Clef record: each question's id to its wire
/// answer, in request order.
///
/// # Errors
///
/// When the record has no joint layout, or `result` does not hold one
/// answer per field of its kind.
pub fn clef_answers(
    rendered: &RenderedDecision,
    result: &DecisionResult,
) -> Result<Map<String, Value>, String> {
    let layout = rendered
        .joint
        .as_ref()
        .ok_or_else(|| "a Clef record was rendered without its fields".to_string())?;
    let n = rendered.questions.len();
    if layout.fields.len() != n || result.answers.len() != n {
        return Err(format!(
            "the run returned {} answers for {} fields of {n} questions",
            result.answers.len(),
            layout.fields.len()
        ));
    }
    rendered
        .questions
        .iter()
        .zip(&layout.fields)
        .zip(&result.answers)
        .map(|((q, field), a)| Ok((q.id.clone(), clef_answer(q, field, a)?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::inference::decision::{answer_from_probs, QuestionMeta};
    use boostr::model::decision::joint_schema::joint_answer_options;
    use boostr::model::decision::{JointOption, JointSchemaItem};
    use serde_json::json;

    /// A field of `kind` with `names` in head order, sent in `wire_order`.
    fn field(
        kind: FieldKind,
        names: &[&str],
        wire_order: Vec<usize>,
        legend: Vec<Value>,
    ) -> (DecisionQuestion, JointField) {
        let item = JointSchemaItem {
            kind,
            question: 0..1,
            options: (0..names.len())
                .map(|j| JointOption {
                    span: j + 1..j + 2,
                    ids: vec![j as u32],
                })
                .collect(),
        };
        let q = DecisionQuestion {
            id: "q".into(),
            meta: item.meta(),
            names: names.iter().map(|n| n.to_string()).collect(),
            legend: legend.iter().map(|v| v.to_string()).collect(),
            scoring_rows: 0..1,
        };
        let field = JointField {
            item,
            wire_order,
            legend,
        };
        (q, field)
    }

    fn answer(q: &DecisionQuestion, f: &JointField, probs: Vec<f64>) -> Value {
        let meta: QuestionMeta = q.meta.clone();
        let a = answer_from_probs(&meta, probs, &joint_answer_options()).expect("answers");
        clef_answer(q, f, &a).expect("formats")
    }

    // Expected values are `systemone_answer` on the same probabilities.

    #[test]
    fn noul_is_p_true() {
        let (q, f) = field(FieldKind::Noul, &["true", "false"], vec![0, 1], Vec::new());
        assert_eq!(
            answer(
                &q,
                &f,
                vec![0.924_959_242_343_902_6, 0.075_040_720_403_194_43]
            ),
            json!({"type": "noul", "noul": 0.925})
        );
    }

    /// Probabilities in the order the request wrote the keys; the pick the
    /// first most likely key in that order; the confidence its probability.
    #[test]
    fn choice_follows_the_request_order() {
        // Sent as sales, billing, support, security; head order sorted.
        let names = ["billing", "sales", "security", "support"];
        let (q, f) = field(FieldKind::Choice, &names, vec![1, 0, 3, 2], Vec::new());
        let p = vec![
            0.920_251_250_267_028_8,
            0.069_048_076_868_057_25,
            0.004_278_288_222_849_369,
            0.006_422_471_255_064_011,
        ];
        let got = answer(&q, &f, p);
        assert_eq!(
            got,
            json!({"type": "choice", "choice": "billing", "confidence": 0.9203,
                "probabilities": {"sales": 0.069, "billing": 0.9203, "support": 0.0064,
                    "security": 0.0043}})
        );
        let keys: Vec<&String> = got["probabilities"]
            .as_object()
            .expect("object")
            .keys()
            .collect();
        assert_eq!(keys, ["sales", "billing", "support", "security"]);

        let (q, f) = field(FieldKind::Choice, &["a", "b"], vec![1, 0], Vec::new());
        assert_eq!(
            answer(&q, &f, vec![0.5, 0.5])["choice"],
            json!("b"),
            "a tie goes to the key sent first"
        );
        let (q, f) = field(FieldKind::Choice, &["only"], vec![0], Vec::new());
        assert_eq!(
            answer(&q, &f, vec![1.0]),
            json!({"type": "choice", "choice": "only", "confidence": 1.0,
                "probabilities": {"only": 1.0}})
        );
    }

    #[test]
    fn score_is_the_expected_index_and_the_top_probability() {
        let legend = vec![json!("negative"), json!("neutral"), json!({"x": 1})];
        let (q, f) = field(FieldKind::Score, &["0", "1", "2"], vec![0, 1, 2], legend);
        let p = vec![
            0.947_489_798_069_000_2,
            0.050_018_090_009_689_33,
            0.002_492_074_854_671_955,
        ];
        let got = answer(&q, &f, p.clone());
        let expected = p[1] + 2.0 * p[2];
        assert_eq!(
            got,
            json!({"type": "score", "score": py_round(expected, 4), "confidence": 0.9475,
                "legend": {"0": "negative", "1": "neutral", "2": {"x": 1}},
                "probabilities": {"0": 0.9475, "1": 0.05, "2": 0.0025}})
        );
        let keys: Vec<&String> = got.as_object().expect("object").keys().collect();
        assert_eq!(
            keys,
            ["type", "score", "confidence", "legend", "probabilities"]
        );

        let (q, f) = field(FieldKind::Score, &["0"], vec![0], vec![json!("only")]);
        assert_eq!(
            answer(&q, &f, vec![1.0]),
            json!({"type": "score", "score": 0.0, "confidence": 1.0,
                "legend": {"0": "only"}, "probabilities": {"0": 1.0}})
        );
    }

    #[test]
    fn a_kind_that_does_not_fit_is_an_error() {
        let (q, f) = field(FieldKind::Noul, &["true", "false"], vec![0, 1], Vec::new());
        let a = Answer::Choice {
            index: 0,
            confidence: 0.0,
            probs: vec![0.5, 0.5],
        };
        assert!(clef_answer(&q, &f, &a).is_err());
    }
}
