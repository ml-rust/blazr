//! The labelled answers of a label file: records as given, and each
//! labelled request rendered by the model's renderer and run raw
//! (temperature 1) on its decision worker, as `"calibration": "raw"` runs.
//!
//! Requests queue on the worker while earlier ones run, up to
//! [`IN_FLIGHT`] at once, so the worker can merge them. A full queue waits
//! for the oldest answer before it queues more.

use std::collections::VecDeque;

use boostr::{DType, Runtime};
use serde_json::Value;

use super::gold::gold_index;
use super::labels::{LabelLine, LabelledRequest};
use super::samples::TypeSamples;
use crate::engine::decision::{DecisionFailure, DecisionService, Pending};
use crate::model::decision_template::{DecisionQuestion, JevRequest};

/// Most requests queued on the worker at once.
pub const IN_FLIGHT: usize = 64;

/// What a label file gave.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Collected {
    /// The labelled answers of each type.
    pub samples: TypeSamples,
    /// Request lines rendered. One without a labelled question does not run.
    pub requests: usize,
    /// Record lines read.
    pub records: usize,
    /// Questions of run requests that had no label.
    pub unlabelled: usize,
}

/// A request on the worker: its line, its labelled questions with their
/// gold indices, and its answer to come.
struct Queued {
    line: usize,
    labelled: Vec<(DecisionQuestion, usize)>,
    pending: Pending,
}

/// The labelled questions of a rendered request with their gold indices,
/// and the count of questions without a label.
///
/// # Errors
///
/// `line N: 'labels.<id>' ...` for a label naming no question or a gold
/// the question does not take.
fn labelled_questions(
    request: &LabelledRequest,
    specs: &[(String, Value)],
    questions: &[DecisionQuestion],
    family: crate::model::decision_template::Family,
) -> Result<(Vec<(DecisionQuestion, usize)>, usize), String> {
    let line = request.line;
    if let Some(id) = request
        .labels
        .keys()
        .find(|id| !questions.iter().any(|q| &q.id == *id))
    {
        return Err(format!(
            "line {line}: 'labels' names '{id}', which is not a question of the request"
        ));
    }
    let mut labelled = Vec::new();
    for question in questions {
        let Some(gold) = request.labels.get(&question.id) else {
            continue;
        };
        let spec = specs
            .iter()
            .find(|(id, _)| id == &question.id)
            .map_or(&Value::Null, |(_, spec)| spec);
        let index = gold_index(family, question, spec, gold)
            .map_err(|e| format!("line {line}: 'labels.{}': {e}", question.id))?;
        labelled.push((question.clone(), index));
    }
    let unlabelled = questions.len() - labelled.len();
    Ok((labelled, unlabelled))
}

/// Wait for the oldest queued request and add its answers.
async fn drain_one(queue: &mut VecDeque<Queued>, samples: &mut TypeSamples) -> Result<(), String> {
    let Some(queued) = queue.pop_front() else {
        return Ok(());
    };
    let line = queued.line;
    let result = queued
        .pending
        .wait()
        .await
        .map_err(|e| format!("line {line}: the run failed: {e}"))?;
    for (question, gold) in &queued.labelled {
        samples
            .push_answer(question, &result.raw_logits, *gold)
            .map_err(|e| format!("line {line}: {e}"))?;
    }
    Ok(())
}

/// The labelled answers of `lines` when every line is a record, so no model
/// runs. `None` when a line is a request.
pub fn records_only(lines: &[LabelLine]) -> Option<Collected> {
    let mut out = Collected::default();
    for item in lines {
        let LabelLine::Record(record) = item else {
            return None;
        };
        out.records += 1;
        out.samples.push_record(record.clone());
    }
    Some(out)
}

/// Collect the labelled answers of `lines` on `service`.
///
/// # Errors
///
/// `line N: ...` for the first request the family refuses, whose label
/// does not resolve, that is over a limit, or whose run fails.
pub async fn collect<R: Runtime<DType = DType>>(
    service: &DecisionService<R>,
    lines: Vec<LabelLine>,
) -> Result<Collected, String> {
    let family = service.family();
    let mut out = Collected::default();
    let mut queue: VecDeque<Queued> = VecDeque::new();
    for item in lines {
        let request = match item {
            LabelLine::Record(record) => {
                out.records += 1;
                out.samples.push_record(record);
                continue;
            }
            LabelLine::Request(request) => request,
        };
        let line = request.line;
        let jev =
            JevRequest::parse(&request.body, family).map_err(|e| format!("line {line}: {e}"))?;
        let specs: Vec<(String, Value)> = jev
            .questions
            .iter()
            .map(|q| (q.id.clone(), q.spec.clone()))
            .collect();
        let rendered = service
            .renderer()
            .render(&jev)
            .map_err(|e| format!("line {line}: {e}"))?;
        let (labelled, unlabelled) =
            labelled_questions(&request, &specs, &rendered.questions, family)?;
        out.requests += 1;
        out.unlabelled += unlabelled;
        if labelled.is_empty() {
            continue;
        }
        let prepared = service
            .prepare(&rendered, true)
            .map_err(|e| format!("line {line}: {e}"))?;
        if queue.len() >= IN_FLIGHT {
            drain_one(&mut queue, &mut out.samples).await?;
        }
        let pending = loop {
            match service.submit(prepared.clone()) {
                Ok(pending) => break pending,
                Err(DecisionFailure::Busy(_)) if !queue.is_empty() => {
                    drain_one(&mut queue, &mut out.samples).await?;
                }
                Err(e) => return Err(format!("line {line}: {e}")),
            }
        };
        queue.push_back(Queued {
            line,
            labelled,
            pending,
        });
    }
    while !queue.is_empty() {
        drain_one(&mut queue, &mut out.samples).await?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::decision_template::Family;
    use boostr::inference::decision::{QuestionKind, QuestionMeta};
    use serde_json::json;

    fn noul(id: &str) -> DecisionQuestion {
        DecisionQuestion {
            id: id.into(),
            meta: QuestionMeta {
                kind: QuestionKind::Noul { yes: 1 },
                options: 2,
            },
            names: vec!["false".into(), "true".into()],
            legend: Vec::new(),
            scoring_rows: 0..1,
        }
    }

    fn request(labels: Value) -> LabelledRequest {
        LabelledRequest {
            line: 4,
            body: json!({}),
            labels: labels.as_object().cloned().unwrap_or_default(),
        }
    }

    #[test]
    fn labels_resolve_per_question_and_unlabelled_ones_are_counted() {
        let questions = [noul("a"), noul("b")];
        let specs = vec![("a".to_string(), json!({})), ("b".to_string(), json!({}))];
        let (labelled, unlabelled) = labelled_questions(
            &request(json!({"b": true})),
            &specs,
            &questions,
            Family::Decider,
        )
        .expect("resolves");
        assert_eq!(unlabelled, 1);
        assert_eq!((labelled[0].0.id.as_str(), labelled[0].1), ("b", 1));
    }

    #[test]
    fn a_label_off_the_request_or_options_names_its_line() {
        let questions = [noul("a")];
        let specs = vec![("a".to_string(), json!({}))];
        let check = |labels: Value| {
            labelled_questions(&request(labels), &specs, &questions, Family::Decider)
        };
        let e = check(json!({"z": true})).expect_err("unknown id");
        assert!(e.starts_with("line 4: 'labels' names 'z'"), "{e}");
        let e = check(json!({"a": "maybe"})).expect_err("bad gold");
        assert!(e.starts_with("line 4: 'labels.a':"), "{e}");
    }
}
