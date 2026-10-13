//! A rendered decision request as the request boostr's decision runner
//! takes: the plan over its rows, one label-token item per scoring row, the
//! question metadata and the temperature policy.

use boostr::inference::decision::{plan_from_rows, DecisionRequest, RowSlot, TemperaturePolicy};
use boostr::model::decision::LabelTokenItem;

use super::error::DecisionFailure;
use crate::model::decision_template::RenderedDecision;

/// A request ready to queue.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRequest {
    /// The plan, items, questions and temperatures.
    pub request: DecisionRequest<LabelTokenItem>,
    /// Prompt rows.
    pub rows: usize,
    /// Tokens the plan runs: the shared prefix once, then every row suffix.
    pub prefill_tokens: usize,
}

impl PreparedRequest {
    /// Whether the request runs alone with its shared prefix, rather than
    /// as full rows merged with other requests.
    pub fn has_prefix(&self) -> bool {
        !self.request.plan.prefix.is_empty()
    }
}

/// The tokens of `rows` with their longest common prefix counted once.
pub fn unique_tokens(rows: &[Vec<u32>]) -> usize {
    let Some((first, rest)) = rows.split_first() else {
        return 0;
    };
    let common = rest.iter().fold(first.len(), |n, row| {
        first[..n]
            .iter()
            .zip(row)
            .take_while(|(a, b)| a == b)
            .count()
    });
    common + rows.iter().map(|r| r.len() - common).sum::<usize>()
}

/// The boostr request of `rendered` at temperatures `temp`. A shared prefix
/// of `shared_min_tokens` or more runs once; a shorter one runs as part of
/// each full row.
///
/// # Errors
///
/// [`DecisionFailure::Internal`] when the rendered rows do not plan, which
/// a renderer never produces.
pub fn prepare(
    rendered: &RenderedDecision,
    temp: TemperaturePolicy,
    shared_min_tokens: usize,
) -> Result<PreparedRequest, DecisionFailure> {
    let slots: Vec<RowSlot> = rendered.scoring_rows.iter().map(|s| s.slot).collect();
    let planned = plan_from_rows(rendered.rows.clone(), &slots, shared_min_tokens)
        .map_err(|e| DecisionFailure::Internal(format!("the rendered rows do not plan: {e}")))?;
    let items = rendered
        .scoring_rows
        .iter()
        .zip(&planned.slot_gathers)
        .map(|(row, &slot)| LabelTokenItem {
            slot,
            labels: row.labels.clone(),
        })
        .collect();
    let plan = planned.plan;
    let prefill_tokens = plan.prefix_len() + plan.rows.iter().map(Vec::len).sum::<usize>();
    Ok(PreparedRequest {
        rows: rendered.rows.len(),
        prefill_tokens,
        request: DecisionRequest {
            plan,
            items,
            meta: rendered.questions.iter().map(|q| q.meta.clone()).collect(),
            temp,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::decision_template::{DecisionQuestion, Family, ScoringRow};
    use boostr::inference::decision::{QuestionKind, QuestionMeta};

    fn rendered(rows: Vec<Vec<u32>>) -> RenderedDecision {
        let n = rows.len();
        RenderedDecision {
            family: Family::Decider,
            scoring_rows: (0..n)
                .map(|r| ScoringRow {
                    slot: RowSlot {
                        row: r,
                        pos: rows[r].len() - 1,
                    },
                    labels: vec![7, 8],
                    question: r,
                })
                .collect(),
            questions: (0..n)
                .map(|q| DecisionQuestion {
                    id: format!("q{q}"),
                    meta: QuestionMeta {
                        kind: QuestionKind::Noul { yes: 1 },
                        options: 2,
                    },
                    names: vec!["false".into(), "true".into()],
                    legend: Vec::new(),
                    scoring_rows: q..q + 1,
                })
                .collect(),
            rows,
            warnings: Vec::new(),
            state_tokens: 3,
        }
    }

    #[test]
    fn unique_tokens_count_the_shared_prefix_once() {
        assert_eq!(unique_tokens(&[]), 0);
        assert_eq!(unique_tokens(&[vec![1, 2, 3]]), 3);
        assert_eq!(unique_tokens(&[vec![1, 2, 3], vec![1, 2, 4, 5]]), 2 + 1 + 2);
    }

    #[test]
    fn short_prefixes_run_as_full_rows_and_long_ones_once() {
        let r = rendered(vec![vec![1, 2, 3, 4, 5], vec![1, 2, 3, 6, 7]]);
        let full = prepare(&r, TemperaturePolicy::Fixed(1.0), 4).expect("plans");
        assert!(!full.has_prefix());
        assert_eq!((full.rows, full.prefill_tokens), (2, 10));
        let shared = prepare(&r, TemperaturePolicy::Fixed(1.0), 3).expect("plans");
        assert!(shared.has_prefix());
        assert_eq!(shared.prefill_tokens, 3 + 2 + 2);
        assert_eq!(shared.request.items.len(), 2);
        assert_eq!(shared.request.items[1].labels, vec![7, 8]);
        assert_eq!(shared.request.meta.len(), 2);
    }
}
