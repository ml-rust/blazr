//! A rendered decision request as the request boostr's decision runner
//! takes: the plan over its rows, one readout item per scoring row, the
//! question metadata and the temperature policy.
//!
//! Under the label-token readout a scoring row reads one gather, its
//! answer slot. Under the pointer readout it reads its decide slot and
//! each option's closing token, every one a gather of the plan. Under the
//! joint-schema readout the one row is the whole record: its plan is
//! `full_sequence_plan`, a gather per position, and each field reads its
//! spans of it.

use boostr::inference::decision::request::plan_from_rows_at;
use boostr::inference::decision::{plan_from_rows, DecisionRequest, RowSlot, TemperaturePolicy};
use boostr::model::decision::joint_schema::full_sequence_plan;

use super::error::DecisionFailure;
use super::readout::ReadoutItem;
use crate::model::decision_template::clef::JointLayout;
use crate::model::decision_template::{Readout, RenderedDecision};

/// Where a request's shared prefix ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixRule {
    /// Shortest prefix that runs once. A shorter one runs as part of each
    /// full row.
    pub min_tokens: usize,
    /// Cut the prefix at the family's state boundary
    /// ([`RenderedDecision::state_end`]), a single row included, so every
    /// request about one state runs over the same prefix and a prefix cache
    /// can keep it. Off: the prefix is the longest one the rows share.
    pub at_state: bool,
}

/// A request ready to queue.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedRequest {
    /// The plan, items, questions and temperatures.
    pub request: DecisionRequest<ReadoutItem>,
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

/// The boostr request of a joint-schema record: its one row read whole,
/// one item per field, at temperature 1, the head's plain softmax.
fn prepare_joint(
    rendered: &RenderedDecision,
    layout: &JointLayout,
) -> Result<PreparedRequest, DecisionFailure> {
    let [record] = rendered.rows.as_slice() else {
        return Err(DecisionFailure::Internal(format!(
            "a joint-schema record renders one row, got {}",
            rendered.rows.len()
        )));
    };
    if layout.fields.len() != rendered.questions.len() {
        return Err(DecisionFailure::Internal(format!(
            "the record has {} fields for {} questions",
            layout.fields.len(),
            rendered.questions.len()
        )));
    }
    let plan = full_sequence_plan(record)
        .map_err(|e| DecisionFailure::Internal(format!("the record does not plan: {e}")))?;
    let last = record.len().saturating_sub(1);
    let items = layout
        .fields
        .iter()
        .map(|field| ReadoutItem {
            slot: last,
            labels: Vec::new(),
            options: Vec::new(),
            joint: Some(field.item.clone()),
        })
        .collect();
    Ok(PreparedRequest {
        rows: 1,
        prefill_tokens: record.len(),
        request: DecisionRequest {
            plan,
            items,
            meta: rendered.questions.iter().map(|q| q.meta.clone()).collect(),
            temp: TemperaturePolicy::Fixed(1.0),
        },
    })
}

/// The boostr request of `rendered` at temperatures `temp`. A shared prefix
/// of `prefix.min_tokens` or more runs once, cut at the state boundary when
/// `prefix.at_state` says so; a shorter one runs as part of each full row.
/// A joint-schema record runs whole, alone, at temperature 1 whatever
/// `temp` says.
///
/// # Errors
///
/// [`DecisionFailure::Internal`] when the rendered rows do not plan, or a
/// joint-schema record lacks its layout, which a renderer never produces.
pub fn prepare(
    rendered: &RenderedDecision,
    temp: TemperaturePolicy,
    prefix: PrefixRule,
) -> Result<PreparedRequest, DecisionFailure> {
    if rendered.family.readout() == Readout::JointSchema {
        let layout = rendered.joint.as_ref().ok_or_else(|| {
            DecisionFailure::Internal(
                "a joint-schema record was rendered without its fields".into(),
            )
        })?;
        return prepare_joint(rendered, layout);
    }
    let pointer = rendered.family.readout() == Readout::Pointer;
    // Each scoring row's slots: its answer slot, then under the pointer
    // readout its option ends.
    let mut slots: Vec<RowSlot> = Vec::with_capacity(rendered.scoring_rows.len());
    for row in &rendered.scoring_rows {
        slots.push(row.slot);
        if pointer {
            let ends = row.option_ends.iter().map(|&pos| RowSlot {
                row: row.slot.row,
                pos,
            });
            slots.extend(ends);
        }
    }
    let state = rendered.state_end;
    let planned = if prefix.at_state && state > 0 && state >= prefix.min_tokens {
        plan_from_rows_at(rendered.rows.clone(), &slots, state)
    } else {
        plan_from_rows(rendered.rows.clone(), &slots, prefix.min_tokens)
    }
    .map_err(|e| DecisionFailure::Internal(format!("the rendered rows do not plan: {e}")))?;
    let mut gathers = planned.slot_gathers.iter().copied();
    let mut items = Vec::with_capacity(rendered.scoring_rows.len());
    for row in &rendered.scoring_rows {
        let slot = gathers.next().ok_or_else(|| {
            DecisionFailure::Internal("the plan has fewer gathers than slots".into())
        })?;
        let item = if pointer {
            let options: Vec<usize> = gathers.by_ref().take(row.option_ends.len()).collect();
            if options.len() != row.option_ends.len() {
                return Err(DecisionFailure::Internal(
                    "the plan has fewer gathers than slots".into(),
                ));
            }
            ReadoutItem {
                slot,
                labels: Vec::new(),
                options,
                joint: None,
            }
        } else {
            ReadoutItem {
                slot,
                labels: row.labels.clone(),
                options: Vec::new(),
                joint: None,
            }
        };
        items.push(item);
    }
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
                    option_ends: Vec::new(),
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
            state_end: 2,
            joint: None,
        }
    }

    /// The prefix the rows share, run once from `min` tokens.
    fn lcp(min: usize) -> PrefixRule {
        PrefixRule {
            min_tokens: min,
            at_state: false,
        }
    }

    /// With a prefix cache the prefix ends at the state, whatever the rows
    /// share past it, and a single row takes it too; a state shorter than
    /// the minimum keeps the shared-prefix rule.
    #[test]
    fn a_cached_prefix_ends_at_the_state() {
        let r = rendered(vec![vec![1, 2, 3, 4, 5], vec![1, 2, 3, 6, 7]]);
        let at_state = |min| PrefixRule {
            min_tokens: min,
            at_state: true,
        };
        let cut = prepare(&r, TemperaturePolicy::Fixed(1.0), at_state(2)).expect("plans");
        assert_eq!(cut.request.plan.prefix.tokens(), &[1, 2]);
        assert_eq!(cut.prefill_tokens, 2 + 3 + 3);
        let shared = prepare(&r, TemperaturePolicy::Fixed(1.0), lcp(2)).expect("plans");
        assert_eq!(shared.request.plan.prefix_len(), 3);
        let short = prepare(&r, TemperaturePolicy::Fixed(1.0), at_state(3)).expect("plans");
        assert_eq!(short.request.plan.prefix_len(), 3, "the state is under 3");
        let one = rendered(vec![vec![1, 2, 3, 4]]);
        let alone = prepare(&one, TemperaturePolicy::Fixed(1.0), at_state(1)).expect("plans");
        assert_eq!(alone.request.plan.prefix.tokens(), &[1, 2]);
        assert_eq!(alone.request.plan.rows, vec![vec![3, 4]]);
        let none = prepare(&one, TemperaturePolicy::Fixed(1.0), lcp(1)).expect("plans");
        assert!(!none.has_prefix());
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
        let full = prepare(&r, TemperaturePolicy::Fixed(1.0), lcp(4)).expect("plans");
        assert!(!full.has_prefix());
        assert_eq!((full.rows, full.prefill_tokens), (2, 10));
        let shared = prepare(&r, TemperaturePolicy::Fixed(1.0), lcp(3)).expect("plans");
        assert!(shared.has_prefix());
        assert_eq!(shared.prefill_tokens, 3 + 2 + 2);
        assert_eq!(shared.request.items.len(), 2);
        assert_eq!(shared.request.items[1].labels, vec![7, 8]);
        assert_eq!(shared.request.meta.len(), 2);
    }

    /// A pointer row reads its decide slot and each option end, every one a
    /// gather of its own, in that order.
    #[test]
    fn pointer_rows_gather_decide_then_option_ends() {
        let mut r = rendered(vec![vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3, 7, 8, 9, 10]]);
        r.family = Family::Kev;
        for row in &mut r.scoring_rows {
            row.labels.clear();
        }
        r.scoring_rows[0].option_ends = vec![3, 4];
        r.scoring_rows[1].option_ends = vec![4, 5];
        let full = prepare(&r, TemperaturePolicy::Fixed(2.0), lcp(100)).expect("plans");
        let items = &full.request.items;
        assert_eq!(full.request.plan.gathers.len(), 6);
        assert_eq!((items[0].slot, items[0].options.clone()), (0, vec![1, 2]));
        assert_eq!((items[1].slot, items[1].options.clone()), (3, vec![4, 5]));
        assert!(items.iter().all(|i| i.labels.is_empty()));
        let shared = prepare(&r, TemperaturePolicy::Fixed(2.0), lcp(3)).expect("plans");
        assert!(shared.has_prefix(), "the state runs once");
        assert_eq!(shared.request.plan.prefix_len(), 3);
    }

    /// A joint-schema record runs whole and alone: every position a
    /// gather, one item per field carrying its spans, temperature 1.
    #[test]
    fn a_joint_record_reads_every_position() {
        use crate::model::decision_template::clef::JointField;
        use boostr::model::decision::{FieldKind, JointSchemaItem};
        let record: Vec<u32> = (10..22).collect();
        let item = JointSchemaItem::from_tokens(FieldKind::Noul, 2..4, &[5..7, 8..10], &record)
            .expect("an item");
        let mut r = rendered(vec![record.clone()]);
        r.family = Family::Clef;
        r.questions[0].meta = item.meta();
        assert!(
            prepare(&r, TemperaturePolicy::Fixed(2.0), lcp(3)).is_err(),
            "a record needs its layout"
        );
        r.joint = Some(JointLayout {
            fields: vec![JointField {
                item: item.clone(),
                wire_order: vec![0, 1],
                legend: Vec::new(),
            }],
        });
        let p = prepare(&r, TemperaturePolicy::Fixed(2.0), lcp(3)).expect("plans");
        assert!(
            p.has_prefix(),
            "the record is the prefix, so it never merges"
        );
        assert_eq!((p.rows, p.prefill_tokens), (1, record.len()));
        assert!(p.request.plan.rows.is_empty());
        assert_eq!(p.request.plan.gathers.len(), record.len());
        assert_eq!(p.request.items.len(), 1);
        assert_eq!(p.request.items[0].joint.as_ref(), Some(&item));
        assert_eq!(p.request.items[0].slot, record.len() - 1);
        assert_eq!(p.request.temp, TemperaturePolicy::Fixed(1.0));
    }
}
