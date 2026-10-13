//! The readout a decision worker scores with: boostr's label-token head for
//! the label-token families, Kev's pointer head for Kev, Clef's
//! joint-schema head for Clef. One item type carries a scoring row for any
//! head, so requests of every family share one queue and one run; requests
//! with no shared prefix also share one merge and split path.

use std::path::Path;

use boostr::inference::decision::{
    run_decision, AnswerOptions, DecisionBackbone, DecisionPlan, DecisionRequest, DecisionResult,
    GatherIndexed, PrefixState, RunnerConfig,
};
use boostr::model::decision::joint_schema::joint_answer_options;
use boostr::model::decision::{
    HeadDtype, JointSchemaHead, JointSchemaItem, LabelTokenHead, LabelTokenItem, PointerHead,
    PointerItem,
};
use boostr::model::{LoadedModel, ModelClient};
use boostr::{DType, Runtime};

use super::error::DecisionFailure;

/// One scoring row as the worker's head reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadoutItem {
    /// The plan gather of the row's answer slot: the label slot, or under
    /// the pointer readout the decide token, or under the joint-schema
    /// readout the record's last position.
    pub slot: usize,
    /// Label-token readout: the label token of each option. Empty under
    /// the other readouts.
    pub labels: Vec<u32>,
    /// Pointer readout: the plan gather of each option's closing token.
    /// Empty under the other readouts.
    pub options: Vec<usize>,
    /// Joint-schema readout: the field, its spans in record positions,
    /// which are the gathers of the record's `full_sequence_plan`. `None`
    /// under the other readouts.
    pub joint: Option<JointSchemaItem>,
}

/// Merging requests moves the slot and every option gather past the
/// earlier requests' gathers. A joint-schema field never merges: its plan
/// holds the record as its prefix, and `merge_requests` refuses a plan
/// with a prefix, so its spans stay as they are.
impl GatherIndexed for ReadoutItem {
    fn offset_gathers(mut self, by: usize) -> Self {
        self.slot += by;
        for o in &mut self.options {
            *o += by;
        }
        self
    }
}

/// The head a decision worker scores with.
pub enum ReadoutHead<R: Runtime<DType = DType>> {
    /// One label token per option, read at the answer slot.
    LabelToken,
    /// Kev's pointer head and its fitted temperature.
    Pointer(Box<PointerHead<R>>),
    /// Clef's joint-schema head, in F32.
    JointSchema(Box<JointSchemaHead<R>>),
}

/// Whether `plan` reads one whole record: no rows, and gather `p` at
/// prefix position `p` for every position, as `full_sequence_plan` builds
/// it.
fn reads_one_record(plan: &DecisionPlan) -> bool {
    plan.rows.is_empty()
        && plan.gathers.len() == plan.prefix_len()
        && plan
            .gathers
            .iter()
            .enumerate()
            .all(|(p, g)| g.row.is_none() && g.pos == p)
}

fn mismatch(head: &str, item: usize) -> boostr::error::Error {
    boostr::error::Error::InvalidArgument {
        arg: "items",
        reason: format!(
            "item {item} is not a {head} item: a request rendered for another readout reached \
             this model"
        ),
    }
}

impl<R: Runtime<DType = DType>> ReadoutHead<R> {
    /// The head of a model whose renderer named `pointer_head` or
    /// `joint_head`: the label head for neither; the pointer head read from
    /// that `head.pt`; or the joint-schema head read in F32 from the
    /// directory holding `joint_head_config.json` and
    /// `joint_head.safetensors`. Each must fit `model`'s hidden size, and
    /// loads on `device`.
    ///
    /// # Errors
    ///
    /// [`DecisionFailure::NotDecisionModel`] when a head is named for a
    /// model that is not a qwen35 text model, [`DecisionFailure::Internal`]
    /// when the head does not load or does not fit the model's hidden size.
    pub fn load(
        pointer_head: Option<&Path>,
        joint_head: Option<&Path>,
        model: &LoadedModel<R>,
        device: &R::Device,
    ) -> Result<Self, DecisionFailure>
    where
        R::Client: ModelClient<R>,
    {
        if pointer_head.is_none() && joint_head.is_none() {
            return Ok(Self::LabelToken);
        }
        let LoadedModel::Qwen35(qwen) = model else {
            return Err(DecisionFailure::NotDecisionModel(
                "a pointer or joint-schema head reads a qwen35 backbone, and this model is not \
                 one"
                .into(),
            ));
        };
        let hidden = qwen.config().hidden_size;
        if let Some(dir) = joint_head {
            let client = R::default_client(device);
            let head = JointSchemaHead::from_dir(&client, dir, HeadDtype::F32).map_err(|e| {
                DecisionFailure::Internal(format!("the joint-schema head does not load: {e}"))
            })?;
            let reads = head.config().hidden_size;
            if reads != hidden {
                return Err(DecisionFailure::Internal(format!(
                    "the joint-schema head in {} reads hidden size {reads}, and the backbone's \
                     is {hidden}",
                    dir.display()
                )));
            }
            return Ok(Self::JointSchema(Box::new(head)));
        }
        let Some(path) = pointer_head else {
            return Ok(Self::LabelToken);
        };
        PointerHead::from_pt(path, hidden, device)
            .map(|head| Self::Pointer(Box::new(head)))
            .map_err(|e| {
                DecisionFailure::Internal(format!(
                    "the pointer head does not load for a backbone of hidden size {hidden}: {e}"
                ))
            })
    }

    /// Run `request` on `backbone` over `prefix` with this head.
    ///
    /// # Errors
    ///
    /// boostr's run errors, and [`boostr::error::Error::InvalidArgument`]
    /// when an item is not this head's kind.
    pub fn run<C, B>(
        &self,
        client: &C,
        backbone: &B,
        request: &DecisionRequest<ReadoutItem>,
        prefix: Option<&PrefixState<R>>,
        cfg: &RunnerConfig,
    ) -> boostr::error::Result<DecisionResult>
    where
        C: ModelClient<R>,
        B: DecisionBackbone<R>,
    {
        let opts = AnswerOptions::default();
        let (plan, meta, temp) = (&request.plan, &request.meta, &request.temp);
        match self {
            Self::LabelToken => {
                let items = request
                    .items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        if item.options.is_empty() && !item.labels.is_empty() {
                            Ok(LabelTokenItem {
                                slot: item.slot,
                                labels: item.labels.clone(),
                            })
                        } else {
                            Err(mismatch("label-token", i))
                        }
                    })
                    .collect::<boostr::error::Result<Vec<_>>>()?;
                run_decision(
                    client,
                    backbone,
                    &LabelTokenHead,
                    plan,
                    prefix,
                    &items,
                    meta,
                    temp,
                    &opts,
                    cfg,
                )
            }
            Self::Pointer(head) => {
                let items = request
                    .items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        if item.labels.is_empty() && !item.options.is_empty() {
                            Ok(PointerItem {
                                decide: item.slot,
                                options: item.options.clone(),
                            })
                        } else {
                            Err(mismatch("pointer", i))
                        }
                    })
                    .collect::<boostr::error::Result<Vec<_>>>()?;
                run_decision(
                    client, backbone, &**head, plan, prefix, &items, meta, temp, &opts, cfg,
                )
            }
            Self::JointSchema(head) => {
                if !reads_one_record(plan) {
                    return Err(boostr::error::Error::InvalidArgument {
                        arg: "plan",
                        reason: "a joint-schema request reads one whole record: its plan must \
                                 be the record's full_sequence_plan"
                            .into(),
                    });
                }
                let items = request
                    .items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        item.joint
                            .clone()
                            .ok_or_else(|| mismatch("joint-schema", i))
                    })
                    .collect::<boostr::error::Result<Vec<_>>>()?;
                run_decision(
                    client,
                    backbone,
                    &**head,
                    plan,
                    prefix,
                    &items,
                    meta,
                    temp,
                    &joint_answer_options(),
                    cfg,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging_moves_every_gather() {
        let label = ReadoutItem {
            slot: 2,
            labels: vec![7, 8],
            options: Vec::new(),
            joint: None,
        };
        assert_eq!(label.clone().offset_gathers(3).slot, 5);
        assert_eq!(label.offset_gathers(3).labels, vec![7, 8]);
        let pointer = ReadoutItem {
            slot: 0,
            labels: Vec::new(),
            options: vec![1, 2],
            joint: None,
        };
        let moved = pointer.offset_gathers(4);
        assert_eq!((moved.slot, moved.options), (4, vec![5, 6]));
    }

    #[test]
    fn a_record_plan_reads_every_position_in_order() {
        use boostr::inference::decision::Gather;
        use boostr::model::decision::joint_schema::full_sequence_plan;
        let plan = full_sequence_plan(&[5, 6, 7]).expect("plans");
        assert!(reads_one_record(&plan));
        let mut swapped = plan.clone();
        swapped.gathers.swap(0, 1);
        assert!(!reads_one_record(&swapped));
        let mut short = plan.clone();
        short.gathers.pop();
        assert!(!reads_one_record(&short));
        let mut rowed = plan;
        rowed.rows.push(vec![1]);
        rowed.gathers.push(Gather {
            row: Some(0),
            pos: 3,
        });
        assert!(!reads_one_record(&rowed));
    }
}
