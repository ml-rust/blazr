//! The readout a decision worker scores with: boostr's label-token head for
//! the label-token families, Kev's pointer head for Kev. One item type
//! carries a scoring row for either head, so requests of every family share
//! one queue, one merge and split path, and one run.

use std::path::Path;

use boostr::inference::decision::{
    run_decision, AnswerOptions, DecisionBackbone, DecisionRequest, DecisionResult, GatherIndexed,
    PrefixState, RunnerConfig,
};
use boostr::model::decision::{LabelTokenHead, LabelTokenItem, PointerHead, PointerItem};
use boostr::model::{LoadedModel, ModelClient};
use boostr::{DType, Runtime};

use super::error::DecisionFailure;

/// One scoring row as the worker's head reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadoutItem {
    /// The plan gather of the row's answer slot: the label slot, or under
    /// the pointer readout the decide token.
    pub slot: usize,
    /// Label-token readout: the label token of each option. Empty under
    /// the pointer readout.
    pub labels: Vec<u32>,
    /// Pointer readout: the plan gather of each option's closing token.
    /// Empty under the label-token readout.
    pub options: Vec<usize>,
}

/// Merging requests moves the slot and every option gather past the
/// earlier requests' gathers.
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
    /// The head of a model whose renderer named `pointer_head`: the label
    /// head for `None`, else the pointer head read from that `head.pt` for
    /// `model`'s hidden size, on `device`.
    ///
    /// # Errors
    ///
    /// [`DecisionFailure::NotDecisionModel`] when a pointer head is named
    /// for a model that is not a qwen35 text model,
    /// [`DecisionFailure::Internal`] when `head.pt` does not load or does
    /// not fit the model's hidden size.
    pub fn load(
        pointer_head: Option<&Path>,
        model: &LoadedModel<R>,
        device: &R::Device,
    ) -> Result<Self, DecisionFailure>
    where
        R::Client: ModelClient<R>,
    {
        let Some(path) = pointer_head else {
            return Ok(Self::LabelToken);
        };
        let LoadedModel::Qwen35(qwen) = model else {
            return Err(DecisionFailure::NotDecisionModel(
                "a pointer head reads a qwen35 backbone, and this model is not one".into(),
            ));
        };
        let hidden = qwen.config().hidden_size;
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
        };
        assert_eq!(label.clone().offset_gathers(3).slot, 5);
        assert_eq!(label.offset_gathers(3).labels, vec![7, 8]);
        let pointer = ReadoutItem {
            slot: 0,
            labels: Vec::new(),
            options: vec![1, 2],
        };
        let moved = pointer.offset_gathers(4);
        assert_eq!((moved.slot, moved.options), (4, vec![5, 6]));
    }
}
