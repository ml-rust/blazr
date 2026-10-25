//! A load of one contiguous layer range, for a pipeline stage: the tensors
//! it reads and the model it builds. Every format filters by tensor name
//! before it reads, so a stage never copies another stage's weights.

use std::ops::Range;

use anyhow::{anyhow, Result};

use boostr::model::llama::LayerSpan;
use boostr::model::{LoadedModel, UniversalConfig};
use boostr::ops::TensorOps;
use boostr::{DType, Runtime, VarBuilder};

/// The span of `layers` in the model `config` describes, or `None` for a
/// load of the whole model.
///
/// # Errors
///
/// A model a layer range cannot hold (one with SSM or Gated DeltaNet
/// state), or a range outside the model. Both fail before any weight is
/// read.
pub(super) fn span_of(
    config: &UniversalConfig,
    layers: Option<Range<usize>>,
) -> Result<Option<LayerSpan>> {
    layers
        .map(|r| LayerSpan::for_model(config, r).map_err(|e| anyhow!("{e}")))
        .transpose()
}

/// Whether a load of `span` reads the tensor a Hugging Face checkpoint
/// stores as `file_name`. A load of the whole model reads every tensor.
pub(super) fn reads_file_name(
    span: Option<&LayerSpan>,
    config: &UniversalConfig,
    file_name: &str,
) -> bool {
    span.is_none_or(|s| s.reads_hf_name(config, file_name))
}

/// `(quant_bases, regular_names)` cut to the tensors a load of `span` reads.
/// A quantized base is read when its `.weight` name is.
pub(super) fn retain_read(
    span: Option<&LayerSpan>,
    config: &UniversalConfig,
    quant_bases: Vec<String>,
    regular_names: Vec<String>,
) -> (Vec<String>, Vec<String>) {
    let reads = |name: &str| reads_file_name(span, config, name);
    let quant_bases = quant_bases
        .into_iter()
        .filter(|base| reads(&format!("{base}.weight")))
        .collect();
    let regular_names = regular_names.into_iter().filter(|n| reads(n)).collect();
    (quant_bases, regular_names)
}

/// The model `vb` holds: the stage of `span`, or the whole model.
///
/// # Errors
///
/// A tensor the model reads that `vb` does not hold, or a config the model
/// refuses.
pub(super) fn build_model<R: Runtime<DType = DType>>(
    config: &UniversalConfig,
    vb: &mut VarBuilder<R>,
    span: Option<LayerSpan>,
) -> Result<LoadedModel<R>>
where
    R::Client: TensorOps<R> + boostr::quant::DequantOps<R>,
{
    let model = match span {
        Some(span) => LoadedModel::load_layers(config, vb, span),
        None => LoadedModel::load(config, vb),
    };
    model.map_err(|e| anyhow!("Failed to load model: {e}"))
}
