//! The SafeTensors load shared by the decomposed INT4 formats, AWQ and GPTQ.
//!
//! Each format names its tensor suffixes, its group-size source and its
//! per-layer load. This module reads the config, classifies the tensors,
//! loads the regular ones, stores each quantized layer as a
//! `DecomposedQuantTensor`, and builds the model.

use std::ops::Range;
use std::path::Path;

use anyhow::{anyhow, Result};

use boostr::format::SafeTensorsLoader;
use boostr::model::LoadedModel;
use boostr::ops::TensorOps;
use boostr::quant::decomposed::DecomposedQuantTensor;
use boostr::Tensor;
use boostr::VarBuilder;
use boostr::VarMap;
use boostr::{DType, Runtime};

use crate::config::{fixed_format_dtype, BlazrConfig, DtypeRequest};

use super::super::layer_scope::{build_model, retain_read, span_of};
use super::config::load_or_create_config;
use super::detect_arch::detect_architecture_from_loader;

/// One decomposed INT4 format: its name, its tensors and its layer load.
pub(super) trait DecomposedFormat {
    /// The format name in logs and in the dtype notice.
    const NAME: &'static str;
    /// Suffixes of the tensors a quantized layer holds beside `.qweight`.
    const PART_SUFFIXES: &'static [&'static str];
    /// The `.bias` of a quantized layer belongs to it, not to the regular tensors.
    const LAYER_BIAS: bool;

    /// The quantization group size the model directory states.
    fn group_size(model_dir: &Path) -> Result<usize>;

    /// Load the quantized layer `base_name` and its bias, if it has one.
    fn load_layer<R: Runtime<DType = DType>>(
        loader: &mut SafeTensorsLoader,
        base_name: &str,
        group_size: usize,
        device: &R::Device,
    ) -> Result<(DecomposedQuantTensor<R>, Option<Tensor<R>>)>
    where
        R::Client: TensorOps<R>;
}

/// Load a model of format `F` from SafeTensors: the whole model, or layers
/// `layers` as a pipeline stage holds them. A range load reads only the
/// tensors of its span: see `LayerSpan::reads`.
pub(super) fn load_decomposed_scoped<F: DecomposedFormat, R: Runtime<DType = DType>>(
    path: &Path,
    device: &R::Device,
    request: Option<&DtypeRequest>,
    layers: Option<Range<usize>>,
) -> Result<(LoadedModel<R>, BlazrConfig)>
where
    R::Client: TensorOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    let config_dir = if path.is_file() {
        path.parent()
    } else {
        Some(path)
    };

    let model_dir = config_dir.ok_or_else(|| anyhow!("Cannot determine model directory"))?;

    let mut loader =
        SafeTensorsLoader::open(path).map_err(|e| anyhow!("Failed to open SafeTensors: {}", e))?;

    let group_size = F::group_size(model_dir)?;

    tracing::info!(
        "Loading {} INT4 model (group_size={}), size: {:.2} GB",
        F::NAME,
        group_size,
        loader.total_size() as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // The model runs in F16: its scales are F16.
    let dtype = fixed_format_dtype(F::NAME, DType::F16, request);
    let config = load_or_create_config(
        model_dir,
        || detect_architecture_from_loader(&loader),
        dtype,
    )?;

    // Separate tensor names into quantized layers and regular tensors
    let tensor_names = loader.tensor_names();
    let (quant_bases, regular_names) =
        classify_tensors(&tensor_names, F::PART_SUFFIXES, F::LAYER_BIAS);
    let span = span_of(&config.model, layers)?;
    let (quant_bases, regular_names) =
        retain_read(span.as_ref(), &config.model, quant_bases, regular_names);

    let mut var_map = VarMap::<R>::new();

    // Load regular tensors (embeddings, norms, lm_head)
    let regular_count = regular_names.len();
    for (idx, name) in regular_names.iter().enumerate() {
        if idx % 20 == 0 {
            tracing::debug!(
                "Loading regular tensor {}/{}: {}",
                idx + 1,
                regular_count,
                name
            );
        }
        let tensor = loader
            .load_tensor::<R>(name, device)
            .map_err(|e| anyhow!("Failed to load tensor '{}': {}", name, e))?;
        // The model is calibrated for F16 inference. BF16 embeddings and norms
        // cast to F16: BF16 has 3 fewer mantissa bits, and the error
        // accumulates across the layers.
        let tensor = if tensor.dtype() == DType::BF16 {
            let client = R::default_client(device);
            boostr::TypeConversionOps::cast(&client, &tensor, DType::F16)
                .map_err(|e| anyhow!("Failed to cast BF16→F16: {}", e))?
        } else {
            tensor
        };
        var_map.insert(name.clone(), tensor);
    }
    tracing::info!("Loaded {} regular tensors", regular_count);

    // Load quantized layers as DecomposedQuantTensor
    let quant_count = quant_bases.len();
    for (idx, base_name) in quant_bases.iter().enumerate() {
        if idx % 10 == 0 {
            tracing::debug!(
                "Loading {} layer {}/{}: {}",
                F::NAME,
                idx + 1,
                quant_count,
                base_name
            );
        }

        let (dqt, bias) = F::load_layer::<R>(&mut loader, base_name, group_size, device)?;

        // Store as "base_name.weight" so VarBuilder finds it with standard naming
        let weight_name = format!("{}.weight", base_name);
        var_map.insert_decomposed_quant(weight_name, dqt);

        if let Some(bias_tensor) = bias {
            let bias_name = format!("{}.bias", base_name);
            var_map.insert(bias_name, bias_tensor);
        }
    }
    tracing::info!(
        "Loaded {} {} quantized layers (group_size={})",
        quant_count,
        F::NAME,
        group_size
    );

    let mut vb = VarBuilder::new(&mut var_map, device);

    let model = build_model(&config.model, &mut vb, span)?;

    Ok((model, config))
}

/// Classify tensor names into quantized layer bases and regular tensor names.
///
/// A quantized layer has a `.qweight` tensor and one tensor per suffix of
/// `part_suffixes`. With `layer_bias`, the `.bias` of a layer that has a
/// `.qweight` belongs to that layer. Returns (unique_base_names,
/// regular_tensor_names).
fn classify_tensors(
    names: &[String],
    part_suffixes: &[&str],
    layer_bias: bool,
) -> (Vec<String>, Vec<String>) {
    let mut quant_bases = std::collections::BTreeSet::new();
    let mut regular = Vec::new();

    for name in names {
        if let Some(base) = name.strip_suffix(".qweight") {
            quant_bases.insert(base.to_string());
            continue;
        }
        if part_suffixes.iter().any(|suffix| name.ends_with(suffix)) {
            // Part of a quantized layer
            continue;
        }
        if layer_bias {
            if let Some(base) = name.strip_suffix(".bias") {
                let qw_name = format!("{}.qweight", base);
                if names.iter().any(|n| n == &qw_name) {
                    // The quantized layer loads its own bias
                    continue;
                }
            }
        }
        regular.push(name.clone());
    }

    (quant_bases.into_iter().collect(), regular)
}

/// Cast F16 raw bytes to a F32 tensor
pub(super) fn cast_f16_bytes_to_f32<R: Runtime<DType = DType>>(
    bytes: &[u8],
    shape: &[usize],
    device: &R::Device,
) -> Result<Tensor<R>> {
    let f16_data: &[half::f16] = bytemuck::cast_slice(bytes);
    let f32_data: Vec<f32> = f16_data.iter().map(|v| v.to_f32()).collect();
    Ok(Tensor::<R>::from_slice(&f32_data, shape, device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn without_layer_bias_every_bias_and_g_idx_is_regular() {
        let all = names(&[
            "model.embed_tokens.weight",
            "layers.0.q.qweight",
            "layers.0.q.qzeros",
            "layers.0.q.scales",
            "layers.0.q.bias",
            "layers.0.q.g_idx",
        ]);
        let (quant, regular) = classify_tensors(&all, &[".qzeros", ".scales"], false);
        assert_eq!(quant, names(&["layers.0.q"]));
        assert_eq!(
            regular,
            names(&[
                "model.embed_tokens.weight",
                "layers.0.q.bias",
                "layers.0.q.g_idx"
            ])
        );
    }

    #[test]
    fn gptq_layer_bias_stays_with_its_layer() {
        let all = names(&[
            "layers.0.q.qweight",
            "layers.0.q.qzeros",
            "layers.0.q.scales",
            "layers.0.q.g_idx",
            "layers.0.q.bias",
            "layers.0.norm.bias",
        ]);
        let (quant, regular) = classify_tensors(&all, &[".qzeros", ".scales", ".g_idx"], true);
        assert_eq!(quant, names(&["layers.0.q"]));
        assert_eq!(regular, names(&["layers.0.norm.bias"]));
    }
}
