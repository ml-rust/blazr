//! Variants over a loaded base model: boostr's
//! `LoadedModel::derive_with_adapter` attaches a PEFT adapter unmerged to a
//! model that shares the base's weight storage. The base stays unmodified
//! and keeps serving.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use boostr::format::PeftAdapter;
use boostr::model::{LoadedModel, ModelClient};
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};

use super::error::LoraError;
use super::spec::AdapterSpec;
use crate::engine::Executor;

/// A variant executor over a base executor's weights.
pub struct DerivedVariant<R: Runtime<DType = DType>> {
    /// The variant's own executor: its own KV pool and caches.
    pub executor: Executor<R>,
    /// The adapter's `r`.
    pub rank: usize,
    /// How many base projections the adapter adapts.
    pub adapted_modules: usize,
}

/// A model over `base`'s weights with `spec`'s adapter, loaded onto
/// `device`, attached unmerged. Returns it with the adapter's rank.
fn derive_model<R>(
    base: &LoadedModel<R>,
    spec: &AdapterSpec,
    device: &R::Device,
) -> Result<(LoadedModel<R>, usize), LoraError>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + ScalarOps<R> + boostr::quant::DequantOps<R>,
{
    if let Some(reason) = base.adapter_refusal() {
        return Err(LoraError::Unsupported(reason));
    }
    let adapter = PeftAdapter::<R>::load(&spec.path, device).map_err(|e| {
        LoraError::Adapter(format!(
            "LoRA adapter {:?} at {} does not load: {e}",
            spec.name,
            spec.path.display()
        ))
    })?;
    let client = R::default_client(device);
    let model = base.derive_with_adapter(&client, &adapter).map_err(|e| {
        LoraError::Adapter(format!(
            "LoRA adapter {:?} at {} does not fit the model: {e}",
            spec.name,
            spec.path.display()
        ))
    })?;
    Ok((model, adapter.config.r))
}

impl<R: Runtime<DType = DType>> Executor<R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::quant::DequantOps<R>,
{
    /// A variant of this executor's model with `spec`'s adapter attached
    /// unmerged. The variant shares every weight with this model, which is
    /// not modified. It gets its own executor, so its own KV pool, prefix
    /// cache and decode sessions, with this executor's config, tokenizer,
    /// chat template, vision tower and loaded speculative draft model.
    ///
    /// # Errors
    ///
    /// [`LoraError::Unsupported`] for a model that takes no adapter (boostr
    /// names why), [`LoraError::Adapter`] when the adapter does not load or
    /// does not fit the model, [`LoraError::BaseLoad`] when the executor
    /// does not start.
    pub fn derive_variant(&self, spec: &AdapterSpec) -> Result<DerivedVariant<R>, LoraError> {
        let (model, rank) = derive_model(&self.model, spec, &self.device)?;
        let adapted_modules = model.adapted_modules().len();
        let mut executor = Executor::build(
            model,
            self.config.clone(),
            Arc::clone(&self.tokenizer),
            self.device.clone(),
            self.num_ctx,
        )
        .map_err(|e| {
            LoraError::BaseLoad(format!(
                "the executor of LoRA adapter {:?} did not start: {e:#}",
                spec.name
            ))
        })?;
        executor.chat_template = self.chat_template.clone();
        executor.model_dir = self.model_dir.clone();
        executor.vision_tower = self.vision_tower.clone();
        // A speculative draft model the base already loaded serves the
        // variant too, rather than a second copy.
        let base_draft = self.draft_model.lock().ok().and_then(|draft| draft.clone());
        if let Ok(slot) = executor.draft_model.get_mut() {
            *slot = base_draft;
        }
        tracing::info!(
            adapter = %spec.name,
            path = %spec.path.display(),
            rank,
            adapted_modules,
            "LoRA variant built"
        );
        Ok(DerivedVariant {
            executor,
            rank,
            adapted_modules,
        })
    }
}

/// The adapter `run` and `generate` decode with, from their `--lora`
/// flags: none, or one.
///
/// # Errors
///
/// A flag that does not parse, or more than one flag.
pub fn run_adapter(flags: &[String]) -> Result<Option<AdapterSpec>> {
    let mut specs = super::spec::configured_adapters(flags, None)?;
    if specs.len() > 1 {
        return Err(anyhow!(
            "run and generate decode with one model, so they take one --lora; got {}. \
             `blazr serve` serves several adapters, one variant each",
            specs.len()
        ));
    }
    Ok(specs.pop())
}

/// The model `run` and `generate` decode with: `model` unchanged without an
/// adapter, else the variant of `model` with `adapter`. The variant shares
/// `model`'s weights, and `model` drops here.
///
/// # Errors
///
/// A model that takes no adapter, or an adapter that does not load or fit.
pub fn derive_run_model<R>(
    model: LoadedModel<R>,
    adapter: Option<&AdapterSpec>,
    device: &R::Device,
) -> Result<LoadedModel<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + ScalarOps<R> + boostr::quant::DequantOps<R>,
{
    let Some(spec) = adapter else {
        return Ok(model);
    };
    let (derived, rank) = derive_model(&model, spec, device)?;
    tracing::info!(
        adapter = %spec.name,
        path = %spec.path.display(),
        rank,
        adapted_modules = derived.adapted_modules().len(),
        "LoRA adapter applied"
    );
    Ok(derived)
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::model::traits::Model;
    use boostr::{CpuDevice, CpuRuntime};

    /// A llama model with a tiny config: a model that takes no adapter.
    fn llama(device: &CpuDevice) -> LoadedModel<CpuRuntime> {
        let yaml = "model_type: llama\nvocab_size: 32\nhidden_size: 16\nnum_layers: 1\n\
                    max_seq_len: 16\nintermediate_size: 32\nattention:\n  num_heads: 2\n";
        let config: boostr::model::UniversalConfig =
            serde_saphyr::from_str(yaml).expect("config parses");
        let model =
            boostr::model::llama::Llama::<CpuRuntime>::from_config(&config, device).expect("llama");
        LoadedModel::Llama(Box::new(model))
    }

    #[test]
    fn a_model_that_takes_no_adapter_is_refused_before_the_adapter_loads() {
        let device = CpuDevice::new();
        let spec = AdapterSpec::new("kev", "/nonexistent/adapter").expect("spec");
        let err = derive_model(&llama(&device), &spec, &device).expect_err("refused");
        assert!(matches!(err, LoraError::Unsupported(_)), "{err:?}");
        assert!(err.message().contains("qwen35 models only"), "{err}");
    }

    #[test]
    fn run_takes_at_most_one_adapter() {
        assert_eq!(run_adapter(&[]).expect("none"), None);
        let one = run_adapter(&["/x/kev=k".to_string()]).expect("one");
        assert_eq!(one.map(|s| s.name).as_deref(), Some("k"));
        let two = ["/x/a".to_string(), "/x/b".to_string()];
        let err = run_adapter(&two).expect_err("refused");
        assert!(err.to_string().contains("one --lora"), "{err}");

        let device = CpuDevice::new();
        assert!(derive_run_model(llama(&device), None, &device).is_ok());
        let spec = AdapterSpec::new("kev", "/nonexistent/adapter").expect("spec");
        let err = derive_run_model(llama(&device), Some(&spec), &device).expect_err("refused");
        assert!(err.to_string().contains("qwen35 models only"), "{err}");
    }
}
