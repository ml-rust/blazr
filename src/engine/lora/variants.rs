//! The LoRA variant side of [`Scheduler`]: register, build, select, list
//! and remove variants.
//!
//! A variant is built from its loaded base with
//! [`Executor::derive_variant`], never by a second load of the base
//! weights. It is kept in the scheduler's loaded models under its id, with
//! its base recorded, so eviction and unload treat it as a model of its own
//! that never outlives its base.

use std::sync::Arc;
use std::time::Instant;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, UnaryOps,
};

use super::error::LoraError;
use super::registry::VariantInfo;
use super::select::{not_registered, select_model};
use super::spec::{split_variant, AdapterSpec};
use crate::engine::scheduler::ModelEntry;
use crate::engine::{Executor, Scheduler};

/// Variants built over a loaded base: at registration, and again after an
/// eviction or unload.
const VARIANT_BUILDS: &str = "blazr_lora_variant_builds_total";

impl<R: Runtime<DType = DType>> Scheduler<R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + boostr::SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    /// The model name a request for `model` with `lora_adapter` set to
    /// `requested` runs on. `default` is the live
    /// `generation.lora_adapter`. See [`select_model`] for the rules.
    ///
    /// # Errors
    ///
    /// [`LoraError::Invalid`] when `model` and `requested` name different
    /// adapters, [`LoraError::AdapterNotFound`] for an adapter that is not
    /// registered for `model`.
    pub fn select_model(
        &self,
        model: &str,
        requested: Option<&str>,
        default: Option<&str>,
    ) -> Result<String, LoraError> {
        select_model(model, requested, default, |id| self.lora.adapter_of(id))
    }

    /// Register `spec` as the variant `<base>:<name>` and build it from
    /// `base`, loading `base` first when needed. A variant of the same id
    /// is replaced.
    ///
    /// # Errors
    ///
    /// [`LoraError::Unsupported`] when `base` is a variant itself or takes
    /// no adapter, [`LoraError::ModelNotFound`] or
    /// [`LoraError::BaseLoad`] when `base` does not load,
    /// [`LoraError::Adapter`] when the adapter does not load or fit.
    pub async fn register_variant(
        &self,
        base: &str,
        spec: AdapterSpec,
    ) -> Result<VariantInfo, LoraError> {
        if let Some(variant) = self.lora.get(base) {
            return Err(LoraError::Unsupported(format!(
                "model {base:?} is the variant of LoRA adapter {:?} on model {:?}, and adapters \
                 do not stack. Register the adapter on {:?}",
                variant.adapter, variant.base, variant.base
            )));
        }
        let base_executor = self.base_executor(base).await?;
        let derived = base_executor.derive_variant(&spec)?;
        let info = VariantInfo {
            id: spec.variant_id(base),
            base: base.to_string(),
            adapter: spec.name,
            path: spec.path,
            rank: derived.rank,
            adapted_modules: derived.adapted_modules,
        };
        self.lora.insert(info.clone());
        self.insert_variant(&info.id, base, Arc::new(derived.executor), true)
            .await;
        metrics::counter!(VARIANT_BUILDS).increment(1);
        tracing::info!(variant = %info.id, path = %info.path.display(), "LoRA variant registered");
        Ok(info)
    }

    /// Unregister the variant `id` and drop its executor. `None` when no
    /// such variant is registered.
    pub async fn remove_variant(&self, id: &str) -> Option<VariantInfo> {
        let info = self.lora.remove(id)?;
        self.models.write().await.remove(id);
        tracing::info!(variant = id, "LoRA variant removed");
        Some(info)
    }

    /// The registered variant `id`.
    pub fn variant(&self, id: &str) -> Option<VariantInfo> {
        self.lora.get(id)
    }

    /// The registered variants of `base`, or of every model for `None`,
    /// by id.
    pub fn variants(&self, base: Option<&str>) -> Vec<VariantInfo> {
        match base {
            Some(base) => self.lora.of_base(base),
            None => self.lora.all(),
        }
    }

    /// Whether `name`, a model or a variant id, is loaded.
    pub async fn is_loaded(&self, name: &str) -> bool {
        self.models.read().await.contains_key(name)
    }

    /// The executor of the base model `name`, loading it when needed.
    async fn base_executor(&self, name: &str) -> Result<Arc<Executor<R>>, LoraError> {
        if let Some(executor) = self.touch(name).await {
            return Ok(executor);
        }
        if self.model_path(name).is_none() {
            return Err(LoraError::ModelNotFound(format!(
                "model {name:?} not found: it is neither a path nor an entry of {}",
                self.model_dir().display()
            )));
        }
        self.load_model(name)
            .await
            .map_err(|e| LoraError::BaseLoad(format!("model {name:?} did not load: {e:#}")))
    }

    /// Build the registered variant `info` from its base, loading the base
    /// when needed: after an eviction or an unload of the variant or its
    /// base.
    pub(crate) async fn load_variant(
        &self,
        info: &VariantInfo,
    ) -> Result<Arc<Executor<R>>, LoraError> {
        let base = self.base_executor(&info.base).await?;
        let spec = AdapterSpec {
            name: info.adapter.clone(),
            path: info.path.clone(),
        };
        let derived = base.derive_variant(&spec)?;
        metrics::counter!(VARIANT_BUILDS).increment(1);
        Ok(self
            .insert_variant(&info.id, &info.base, Arc::new(derived.executor), false)
            .await)
    }

    /// The error for `name` when it has the shape `<model>:<adapter>`, is
    /// no model of its own, and names a known model: one that is loaded,
    /// has a registered variant, or resolves to a path. `None` otherwise.
    pub(crate) async fn unknown_variant(&self, name: &str) -> Option<LoraError> {
        if self.model_path(name).is_some() {
            return None;
        }
        let (base, adapter) = split_variant(name)?;
        let known = self.lora.has_variants(base)
            || self.model_path(base).is_some()
            || self.is_loaded(base).await;
        known.then(|| not_registered(base, adapter))
    }

    /// Keep `executor` loaded as the variant `id` of `base`, and return the
    /// executor that serves `id`. With `replace` false, a variant loaded
    /// meanwhile stays and is returned. When `base` is no longer loaded,
    /// nothing is kept, since a variant never outlives its base: the
    /// executor serves the request that built it and drops.
    async fn insert_variant(
        &self,
        id: &str,
        base: &str,
        executor: Arc<Executor<R>>,
        replace: bool,
    ) -> Arc<Executor<R>> {
        let mut models = self.models.write().await;
        let Some(base_entry) = models.get(base) else {
            return executor;
        };
        let path = base_entry.path.clone();
        let keep_alive = base_entry.keep_alive;
        if !replace {
            if let Some(existing) = models.get(id) {
                return Arc::clone(&existing.executor);
            }
        }
        models.insert(
            id.to_string(),
            ModelEntry {
                executor: Arc::clone(&executor),
                last_accessed: Instant::now(),
                path,
                keep_alive,
                variant_of: Some(base.to_string()),
            },
        );
        executor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::lora::variant_id;
    use boostr::{CpuDevice, CpuRuntime};

    fn scheduler(dir: &std::path::Path) -> Scheduler<CpuRuntime> {
        Scheduler::new(dir.to_path_buf(), CpuDevice::new())
    }

    fn registered(base: &str, adapter: &str) -> VariantInfo {
        VariantInfo {
            id: variant_id(base, adapter),
            base: base.to_string(),
            adapter: adapter.to_string(),
            path: format!("/adapters/{adapter}").into(),
            rank: 8,
            adapted_modules: 4,
        }
    }

    #[tokio::test]
    async fn an_unknown_adapter_of_a_known_model_is_not_found() {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::create_dir(dir.path().join("m")).expect("mkdir");
        std::fs::write(dir.path().join("m/config.json"), "{}").expect("write");
        let scheduler = scheduler(dir.path());

        let err = scheduler
            .get_executor("m:nope")
            .await
            .err()
            .expect("refused");
        let lora = err.downcast_ref::<LoraError>().expect("a LoraError");
        assert!(matches!(lora, LoraError::AdapterNotFound(_)), "{lora:?}");

        // No model of that name: the plain load error, not an adapter error.
        let err = scheduler
            .get_executor("absent:kev")
            .await
            .err()
            .expect("refused");
        assert!(err.downcast_ref::<LoraError>().is_none(), "{err:#}");
        assert!(err.to_string().contains("Model not found"), "{err:#}");
    }

    /// Windows file names hold no `:`.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_directory_named_with_a_colon_stays_a_model() {
        let dir = tempfile::tempdir().expect("temp dir");
        let scheduler = scheduler(dir.path());
        std::fs::create_dir(dir.path().join("m:kev")).expect("mkdir");
        assert!(scheduler.unknown_variant("m:kev").await.is_none());
    }

    #[tokio::test]
    async fn registration_drives_selection_and_removal() {
        let dir = tempfile::tempdir().expect("temp dir");
        let scheduler = scheduler(dir.path());
        scheduler.lora.insert(registered("m", "kev"));

        assert_eq!(
            scheduler.select_model("m", Some("kev"), None).unwrap(),
            "m:kev"
        );
        let err = scheduler
            .select_model("m:kev", Some("jev"), None)
            .unwrap_err();
        assert!(matches!(err, LoraError::Invalid(_)), "{err:?}");
        let err = scheduler.select_model("m", Some("jev"), None).unwrap_err();
        assert!(matches!(err, LoraError::AdapterNotFound(_)), "{err:?}");
        assert!(scheduler.unknown_variant("m:jev").await.is_some());

        assert_eq!(scheduler.variants(Some("m")).len(), 1);
        assert!(scheduler.variants(Some("n")).is_empty());
        assert!(!scheduler.is_loaded("m:kev").await);
        assert_eq!(
            scheduler.remove_variant("m:kev").await.map(|v| v.rank),
            Some(8)
        );
        assert!(scheduler.remove_variant("m:kev").await.is_none());
        assert!(scheduler.variants(None).is_empty());
    }

    #[tokio::test]
    async fn a_variant_is_never_a_base() {
        let dir = tempfile::tempdir().expect("temp dir");
        let scheduler = scheduler(dir.path());
        scheduler.lora.insert(registered("m", "kev"));
        let spec = AdapterSpec::new("jev", "/adapters/jev").expect("spec");
        let err = scheduler.register_variant("m:kev", spec).await.unwrap_err();
        assert!(matches!(err, LoraError::Unsupported(_)), "{err:?}");
        assert!(err.message().contains("do not stack"), "{err}");

        let spec = AdapterSpec::new("jev", "/adapters/jev").expect("spec");
        let err = scheduler
            .register_variant("absent", spec)
            .await
            .unwrap_err();
        assert!(matches!(err, LoraError::ModelNotFound(_)), "{err:?}");
    }
}
