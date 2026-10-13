//! Kev models on the scheduler. A Kev directory holds an adapter and a
//! pointer head, no weights: its name loads as a LoRA variant of the base
//! it resolves to, registered under the Kev directory's own name and built
//! from the loaded base with `derive_with_adapter`. The base loads once
//! and keeps serving under its own name, and eviction treats the Kev model
//! as any variant of it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, UnaryOps,
};

use super::kev_base::{hub_caches, resolve_kev_base};
use crate::engine::lora::spec::check_name;
use crate::engine::{AdapterSpec, Executor, LoraError, Scheduler};
use crate::loader::decision::is_kev_dir;

/// The adapter name of a Kev variant whose directory name is no valid
/// adapter name.
const KEV_ADAPTER: &str = "kev";

/// The directory of an adapter path: the path itself, or the directory
/// holding an adapter `.safetensors` file.
fn adapter_dir(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        Some(path.to_path_buf())
    } else {
        path.parent().map(Path::to_path_buf)
    }
}

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
    /// The Kev directory `name` serves: the adapter of a registered
    /// variant when it is a Kev directory, else `name`'s own path when it
    /// is one. `None` for any other model.
    pub(crate) fn kev_dir(&self, name: &str) -> Option<PathBuf> {
        match self.variant(name) {
            Some(v) => adapter_dir(&v.path).filter(|d| is_kev_dir(d)),
            None => self.model_path(name).filter(|p| is_kev_dir(p)),
        }
    }

    /// The base checkpoint the Kev model `name` at `kev_dir` runs on: its
    /// registered variant's base, else the base it resolves to. `None`
    /// when it resolves to none.
    pub(crate) fn kev_base_path(&self, name: &str, kev_dir: &Path) -> Option<PathBuf> {
        match self.variant(name) {
            Some(v) => self.model_path(&v.base),
            None => {
                let configured = self.decision_config.base_for(name);
                resolve_kev_base(kev_dir, configured, self.model_dir(), &hub_caches())
                    .ok()
                    .map(|b| b.path)
            }
        }
    }

    /// Build the Kev model `name` over its base, loading the base first
    /// when needed, and register it as the variant `name`. `None` when
    /// `name` is not a Kev directory.
    ///
    /// # Errors
    ///
    /// [`LoraError::ModelNotFound`] when the base resolves to no model,
    /// naming `--decision-base`; the [`LoraError`] kinds of a base that does
    /// not load or an adapter that does not load or fit.
    pub(crate) async fn load_kev(&self, name: &str) -> Result<Option<Arc<Executor<R>>>, LoraError> {
        let Some(dir) = self.model_path(name).filter(|p| is_kev_dir(p)) else {
            return Ok(None);
        };
        let configured = self.decision_config.base_for(name);
        let base = resolve_kev_base(&dir, configured, self.model_dir(), &hub_caches())
            .map_err(|e| LoraError::ModelNotFound(format!("Kev model '{name}': {e}")))?;
        let adapter = dir
            .file_name()
            .and_then(|n| n.to_str())
            .filter(|n| check_name(n).is_ok())
            .unwrap_or(KEV_ADAPTER);
        let spec = AdapterSpec::new(adapter, &dir)?;
        let (info, executor) = self
            .build_variant(name.to_string(), &base.name, spec, false)
            .await?;
        tracing::info!(
            model = name,
            base = %base.path.display(),
            rank = info.rank,
            "Kev model built over its base"
        );
        Ok(Some(executor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::decision::kev::tests::kev_dir;
    use boostr::{CpuDevice, CpuRuntime};

    #[test]
    fn adapter_dirs_of_a_directory_and_a_file() {
        let dir = kev_dir("b");
        assert_eq!(adapter_dir(dir.path()), Some(dir.path().to_path_buf()));
        let file = dir.path().join("adapter_model.safetensors");
        assert_eq!(adapter_dir(&file), Some(dir.path().to_path_buf()));
    }

    #[tokio::test]
    async fn a_kev_directory_without_a_base_names_the_flag() {
        let models = tempfile::tempdir().expect("temp dir");
        let kev = models.path().join("kev-4b");
        std::fs::create_dir(&kev).expect("mkdir");
        let source = kev_dir("Nobody/No-Such-Base-For-Blazr-Tests");
        for file in [
            "adapter_config.json",
            "adapter_model.safetensors",
            "head.pt",
        ] {
            std::fs::copy(source.path().join(file), kev.join(file)).expect("copy");
        }
        let scheduler = Scheduler::<CpuRuntime>::new(models.path().to_path_buf(), CpuDevice::new());
        assert_eq!(scheduler.kev_dir("kev-4b"), Some(kev.clone()));
        assert_eq!(scheduler.kev_dir("other"), None);
        assert_eq!(scheduler.kev_base_path("kev-4b", &kev), None);
        let err = scheduler
            .get_executor("kev-4b")
            .await
            .err()
            .expect("no base");
        let lora = err.downcast_ref::<LoraError>().expect("a LoraError");
        assert!(matches!(lora, LoraError::ModelNotFound(_)), "{lora:?}");
        assert!(lora.message().contains("--decision-base"), "{lora}");
    }
}
