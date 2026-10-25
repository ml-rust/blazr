//! The draft model of `drafter: model`: its load and its compatibility check.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};

use super::eligibility::load_fallback;
use crate::config::{DrafterKind, DtypeRequest, DtypeSource, InferenceConfig};
use crate::engine::executor::Executor;
use crate::engine::load::parts::{load_model_parts, start_executor};

/// Texts both tokenizers encode in [`check_draft_compat`].
///
/// They cover words, digits, code, whitespace runs and multi-byte UTF-8.
const PROBE_TEXTS: [&str; 7] = [
    "Hello, world!",
    " The quick brown fox jumps over the lazy dog.",
    "0123456789 3.14159 -42",
    "def f(x):\n    return x * 2\n",
    "  two  spaces\tand\ttabs\n\n",
    "naïve café, Straße, 東京, Москва",
    "emoji 🚀🔥 and symbols <>&\"'",
];

/// Turn off every feature of `inference` a draft executor never runs.
///
/// The draft runs contiguous KV on one device, with no speculation of its own.
fn draft_only(inference: &mut InferenceConfig) {
    inference.speculative = None;
    inference.paged_attention = false;
    inference.prefix_cache = false;
    inference.gpu_prefix_cache = false;
    inference.graphs = false;
    inference.tensor_parallel_size = 1;
    inference.mmproj = None;
    inference.lora.clear();
}

/// Check that `draft` drafts in the token ids of `target`.
///
/// The model vocabularies must match. Both tokenizers must encode every probe
/// text to the same ids. Special tokens past the shared text vocabulary can
/// differ: the target alone decides every emitted token, so a draft id it does
/// not share only lowers acceptance.
///
/// # Errors
///
/// An error naming `inference.speculative.draft_model`, the mismatch, both
/// vocabulary sizes and the next action.
pub(in crate::engine) fn check_draft_compat<R>(
    target: &Executor<R>,
    draft: &Executor<R>,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R> + SamplingOps<R> + boostr::GrammarDfaOps<R>,
{
    let (tv, dv) = (target.vocab_size(), draft.vocab_size());
    let refuse = |what: String| {
        anyhow!(
            "inference.speculative.draft_model does not fit the target model: {what}. The \
             target vocab_size is {tv}, the draft vocab_size is {dv}. Set draft_model to a \
             model with the target's tokenizer"
        )
    };
    if tv != dv {
        return Err(refuse("the model vocabularies differ".into()));
    }
    let (t, d) = (&*target.tokenizer, &*draft.tokenizer);
    for text in PROBE_TEXTS {
        let (a, b) = (t.encode(text), d.encode(text));
        if a != b {
            return Err(refuse(format!(
                "the tokenizers encode {text:?} as {a:?} and {b:?}"
            )));
        }
    }
    Ok(())
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
        + ModelClient<R>,
{
    /// Load the draft model `inference.speculative.draft_model` names.
    ///
    /// It loads only with speculation on, `drafter: model`, and a target
    /// that can speculate. The draft takes this executor's device and
    /// compute dtype, contiguous KV, no warmup and no tensor parallelism.
    /// Every request shares its weights. Each request builds its own draft
    /// KV cache.
    ///
    /// # Errors
    ///
    /// An error naming `inference.speculative.draft_model` when the path is
    /// unset, the draft does not load, or it does not fit the target.
    pub(crate) fn with_configured_draft_model(self) -> Result<Self> {
        let Some(spec) = self.config.inference.speculative.as_ref() else {
            return Ok(self);
        };
        if !spec.enabled
            || spec.drafter != DrafterKind::Model
            || self
                .model()
                .map_or(true, |model| load_fallback(model).is_err())
        {
            return Ok(self);
        }
        let path = spec
            .draft_model()
            .ok_or_else(|| {
                anyhow!(
                    "inference.speculative.draft_model is unset; drafter model needs a draft \
                     model. Set draft_model to a GGUF file or a SafeTensors directory"
                )
            })?
            .to_string();
        let draft = self.load_draft(Path::new(&path)).map_err(|e| {
            anyhow!(
                "inference.speculative.draft_model {path:?} did not load: {e:#}. Set \
                 draft_model to a GGUF file or a SafeTensors directory"
            )
        })?;
        let exec = self.with_draft(draft)?;
        tracing::info!(draft_model = %path, "speculative draft model loaded");
        Ok(exec)
    }

    /// The draft executor at `path`, on this executor's device and dtype.
    fn load_draft(&self, path: &Path) -> Result<Executor<R>> {
        let dtype = DtypeRequest {
            dtype: self.config.compute_dtype()?,
            source: DtypeSource::File,
        };
        let (model, mut config, tokenizer) =
            load_model_parts::<R>(path, &self.device, Some(&dtype), None)?;
        draft_only(&mut config.inference);
        start_executor(
            path,
            (model, config, tokenizer),
            self.device.clone(),
            self.num_ctx,
            None,
        )
    }

    /// Attach `draft` as the draft model of every request.
    ///
    /// # Errors
    ///
    /// The error of [`check_draft_compat`].
    pub(crate) fn with_draft(mut self, draft: Executor<R>) -> Result<Self> {
        check_draft_compat(&self, &draft)?;
        self.draft = Some(Arc::new(draft));
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::SpeculativeDecodingConfig;
    use crate::engine::test_support::{full_vocab_executor, seeded_executor, seeded_executor_pair};

    const SEED: u64 = 0x5bec;

    fn model_drafter(path: &str) -> SpeculativeDecodingConfig {
        SpeculativeDecodingConfig {
            enabled: true,
            drafter: DrafterKind::Model,
            draft_model: Some(path.to_string()),
            ..SpeculativeDecodingConfig::default()
        }
    }

    #[test]
    fn a_matching_draft_attaches() -> Result<()> {
        let target = seeded_executor_pair(SEED, SEED + 1, 64, None)?;
        assert!(target.draft.is_some());
        assert!(target.with_draft(seeded_executor(SEED + 2, None)?).is_ok());
        Ok(())
    }

    #[test]
    fn a_vocab_mismatch_is_refused_with_both_sizes() -> Result<()> {
        let target = seeded_executor(SEED, None)?;
        let draft = full_vocab_executor(None)?;
        let (tv, dv) = (target.vocab_size(), draft.vocab_size());
        assert_ne!(tv, dv);
        let err = target
            .with_draft(draft)
            .map(|_| ())
            .map_err(|e| e.to_string());
        let Err(msg) = err else {
            panic!("a vocab mismatch attached");
        };
        assert!(msg.contains("inference.speculative.draft_model"), "{msg}");
        assert!(
            msg.contains(&tv.to_string()) && msg.contains(&dv.to_string()),
            "{msg}"
        );
        assert!(msg.contains("Set draft_model"), "{msg}");
        Ok(())
    }

    #[test]
    fn a_missing_draft_path_fails_the_load_naming_the_key() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let missing = dir.path().join("absent");
        let mut exec = seeded_executor(SEED, None)?;
        exec.config.inference.speculative = Some(model_drafter(&missing.to_string_lossy()));
        let err = exec
            .with_configured_draft_model()
            .map(|_| ())
            .map_err(|e| e.to_string());
        assert!(
            matches!(&err, Err(msg) if msg.contains("inference.speculative.draft_model")),
            "{err:?}"
        );
        Ok(())
    }

    #[test]
    fn no_draft_loads_unless_the_model_drafter_is_on() -> Result<()> {
        let off = SpeculativeDecodingConfig {
            enabled: false,
            ..model_drafter("/absent/draft")
        };
        let lookup = SpeculativeDecodingConfig {
            drafter: DrafterKind::PromptLookup,
            ..model_drafter("/absent/draft")
        };
        for spec in [None, Some(off), Some(lookup)] {
            let mut exec = seeded_executor(SEED, None)?;
            exec.config.inference.speculative = spec;
            assert!(exec.with_configured_draft_model()?.draft.is_none());
        }
        Ok(())
    }

    #[test]
    fn draft_only_turns_off_what_a_draft_never_runs() {
        let mut inference = InferenceConfig {
            speculative: Some(model_drafter("x")),
            paged_attention: true,
            prefix_cache: true,
            graphs: true,
            tensor_parallel_size: 2,
            lora: vec!["adapter".into()],
            ..InferenceConfig::default()
        };
        draft_only(&mut inference);
        assert!(inference.speculative.is_none());
        assert!(!inference.paged_attention && !inference.prefix_cache && !inference.graphs);
        assert_eq!(inference.tensor_parallel_size, 1);
        assert!(inference.lora.is_empty());
    }
}
