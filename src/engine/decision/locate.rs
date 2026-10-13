//! The disk reads of a decision model's load: where its files are, its
//! family, its GGUF header and its renderer. Each reads the disk, so the
//! async load runs it on a blocking thread through [`blocking`].

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use super::error::DecisionFailure;
use super::kev_model::{kev_base_of, kev_dir_of};
use super::registry::{lock, variant_decision_config};
use crate::config::DecisionConfig;
use crate::engine::load::blocking::run_blocking;
use crate::engine::model_dir::find_model_path;
use crate::engine::VariantInfo;
use crate::loader::decision::{
    detect_decision_with, load_renderer, read_gguf_facts, DecisionSource, GgufFacts, LoadedRenderer,
};
use crate::model::decision_template::Family;

/// The GGUF headers a registry read, by file, with the file's modified
/// time at the read.
pub(super) type GgufHeaders = HashMap<PathBuf, (Option<SystemTime>, GgufFacts)>;

/// What the GGUF header of `file` says, read once per file version and
/// kept in `headers`.
///
/// # Errors
///
/// When the header does not read.
pub(super) fn cached_gguf_facts(
    headers: &Mutex<GgufHeaders>,
    file: &Path,
) -> Result<GgufFacts, String> {
    let stamp = std::fs::metadata(file).and_then(|m| m.modified()).ok();
    if let Some((seen, facts)) = lock(headers).get(file) {
        if *seen == stamp && stamp.is_some() {
            return Ok(facts.clone());
        }
    }
    let facts = read_gguf_facts(file)?;
    lock(headers).insert(file.to_path_buf(), (stamp, facts.clone()));
    Ok(facts)
}

/// The decision settings detection runs with: `cfg`, or for the LoRA
/// variant `variant`, [`variant_decision_config`] over it.
pub(super) fn settings_of<'a>(
    cfg: &'a DecisionConfig,
    variant: Option<&VariantInfo>,
) -> Cow<'a, DecisionConfig> {
    match variant {
        Some(v) => Cow::Owned(variant_decision_config(cfg, &v.base, &v.path)),
        None => Cow::Borrowed(cfg),
    }
}

/// The path the model `name` serves decisions from and its decision
/// source: the Kev directory `name` serves, else the weights of `name`, or
/// of its base for the LoRA variant `variant`. `Ok(None)` when no such
/// weights exist.
///
/// # Errors
///
/// [`DecisionFailure::Internal`] when a sidecar or GGUF header does not
/// read, [`DecisionFailure::NotDecisionModel`] when the model is not a
/// decision model.
pub(super) fn locate_decision(
    model_dir: &Path,
    cfg: &DecisionConfig,
    variant: Option<&VariantInfo>,
    name: &str,
    headers: &Mutex<GgufHeaders>,
) -> Result<Option<(PathBuf, DecisionSource)>, DecisionFailure> {
    let path = match kev_dir_of(model_dir, variant, name) {
        Some(dir) => dir,
        None => {
            let weights = variant.map_or(name, |v| v.base.as_str());
            let Some(path) = find_model_path(model_dir, weights) else {
                return Ok(None);
            };
            path
        }
    };
    let settings = settings_of(cfg, variant);
    let source = detect_decision_with(&path, name, &settings, |file| {
        cached_gguf_facts(headers, file)
    })
    .map_err(DecisionFailure::Internal)?
    .ok_or_else(|| {
        DecisionFailure::NotDecisionModel(format!(
            "model '{name}' is not a System One decision model: it ships no \
             decider_config.json, prompt_contract.json, jebadiah.json or joint-schema head \
             (joint_head_config.json with joint_head.safetensors), its GGUF header names \
             no decision model, and no --decision-family applies to it"
        ))
    })?;
    Ok(Some((path, source)))
}

/// The renderer of the decision model `name` found at `path` as `source`:
/// for a Kev model, over the base it runs on.
///
/// # Errors
///
/// [`DecisionFailure::Internal`] naming `name` when the tokenizer,
/// template or a sidecar does not load, or the contract does not match.
pub(super) fn renderer_of(
    model_dir: &Path,
    cfg: &DecisionConfig,
    variant: Option<&VariantInfo>,
    name: &str,
    path: &Path,
    source: &DecisionSource,
) -> Result<LoadedRenderer, DecisionFailure> {
    let base = match source.family {
        Family::Kev => kev_base_of(model_dir, cfg, variant, name, path),
        Family::Jebadiah | Family::Decider | Family::Clef => None,
    };
    load_renderer(source, cfg, base.as_deref())
        .map_err(|e| DecisionFailure::Internal(format!("model '{name}': {e}")))
}

/// Run `work`, which reads the disk, off the async executor through
/// [`run_blocking`].
///
/// # Errors
///
/// The error of `work`, or [`DecisionFailure::Internal`] naming the model
/// `name` when `work` panics or the runtime shuts down before it runs.
pub(super) async fn blocking<T: Send + 'static>(
    name: &str,
    work: impl FnOnce() -> Result<T, DecisionFailure> + Send + 'static,
) -> Result<T, DecisionFailure> {
    run_blocking(work).await.map_err(|e| {
        DecisionFailure::Internal(format!("model '{name}': the decision load stopped: {e}"))
    })?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_model_locates_to_none_and_a_plain_one_is_refused() {
        let models = tempfile::tempdir().expect("temp dir");
        let headers = Mutex::new(GgufHeaders::new());
        let cfg = DecisionConfig::default();
        let missing = locate_decision(models.path(), &cfg, None, "absent", &headers);
        assert!(matches!(missing, Ok(None)), "{missing:?}");

        std::fs::create_dir(models.path().join("plain")).expect("mkdir");
        let plain = locate_decision(models.path(), &cfg, None, "plain", &headers);
        assert!(
            matches!(plain, Err(DecisionFailure::NotDecisionModel(_))),
            "{plain:?}"
        );
    }

    #[tokio::test]
    async fn blocking_work_returns_its_value_and_its_error() {
        assert_eq!(blocking("m", || Ok(7)).await, Ok(7));
        let failed = blocking::<()>("m", || Err(DecisionFailure::Busy("b".into()))).await;
        assert_eq!(failed, Err(DecisionFailure::Busy("b".into())));
    }
}
