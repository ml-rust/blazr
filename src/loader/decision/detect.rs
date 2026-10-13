//! Whether a model is a System One decision model, its family, and where
//! its sidecar files are.
//!
//! A Kev directory (`head.pt`, `adapter_config.json` and
//! `adapter_model.safetensors`, no base weights) is a Kev model whatever
//! else is configured. For any other model the resolution order is
//! `inference.decision.family` (`--decision-family`), the
//! `inference.decision.config` sidecar's file name (`--decision-config`),
//! a sidecar in the model directory or beside the `.gguf`
//! (`decider_config.json`: decider; `prompt_contract.json` or
//! `jebadiah.json`: Jebadiah), then a GGUF header whose `general.basename`
//! is `decider` or `jebadiah` and whose `general.tags` hold `system-one` or
//! `decision-model`.

use std::path::{Path, PathBuf};

use boostr::format::{Gguf, GgufMetadata};

use super::kev::{is_kev_dir, kev_source};
use super::sidecar::{
    context_length, license_and_base, DECIDER_CONFIG, JEBADIAH_CARD, PROMPT_CONTRACT,
};
use crate::config::DecisionConfig;
use crate::loader::{chat_template_from_metadata, detect_model_source, ModelFormat};
use crate::model::chat_template::ModelTemplate;
use crate::model::decision_template::Family;

/// What a GGUF header says about a decision model.
#[derive(Debug, Clone, Default)]
pub struct GgufFacts {
    /// The family `general.basename` and `general.tags` name.
    pub family: Option<Family>,
    /// `general.license`.
    pub license: Option<String>,
    /// `general.base_model.0.repo_url`, else its `name`.
    pub base_model: Option<String>,
    /// `<arch>.context_length`.
    pub context_length: Option<u64>,
    /// `tokenizer.chat_template`.
    pub chat_template: Option<ModelTemplate>,
}

impl GgufFacts {
    /// The facts of an open header.
    pub fn of(metadata: &GgufMetadata) -> Self {
        let (license, base_model) = license_and_base(None, Some(metadata));
        Self {
            family: header_family(metadata),
            license,
            base_model,
            context_length: context_length(None, Some(metadata)),
            chat_template: chat_template_from_metadata(metadata),
        }
    }
}

/// The family a GGUF header names: `general.basename` `decider` or
/// `jebadiah`, with `system-one` or `decision-model` among the tags.
pub fn header_family(metadata: &GgufMetadata) -> Option<Family> {
    let tags = metadata
        .get_string_array("general.tags")
        .unwrap_or_default();
    if !tags
        .iter()
        .any(|t| t == "system-one" || t == "decision-model")
    {
        return None;
    }
    match metadata
        .get_string("general.basename")?
        .to_ascii_lowercase()
        .as_str()
    {
        "decider" => Some(Family::Decider),
        "jebadiah" => Some(Family::Jebadiah),
        _ => None,
    }
}

/// A decision model: its family and its files.
#[derive(Debug, Clone)]
pub struct DecisionSource {
    /// The family whose prompt the model reads.
    pub family: Family,
    /// What named the family, as `/v1/models` reports it.
    pub detected_by: String,
    /// The checkpoint directory of a SafeTensors model: tokenizer, chat
    /// template, `config.json` and model card.
    pub model_dir: Option<PathBuf>,
    /// The `.gguf` file of a GGUF model.
    pub gguf: Option<PathBuf>,
    /// What the GGUF header says. `None` for a SafeTensors model.
    pub gguf_facts: Option<GgufFacts>,
    /// The directory the sidecars are read from.
    pub sidecar_dir: PathBuf,
    /// The sidecar `inference.decision.config` names, when it applies.
    pub config_file: Option<PathBuf>,
}

/// The family a sidecar file name names.
fn sidecar_family(file: &Path) -> Option<Family> {
    match file.file_name()?.to_str()? {
        DECIDER_CONFIG => Some(Family::Decider),
        PROMPT_CONTRACT | JEBADIAH_CARD => Some(Family::Jebadiah),
        _ => None,
    }
}

/// What the header of the GGUF `file` says.
///
/// # Errors
///
/// When the header does not read.
pub fn read_gguf_facts(file: &Path) -> Result<GgufFacts, String> {
    let reader = Gguf::open(file)
        .map_err(|e| format!("{}: the GGUF header does not read: {e}", file.display()))?;
    Ok(GgufFacts::of(reader.metadata()))
}

/// Whether the model at `path`, served as `name`, is a decision model.
///
/// # Errors
///
/// When a family is configured for a model whose weights are not found,
/// when the configured sidecar names no family and none is configured, or
/// when a GGUF header does not read.
pub fn detect_decision(
    path: &Path,
    name: &str,
    cfg: &DecisionConfig,
) -> Result<Option<DecisionSource>, String> {
    detect_decision_with(path, name, cfg, read_gguf_facts)
}

/// [`detect_decision`] with the GGUF header read by `read_gguf`, such as
/// a cache of [`read_gguf_facts`].
///
/// # Errors
///
/// As [`detect_decision`], and `read_gguf`'s errors.
pub fn detect_decision_with(
    path: &Path,
    name: &str,
    cfg: &DecisionConfig,
    read_gguf: impl FnOnce(&Path) -> Result<GgufFacts, String>,
) -> Result<Option<DecisionSource>, String> {
    if is_kev_dir(path) {
        return Ok(Some(kev_source(path)));
    }
    let forced = cfg.overrides_apply_to(name);
    let flag_family = cfg.family.filter(|_| forced).map(|f| f.family());
    let config_file = cfg.config.clone().filter(|_| forced);
    let source = match detect_model_source(path) {
        Ok(source) => source,
        Err(_) if flag_family.is_none() && config_file.is_none() => return Ok(None),
        Err(e) => return Err(format!("model '{name}': {e:#}")),
    };
    let (model_dir, gguf) = match source.format {
        ModelFormat::Gguf => (None, Some(source.weights_path.clone())),
        ModelFormat::SafeTensors if path.is_dir() => (Some(path.to_path_buf()), None),
        ModelFormat::SafeTensors => (path.parent().map(Path::to_path_buf), None),
    };
    let gguf_facts = match &gguf {
        Some(file) => Some(read_gguf(file)?),
        None => None,
    };
    let sidecar_dir = config_file
        .as_deref()
        .and_then(Path::parent)
        .or(model_dir.as_deref())
        .or_else(|| gguf.as_deref().and_then(Path::parent))
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    let named = if let Some(family) = flag_family {
        Some((family, "inference.decision.family".to_string()))
    } else if let Some(file) = &config_file {
        let family = sidecar_family(file).ok_or_else(|| {
            format!(
                "{}: the family of this sidecar is unknown. Name it {DECIDER_CONFIG}, \
                 {PROMPT_CONTRACT} or {JEBADIAH_CARD}, or set --decision-family",
                file.display()
            )
        })?;
        Some((family, file.display().to_string()))
    } else {
        [DECIDER_CONFIG, PROMPT_CONTRACT, JEBADIAH_CARD]
            .iter()
            .map(|f| sidecar_dir.join(f))
            .find(|f| f.is_file())
            .and_then(|f| Some((sidecar_family(&f)?, f.display().to_string())))
            .or_else(|| {
                let family = gguf_facts.as_ref()?.family?;
                Some((family, "GGUF general.basename and general.tags".to_string()))
            })
    };
    Ok(named.map(|(family, detected_by)| DecisionSource {
        family,
        detected_by,
        model_dir,
        gguf,
        gguf_facts,
        sidecar_dir,
        config_file,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DecisionFamily;

    fn model_dir(sidecar: Option<&str>) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("model.safetensors"), b"").expect("write");
        if let Some(file) = sidecar {
            std::fs::write(dir.path().join(file), "{}").expect("write");
        }
        dir
    }

    fn family(path: &Path, cfg: &DecisionConfig) -> Option<Family> {
        detect_decision(path, "m", cfg)
            .expect("detects")
            .map(|s| s.family)
    }

    #[test]
    fn sidecars_name_the_family() {
        let cfg = DecisionConfig::default();
        let decider = model_dir(Some(DECIDER_CONFIG));
        assert_eq!(family(decider.path(), &cfg), Some(Family::Decider));
        let jebadiah = model_dir(Some(PROMPT_CONTRACT));
        assert_eq!(family(jebadiah.path(), &cfg), Some(Family::Jebadiah));
        let card = model_dir(Some(JEBADIAH_CARD));
        assert_eq!(family(card.path(), &cfg), Some(Family::Jebadiah));
        let plain = model_dir(None);
        assert_eq!(family(plain.path(), &cfg), None);
    }

    #[test]
    fn configured_family_and_sidecar_win_for_their_model() {
        let plain = model_dir(None);
        let cfg = DecisionConfig {
            family: Some(DecisionFamily::Jebadiah),
            start_model: Some("m".into()),
            ..Default::default()
        };
        assert_eq!(family(plain.path(), &cfg), Some(Family::Jebadiah));
        let other = detect_decision(plain.path(), "other", &cfg).expect("detects");
        assert!(
            other.is_none(),
            "the family applies to the start model only"
        );

        let sidecars = model_dir(Some(DECIDER_CONFIG));
        let cfg = DecisionConfig {
            config: Some(sidecars.path().join(DECIDER_CONFIG)),
            ..Default::default()
        };
        let source = detect_decision(plain.path(), "m", &cfg)
            .expect("detects")
            .expect("a decision model");
        assert_eq!(source.family, Family::Decider);
        assert_eq!(source.sidecar_dir, sidecars.path());

        let cfg = DecisionConfig {
            config: Some(sidecars.path().join("other.json")),
            ..Default::default()
        };
        assert!(detect_decision(plain.path(), "m", &cfg).is_err());
    }

    #[test]
    fn missing_weights_are_not_a_decision_model_unless_configured() {
        let empty = tempfile::tempdir().expect("temp dir");
        assert_eq!(family(empty.path(), &DecisionConfig::default()), None);
        let cfg = DecisionConfig {
            family: Some(DecisionFamily::Decider),
            ..Default::default()
        };
        assert!(detect_decision(empty.path(), "m", &cfg).is_err());
    }

    #[test]
    fn a_kev_directory_is_kev_whatever_is_configured() {
        let kev = crate::loader::decision::kev::tests::kev_dir("Qwen/Qwen3.5-4B-Base");
        assert_eq!(
            family(kev.path(), &DecisionConfig::default()),
            Some(Family::Kev)
        );
        let forced = DecisionConfig {
            family: Some(DecisionFamily::Jebadiah),
            ..Default::default()
        };
        let source = detect_decision(kev.path(), "m", &forced)
            .expect("detects")
            .expect("a decision model");
        assert_eq!(source.family, Family::Kev);
        assert_eq!(source.model_dir.as_deref(), Some(kev.path()));
    }
}
