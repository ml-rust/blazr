//! A Kev directory: a PEFT LoRA adapter and a pointer head (`head.pt`)
//! trained over a Qwen3.5 base the directory does not hold. Its detection,
//! the adapter's base and rank, and the head's fitted temperature.

use std::collections::BTreeMap;
use std::path::Path;

use boostr::format::TorchStateDict;
use serde_json::Value;

use super::detect::DecisionSource;
use super::sidecar::{read_json_opt, Calibration, TYPES};
use crate::loader::detect_model_source;
use crate::model::decision_template::Family;

/// The pointer head and its fitted temperature.
pub const HEAD_FILE: &str = "head.pt";
/// The PEFT adapter config: base model, rank and target modules.
pub const ADAPTER_CONFIG: &str = "adapter_config.json";
/// The PEFT adapter weights.
pub const ADAPTER_WEIGHTS: &str = "adapter_model.safetensors";
/// Kev's training arguments, `base_revision` among them.
pub const TRAINING_CONFIG: &str = "training_config.json";
/// Kev's provenance record, `config.base_revision` among them.
pub const PROVENANCE: &str = "provenance.json";

/// Whether `path` is a Kev directory: `head.pt`, `adapter_config.json`
/// and `adapter_model.safetensors`, and no base weights of its own.
pub fn is_kev_dir(path: &Path) -> bool {
    path.is_dir()
        && [HEAD_FILE, ADAPTER_CONFIG, ADAPTER_WEIGHTS]
            .iter()
            .all(|f| path.join(f).is_file())
        && detect_model_source(path).is_err()
}

/// The decision source of the Kev directory `dir`.
pub fn kev_source(dir: &Path) -> DecisionSource {
    DecisionSource {
        family: Family::Kev,
        detected_by: format!("{HEAD_FILE}, {ADAPTER_CONFIG} and {ADAPTER_WEIGHTS}"),
        model_dir: Some(dir.to_path_buf()),
        gguf: None,
        gguf_facts: None,
        sidecar_dir: dir.to_path_buf(),
        config_file: None,
    }
}

/// What a Kev directory's adapter says about its base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KevAdapter {
    /// `base_model_name_or_path`: a Hugging Face id such as
    /// `Qwen/Qwen3.5-4B-Base`, or a path.
    pub base: String,
    /// The adapter's `r`.
    pub rank: usize,
    /// The base revision Kev trained on: `training_config.json`'s
    /// `base_revision`, else `provenance.json`'s `config.base_revision`.
    pub base_revision: Option<String>,
}

/// The adapter facts of the Kev directory `dir`.
///
/// # Errors
///
/// The file and the field when `adapter_config.json` does not read or
/// lacks `base_model_name_or_path` or a positive `r`, or a revision file
/// exists and does not parse.
pub fn read_kev_adapter(dir: &Path) -> Result<KevAdapter, String> {
    let path = dir.join(ADAPTER_CONFIG);
    let cfg = read_json_opt(&path)?.ok_or_else(|| format!("{} is missing", path.display()))?;
    let base = cfg
        .get("base_model_name_or_path")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            format!(
                "{}: base_model_name_or_path must name the base model",
                path.display()
            )
        })?
        .trim()
        .to_string();
    let rank = cfg
        .get("r")
        .and_then(Value::as_u64)
        .and_then(|r| usize::try_from(r).ok())
        .filter(|&r| r > 0)
        .ok_or_else(|| format!("{}: r must be a positive integer", path.display()))?;
    let from_training = read_json_opt(&dir.join(TRAINING_CONFIG))?.and_then(|c| {
        c.get("base_revision")
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    let base_revision = match from_training {
        Some(rev) => Some(rev),
        None => read_json_opt(&dir.join(PROVENANCE))?.and_then(|p| {
            p.get("config")?
                .get("base_revision")?
                .as_str()
                .map(str::to_string)
        }),
    };
    Ok(KevAdapter {
        base,
        rank,
        base_revision,
    })
}

/// The temperature `head.pt` was fitted with. `None` when it sets none,
/// which Kev reads as 1.
///
/// # Errors
///
/// When `head.pt` does not open, or its temperature is not a number above
/// 0.
pub fn head_temperature(dir: &Path) -> Result<Option<f64>, String> {
    let path = dir.join(HEAD_FILE);
    let head = TorchStateDict::open(&path).map_err(|e| e.to_string())?;
    let Some(t) = head.scalar("temperature") else {
        return Ok(None);
    };
    match t.as_f64() {
        Some(t) if t.is_finite() && t > 0.0 => Ok(Some(t)),
        _ => Err(format!(
            "{}: temperature must be a number above 0, got {t:?}",
            path.display()
        )),
    }
}

/// The temperature a Kev directory ships, for every answer type: one
/// value, `head.pt`'s. `None` when `head.pt` sets none.
///
/// # Errors
///
/// As [`head_temperature`].
pub fn kev_calibration(dir: &Path) -> Result<Option<Calibration>, String> {
    Ok(head_temperature(dir)?.map(|t| {
        let temperatures: BTreeMap<String, f64> =
            TYPES.iter().map(|kind| (kind.to_string(), t)).collect();
        Calibration::from_file(HEAD_FILE, temperatures)
    }))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A Kev directory without a readable head: the files detection reads.
    pub(crate) fn kev_dir(base: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        let cfg = serde_json::json!({"base_model_name_or_path": base, "r": 16,
            "lora_alpha": 32, "peft_type": "LORA"});
        std::fs::write(dir.path().join(ADAPTER_CONFIG), cfg.to_string()).expect("write");
        std::fs::write(dir.path().join(ADAPTER_WEIGHTS), b"").expect("write");
        std::fs::write(dir.path().join(HEAD_FILE), b"").expect("write");
        dir
    }

    #[test]
    fn detection_needs_the_three_files_and_no_weights() {
        let dir = kev_dir("Qwen/Qwen3.5-4B-Base");
        assert!(is_kev_dir(dir.path()));
        let source = kev_source(dir.path());
        assert_eq!(source.family, Family::Kev);
        assert_eq!(source.sidecar_dir, dir.path());

        std::fs::write(dir.path().join("model.safetensors"), b"").expect("write");
        assert!(
            !is_kev_dir(dir.path()),
            "a directory with weights is a model"
        );
        let partial = kev_dir("b");
        std::fs::remove_file(partial.path().join(ADAPTER_WEIGHTS)).expect("remove");
        assert!(!is_kev_dir(partial.path()));
        assert!(!is_kev_dir(&partial.path().join(HEAD_FILE)));
    }

    #[test]
    fn adapter_names_base_rank_and_revision() {
        let dir = kev_dir(" Qwen/Qwen3.5-4B-Base ");
        let adapter = read_kev_adapter(dir.path()).expect("reads");
        assert_eq!(adapter.base, "Qwen/Qwen3.5-4B-Base");
        assert_eq!(adapter.rank, 16);
        assert_eq!(adapter.base_revision, None);

        let provenance = serde_json::json!({"config": {"base_revision": "abc"}});
        std::fs::write(dir.path().join(PROVENANCE), provenance.to_string()).expect("write");
        let rev = |d: &Path| read_kev_adapter(d).expect("reads").base_revision;
        assert_eq!(rev(dir.path()).as_deref(), Some("abc"));
        let training = serde_json::json!({"base_revision": "def"});
        std::fs::write(dir.path().join(TRAINING_CONFIG), training.to_string()).expect("write");
        assert_eq!(rev(dir.path()).as_deref(), Some("def"));

        let zero_rank = r#"{"r": 0, "base_model_name_or_path": "b"}"#;
        std::fs::write(dir.path().join(ADAPTER_CONFIG), zero_rank).expect("write");
        let e = read_kev_adapter(dir.path()).expect_err("r 0");
        assert!(e.contains("r must be"), "{e}");
        std::fs::write(dir.path().join(ADAPTER_CONFIG), r#"{"r": 4}"#).expect("write");
        let e = read_kev_adapter(dir.path()).expect_err("no base");
        assert!(e.contains("base_model_name_or_path"), "{e}");
    }

    #[test]
    fn an_unreadable_head_is_an_error() {
        let dir = kev_dir("b");
        assert!(head_temperature(dir.path()).is_err());
        assert!(kev_calibration(dir.path()).is_err());
    }
}
