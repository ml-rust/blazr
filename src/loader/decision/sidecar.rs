//! Decision sidecar files: the calibration temperatures each family ships,
//! the chat template hash a prompt contract pins, and the license, base
//! model and context length a model card and config name.

use std::collections::BTreeMap;
use std::path::Path;

use boostr::format::GgufMetadata;
use boostr::inference::decision::TemperaturePolicy;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::model::decision_template::py_text::py_float;

/// The decider's config: layout, flags and temperatures.
pub const DECIDER_CONFIG: &str = "decider_config.json";
/// Jebadiah's prompt contract.
pub const PROMPT_CONTRACT: &str = "prompt_contract.json";
/// Jebadiah's model card, holding a copy of the prompt contract.
pub const JEBADIAH_CARD: &str = "jebadiah.json";
/// Jebadiah's fitted temperatures.
pub const TEMPERATURES: &str = "temperatures.json";

/// Every vendor sidecar `blazr decision calibrate` never writes.
pub const VENDOR_SIDECARS: [&str; 8] = [
    DECIDER_CONFIG,
    PROMPT_CONTRACT,
    JEBADIAH_CARD,
    TEMPERATURES,
    super::kev::HEAD_FILE,
    super::kev::ADAPTER_CONFIG,
    super::clef::HEAD_CONFIG,
    super::clef::HEAD_WEIGHTS,
];

/// The answer types a temperature applies to.
pub const TYPES: [&str; 3] = ["choice", "noul", "score"];

/// The temperatures a model was calibrated with.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Calibration {
    /// The file the temperatures came from: `blazr_calibration.json` when
    /// it sets any type, else the vendor sidecar.
    pub source: String,
    /// The temperature of each answer type a file sets.
    pub temperatures: BTreeMap<String, f64>,
    /// The file each type's temperature came from.
    pub sources: BTreeMap<String, String>,
    /// The fit date of `blazr_calibration.json` when it sets any type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fitted_at: Option<String>,
}

impl Calibration {
    /// Temperatures that all come from the one file `source`.
    pub fn from_file(source: &str, temperatures: BTreeMap<String, f64>) -> Self {
        Self {
            source: source.to_string(),
            sources: temperatures
                .keys()
                .map(|kind| (kind.clone(), source.to_string()))
                .collect(),
            temperatures,
            fitted_at: None,
        }
    }

    /// One temperature per answer type. A type the file leaves out takes 1.
    /// When every type takes the same temperature, as Kev's `head.pt`
    /// sets it, the policy is that one fixed temperature.
    pub fn policy(&self) -> TemperaturePolicy {
        let t = |kind: &str| self.temperatures.get(kind).copied().unwrap_or(1.0);
        let (choice, noul, score) = (t("choice"), t("noul"), t("score"));
        if choice == noul && noul == score {
            TemperaturePolicy::Fixed(choice)
        } else {
            TemperaturePolicy::ByKind {
                choice,
                noul,
                score,
            }
        }
    }
}

/// A JSON file, `None` when absent.
///
/// # Errors
///
/// The path and the reason when the file exists and does not read or parse.
pub fn read_json_opt(path: &Path) -> Result<Option<Value>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("{} does not read: {e}", path.display()))?;
    serde_json::from_str(&text)
        .map(Some)
        .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))
}

/// Python's `float(value)` for a JSON value: a number, or a string that
/// parses as one. `None` for anything else, `true` and `false` included.
fn as_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => py_float(s).ok(),
        _ => None,
    }
}

/// Jebadiah's `read_temperatures`: `temperatures.json`'s `temperatures`
/// map, each type that holds a finite positive number. `None` when the
/// file is missing, does not parse, or sets no type.
pub fn jebadiah_temperatures(dir: &Path) -> Option<Calibration> {
    let raw = read_json_opt(&dir.join(TEMPERATURES)).ok().flatten()?;
    let temps = raw.get("temperatures")?.as_object()?;
    let temperatures: BTreeMap<String, f64> = TYPES
        .iter()
        .filter_map(|kind| {
            let t = temps.get(*kind).and_then(as_float)?;
            (t.is_finite() && t > 0.0).then(|| (kind.to_string(), t))
        })
        .collect();
    (!temperatures.is_empty()).then(|| Calibration::from_file(TEMPERATURES, temperatures))
}

fn positive(value: &Value, at: &str) -> Result<f64, String> {
    match as_float(value).filter(|_| !value.is_boolean()) {
        Some(t) if t.is_finite() && t > 0.0 => Ok(t),
        _ => Err(format!("{at} must be a finite number > 0, got {value}")),
    }
}

/// The decider's temperatures (`decider.temperature.from_config`): each
/// type takes `temperature_by_type[type]`, else `temperature`, else 1.
/// `None` when the config sets neither key.
///
/// # Errors
///
/// The key and value when a temperature is not a finite number > 0, or
/// `temperature_by_type` is not a map of the three answer types.
pub fn decider_temperatures(cfg: &Value, source: &str) -> Result<Option<Calibration>, String> {
    let base = cfg.get("temperature").filter(|v| !v.is_null());
    let by_type = cfg.get("temperature_by_type").filter(|v| !v.is_null());
    if base.is_none() && by_type.is_none() {
        return Ok(None);
    }
    let t = match base {
        Some(v) => positive(v, &format!("{source} \"temperature\""))?,
        None => 1.0,
    };
    let mut temperatures: BTreeMap<String, f64> =
        TYPES.iter().map(|k| (k.to_string(), t)).collect();
    if let Some(map) = by_type {
        let where_ = format!("{source} \"temperature_by_type\"");
        let map = map.as_object().ok_or_else(|| {
            format!("{where_} must be a map {{answer type: temperature}}, got {map}")
        })?;
        for (kind, value) in map {
            if !TYPES.contains(&kind.as_str()) {
                return Err(format!(
                    "{where_} has the unknown key '{kind}'; the keys are \"choice\", \"noul\" \
                     and \"score\""
                ));
            }
            if !value.is_number() {
                return Err(format!(
                    "{where_}['{kind}'] must be a finite number > 0, got {value}"
                ));
            }
            temperatures.insert(
                kind.clone(),
                positive(value, &format!("{where_}['{kind}']"))?,
            );
        }
    }
    Ok(Some(Calibration::from_file(source, temperatures)))
}

/// The lowercase hex SHA-256 of a chat template's source, as Jebadiah's
/// `template_sha256` hashes it.
pub fn template_sha256(source: &str) -> String {
    Sha256::digest(source.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The YAML front matter of the model card `README.md` in `dir`.
fn card_front_matter(dir: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(dir.join("README.md")).ok()?;
    let rest = text.strip_prefix("---")?;
    let end = rest.find("\n---")?;
    serde_saphyr::from_str::<Value>(&rest[..end]).ok()
}

/// The first string of a front matter value: the string, or a list's first.
fn first_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Array(items) => items.iter().find_map(first_string),
        _ => None,
    }
}

/// The license and base model: the model card's front matter in `dir`,
/// else the GGUF header's `general.license` and `general.base_model.0.*`.
pub fn license_and_base(
    dir: Option<&Path>,
    gguf: Option<&GgufMetadata>,
) -> (Option<String>, Option<String>) {
    let card = dir.and_then(card_front_matter);
    let from_card = |key: &str| {
        card.as_ref()
            .and_then(|c| c.get(key))
            .and_then(first_string)
    };
    let license = from_card("license")
        .or_else(|| gguf.and_then(|m| m.get_string("general.license").map(str::to_string)));
    let base = from_card("base_model").or_else(|| {
        let m = gguf?;
        m.get_string("general.base_model.0.repo_url")
            .or_else(|| m.get_string("general.base_model.0.name"))
            .map(str::to_string)
    });
    (license, base)
}

/// The positions the model addresses: `config.json`'s
/// `max_position_embeddings` (top level or `text_config`) in `dir`, else
/// the GGUF header's `<arch>.context_length`.
pub fn context_length(dir: Option<&Path>, gguf: Option<&GgufMetadata>) -> Option<u64> {
    let from_config = dir
        .and_then(|d| read_json_opt(&d.join("config.json")).ok().flatten())
        .and_then(|c| {
            c.get("max_position_embeddings")
                .or_else(|| c.get("text_config")?.get("max_position_embeddings"))
                .and_then(Value::as_u64)
        });
    from_config.or_else(|| {
        let m = gguf?;
        let key = format!("{}.context_length", m.architecture()?);
        m.get(&key)
            .and_then(|v| v.as_i64())
            .and_then(|n| u64::try_from(n).ok())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decider_temperatures_follow_from_config() {
        let cfg = json!({"temperature": 1.099,
            "temperature_by_type": {"choice": 1.11, "noul": 1.56, "score": 1.287}});
        let cal = decider_temperatures(&cfg, DECIDER_CONFIG)
            .expect("valid")
            .expect("set");
        assert_eq!(cal.temperatures["noul"], 1.56);
        assert_eq!(
            cal.policy(),
            TemperaturePolicy::ByKind {
                choice: 1.11,
                noul: 1.56,
                score: 1.287
            }
        );
        let only_t = decider_temperatures(&json!({"temperature": "1.5"}), DECIDER_CONFIG)
            .expect("valid")
            .expect("set");
        assert_eq!(only_t.temperatures["score"], 1.5);
        assert_eq!(decider_temperatures(&json!({}), DECIDER_CONFIG), Ok(None));
        for bad in [
            json!({"temperature": 0}),
            json!({"temperature": true}),
            json!({"temperature_by_type": {"bool": 1.0}}),
            json!({"temperature_by_type": {"noul": "2"}}),
            json!({"temperature_by_type": [1.0]}),
        ] {
            assert!(decider_temperatures(&bad, DECIDER_CONFIG).is_err(), "{bad}");
        }
    }

    #[test]
    fn one_temperature_for_every_type_is_a_fixed_policy() {
        let one = Calibration::from_file(
            "head.pt",
            TYPES.iter().map(|k| (k.to_string(), 2.4)).collect(),
        );
        assert_eq!(one.policy(), TemperaturePolicy::Fixed(2.4));
    }

    #[test]
    fn jebadiah_temperatures_skip_what_is_not_a_temperature() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert_eq!(jebadiah_temperatures(dir.path()), None);
        std::fs::write(
            dir.path().join(TEMPERATURES),
            r#"{"temperatures": {"choice": 1.1167, "noul": true, "score": -1, "other": 2}}"#,
        )
        .expect("write");
        let cal = jebadiah_temperatures(dir.path()).expect("one type set");
        assert_eq!(cal.temperatures.len(), 1);
        assert_eq!(
            cal.policy(),
            TemperaturePolicy::ByKind {
                choice: 1.1167,
                noul: 1.0,
                score: 1.0
            }
        );
    }

    #[test]
    fn template_hash_and_card() {
        assert_eq!(
            template_sha256("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(
            dir.path().join("README.md"),
            "---\nlicense: apache-2.0\nbase_model:\n- Qwen/Qwen3.5-4B\ntags: [a]\n---\n# Card\n",
        )
        .expect("write");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"text_config": {"max_position_embeddings": 262144}}"#,
        )
        .expect("write");
        assert_eq!(
            license_and_base(Some(dir.path()), None),
            (
                Some("apache-2.0".to_string()),
                Some("Qwen/Qwen3.5-4B".to_string())
            )
        );
        assert_eq!(context_length(Some(dir.path()), None), Some(262144));
    }
}
