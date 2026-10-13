//! `blazr_calibration.json`: the temperatures `blazr decision calibrate`
//! fitted on a user's labelled data. It sits in the sidecar directory and
//! sets each answer type it holds over the vendor's temperature
//! (`decider_config.json`, `temperatures.json` or Kev's `head.pt`); a type
//! it lacks keeps the vendor's. Vendor files are never written.

use std::collections::BTreeMap;
use std::path::Path;

use boostr::inference::decision::TemperatureBounds;
use serde::{Deserialize, Serialize};

use super::detect::DecisionSource;
use super::family_files::decider_config;
use super::kev::kev_calibration;
use super::sidecar::{
    decider_temperatures, jebadiah_temperatures, read_json_opt, Calibration, TYPES,
};
use crate::model::decision_template::Family;

/// The file name, in the sidecar directory.
pub const BLAZR_CALIBRATION: &str = "blazr_calibration.json";
/// The `schema` string of the format this build reads and writes.
pub const CALIBRATION_SCHEMA: &str = "blazr.decision.calibration.v1";
/// The `source` string the command writes.
pub const CALIBRATION_WRITER: &str = "blazr decision calibrate";

/// One answer type's fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeFit {
    /// Labelled answers the fit used.
    pub n: usize,
    /// Mean NLL of the gold answers at temperature 1.
    pub nll_before: f64,
    /// Mean NLL at `t`.
    pub nll_after: f64,
    /// The fitted temperature.
    pub t: f64,
    /// How the fit ended: `converged`, `bound`, `flat` or `iteration_limit`.
    pub outcome: String,
}

/// The file's contents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationFile {
    /// [`CALIBRATION_SCHEMA`].
    pub schema: String,
    /// The family the model was served as: `decider` or `jebadiah`.
    pub family: String,
    /// The UTC date of the fit, `YYYY-MM-DD`.
    pub fitted_at: String,
    /// The model name the fit ran on.
    pub model: String,
    /// The fitted temperature of each answer type with enough labels.
    pub temperatures: BTreeMap<String, f64>,
    /// Each fitted type's fit.
    pub fits: BTreeMap<String, TypeFit>,
    /// [`CALIBRATION_WRITER`].
    pub source: String,
}

impl CalibrationFile {
    /// Check the file read from `path` for a model of `family`: the schema,
    /// the family, the date, and temperatures that are answer types inside
    /// `bounds`.
    ///
    /// # Errors
    ///
    /// The file and the field that fails, and how to fix it.
    pub fn check(
        &self,
        path: &Path,
        family: Family,
        bounds: TemperatureBounds,
    ) -> Result<(), String> {
        let at = |field: &str| format!("{} \"{field}\"", path.display());
        if self.schema != CALIBRATION_SCHEMA {
            return Err(format!(
                "{} is '{}', this build reads '{CALIBRATION_SCHEMA}'. Run blazr decision \
                 calibrate again",
                at("schema"),
                self.schema
            ));
        }
        if self.family != family.name() {
            return Err(format!(
                "{} is '{}', the model is served as '{}'. Calibrate it as that family, or \
                 remove the file",
                at("family"),
                self.family,
                family.name()
            ));
        }
        if chrono::NaiveDate::parse_from_str(&self.fitted_at, "%Y-%m-%d").is_err() {
            return Err(format!(
                "{} is '{}', not a YYYY-MM-DD date",
                at("fitted_at"),
                self.fitted_at
            ));
        }
        if self.temperatures.is_empty() {
            return Err(format!(
                "{} sets no answer type. Remove the file or calibrate again",
                at("temperatures")
            ));
        }
        for (kind, &t) in &self.temperatures {
            if !TYPES.contains(&kind.as_str()) {
                return Err(format!(
                    "{} has the unknown key '{kind}'; the keys are \"choice\", \"noul\" and \
                     \"score\"",
                    at("temperatures")
                ));
            }
            if !(t.is_finite() && t >= bounds.min && t <= bounds.max) {
                return Err(format!(
                    "{}['{kind}'] is {t}, outside the fit range [{}, {}]",
                    at("temperatures"),
                    bounds.min,
                    bounds.max
                ));
            }
        }
        if let Some(kind) = self
            .fits
            .keys()
            .find(|k| !self.temperatures.contains_key(*k))
        {
            return Err(format!(
                "{}['{kind}'] has no temperature in \"temperatures\"",
                at("fits")
            ));
        }
        Ok(())
    }
}

/// The checked `blazr_calibration.json` of `dir` for a model of `family`.
/// `None` when there is none.
///
/// # Errors
///
/// When the file exists and does not read, parse or pass
/// [`CalibrationFile::check`] within the default fit range.
pub fn read_calibration_file(
    dir: &Path,
    family: Family,
) -> Result<Option<CalibrationFile>, String> {
    let path = dir.join(BLAZR_CALIBRATION);
    let Some(value) = read_json_opt(&path)? else {
        return Ok(None);
    };
    let file: CalibrationFile = serde_json::from_value(value).map_err(|e| {
        format!(
            "{}: {e}. Run blazr decision calibrate again",
            path.display()
        )
    })?;
    file.check(&path, family, TemperatureBounds::default())?;
    Ok(Some(file))
}

/// `vendor` with each type `file` sets taken from `file`. The source is
/// `blazr_calibration.json` and its fit date is kept.
pub fn prefer_calibration_file(
    vendor: Option<Calibration>,
    file: Option<&CalibrationFile>,
) -> Option<Calibration> {
    let Some(file) = file else {
        return vendor;
    };
    let mut merged = vendor.unwrap_or_else(|| Calibration::from_file("", BTreeMap::new()));
    for (kind, &t) in &file.temperatures {
        merged.temperatures.insert(kind.clone(), t);
        merged
            .sources
            .insert(kind.clone(), BLAZR_CALIBRATION.to_string());
    }
    merged.source = BLAZR_CALIBRATION.to_string();
    merged.fitted_at = Some(file.fitted_at.clone());
    Some(merged)
}

/// The temperatures the vendor ships for `source`: the decider config's,
/// Jebadiah's `temperatures.json`, or the one temperature of Kev's
/// `head.pt` for every type. `None` when there are none.
///
/// # Errors
///
/// When the decider config exists and does not parse, a temperature in it
/// is invalid, or Kev's `head.pt` does not open or holds an invalid
/// temperature.
pub fn vendor_calibration(source: &DecisionSource) -> Result<Option<Calibration>, String> {
    match source.family {
        Family::Decider => match decider_config(source)? {
            Some(cfg) => decider_temperatures(&cfg.value, &cfg.name()),
            None => Ok(None),
        },
        Family::Jebadiah => Ok(jebadiah_temperatures(&source.sidecar_dir)),
        Family::Kev => kev_calibration(&source.sidecar_dir),
    }
}

/// The temperatures `source` serves with: the vendor's, each type
/// `blazr_calibration.json` in the sidecar directory sets taken from it.
///
/// # Errors
///
/// As [`vendor_calibration`] and [`read_calibration_file`]: a bad file
/// stops the load, never falls back.
pub fn served_calibration(source: &DecisionSource) -> Result<Option<Calibration>, String> {
    let vendor = vendor_calibration(source)?;
    let file = read_calibration_file(&source.sidecar_dir, source.family)?;
    Ok(prefer_calibration_file(vendor, file.as_ref()))
}

/// Write `file` to `path` as indented JSON with a final newline.
///
/// # Errors
///
/// The path and the reason when it does not write.
pub fn write_calibration_file(path: &Path, file: &CalibrationFile) -> Result<(), String> {
    let mut text = serde_json::to_string_pretty(file)
        .map_err(|e| format!("the calibration does not serialize: {e}"))?;
    text.push('\n');
    std::fs::write(path, text).map_err(|e| format!("{} does not write: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn file() -> CalibrationFile {
        CalibrationFile {
            schema: CALIBRATION_SCHEMA.into(),
            family: "decider".into(),
            fitted_at: "2026-10-13".into(),
            model: "decider-4b".into(),
            temperatures: BTreeMap::from([("choice".to_string(), 1.25)]),
            fits: BTreeMap::from([(
                "choice".to_string(),
                TypeFit {
                    n: 80,
                    nll_before: 0.5,
                    nll_after: 0.4,
                    t: 1.25,
                    outcome: "converged".into(),
                },
            )]),
            source: CALIBRATION_WRITER.into(),
        }
    }

    #[test]
    fn round_trip_through_the_sidecar_directory() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert_eq!(read_calibration_file(dir.path(), Family::Decider), Ok(None));
        let path = dir.path().join(BLAZR_CALIBRATION);
        write_calibration_file(&path, &file()).expect("writes");
        let text = std::fs::read_to_string(&path).expect("reads");
        let keys: Vec<String> = serde_json::from_str::<serde_json::Value>(&text)
            .expect("json")
            .as_object()
            .expect("object")
            .keys()
            .cloned()
            .collect();
        assert_eq!(
            keys,
            [
                "schema",
                "family",
                "fitted_at",
                "model",
                "temperatures",
                "fits",
                "source"
            ]
        );
        let back = read_calibration_file(dir.path(), Family::Decider)
            .expect("valid")
            .expect("present");
        assert_eq!(back, file());
    }

    #[test]
    fn the_file_wins_per_type_and_vendor_values_stay() {
        let vendor = Calibration::from_file(
            "decider_config.json",
            BTreeMap::from([
                ("choice".to_string(), 1.11),
                ("noul".to_string(), 1.56),
                ("score".to_string(), 1.287),
            ]),
        );
        let merged = prefer_calibration_file(Some(vendor.clone()), Some(&file())).expect("set");
        assert_eq!(merged.source, BLAZR_CALIBRATION);
        assert_eq!(merged.fitted_at.as_deref(), Some("2026-10-13"));
        assert_eq!(merged.temperatures["choice"], 1.25);
        assert_eq!(merged.temperatures["noul"], 1.56);
        assert_eq!(merged.sources["choice"], BLAZR_CALIBRATION);
        assert_eq!(merged.sources["score"], "decider_config.json");
        assert_eq!(
            prefer_calibration_file(Some(vendor.clone()), None),
            Some(vendor)
        );
        let alone = prefer_calibration_file(None, Some(&file())).expect("set");
        assert_eq!(alone.temperatures.len(), 1);
        assert_eq!(alone.source, BLAZR_CALIBRATION);
    }

    #[test]
    fn bad_files_name_the_file_and_field() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join(BLAZR_CALIBRATION);
        let bounds = TemperatureBounds::default();
        let check = |f: CalibrationFile, family| f.check(&path, family, bounds);
        let mut f = file();
        f.schema = "other.v0".into();
        let e = check(f, Family::Decider).expect_err("schema");
        assert!(
            e.contains("\"schema\"") && e.contains(BLAZR_CALIBRATION),
            "{e}"
        );
        let e = check(file(), Family::Jebadiah).expect_err("family");
        assert!(e.contains("\"family\""), "{e}");
        for t in [0.0, -1.0, 0.01, 1000.0] {
            let mut f = file();
            f.temperatures.insert("noul".into(), t);
            let e = check(f, Family::Decider).expect_err("range");
            assert!(e.contains("\"temperatures\"['noul']"), "{e}");
        }
        let mut f = file();
        f.temperatures.insert("bool".into(), 1.0);
        assert!(check(f, Family::Decider)
            .expect_err("key")
            .contains("'bool'"));
        let mut f = file();
        f.temperatures.clear();
        assert!(check(f, Family::Decider).is_err());
        let mut f = file();
        f.fitted_at = "yesterday".into();
        assert!(check(f, Family::Decider)
            .expect_err("date")
            .contains("fitted_at"));
        let mut f = file();
        f.temperatures.remove("choice");
        f.temperatures.insert("noul".into(), 1.0);
        assert!(check(f, Family::Decider)
            .expect_err("fits")
            .contains("\"fits\""));

        std::fs::write(&path, json!({"schema": CALIBRATION_SCHEMA}).to_string()).expect("write");
        let e = read_calibration_file(dir.path(), Family::Decider).expect_err("missing fields");
        assert!(
            e.contains(BLAZR_CALIBRATION) && e.contains("missing field"),
            "{e}"
        );
        std::fs::write(&path, "{bad").expect("write");
        assert!(read_calibration_file(dir.path(), Family::Decider).is_err());
    }
}
