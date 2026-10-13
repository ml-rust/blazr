//! `blazr decision calibrate` against the decider's own calibrator,
//! `decider/calibrate.py` `fit_by_type`, on the same precomputed records.
//!
//! # Fixtures
//!
//! - `BLAZR_DECISION_MODEL_DIR`: the decider checkpoint directory. The
//!   command detects the family and the sidecar directory from it; a
//!   records-only label file loads no weights.
//! - `BLAZR_CALIBRATE_REF`: `reference.json` from
//!   `~/.cache/s1-ref/scripts/ref_calibrate.py`, which writes the records
//!   (`records.jsonl`, from the decider fixtures' raw logits with
//!   deterministic golds) and the reference fits on them:
//!
//!   ```text
//!   python -I ~/.cache/s1-ref/scripts/ref_calibrate.py \
//!       "$BLAZR_DECISION_MODEL_DIR" ~/.cache/s1-ref/fixtures/decider \
//!       ~/.cache/s1-ref/calibrate
//!   BLAZR_CALIBRATE_REF=~/.cache/s1-ref/calibrate/reference.json
//!   ```
//!
//! Unset or missing, the test prints one `skip:` line and passes.
//!
//! # Tolerance
//!
//! Both sides minimise the same objective: the mean NLL of the gold under
//! `softmax(logits / T)`, and for an isolated Score answer under the
//! per-level yes probabilities normalised to sum 1, pooled over a type's
//! listed and isolated answers. The NLL at T = 1 matches to 1e-9 relative,
//! which checks that. The fitted T matches to 1e-3 relative, and blazr's
//! NLL at its T is no higher than the reference's NLL at the reference's T.

#![cfg(all(feature = "cli", feature = "server"))]

use std::path::PathBuf;

use blazr::boostr::inference::decision::isolated::evaluate_mixed;
use blazr::boostr::inference::decision::objective::evaluate;
use blazr::boostr::inference::decision::TemperatureBounds;
use blazr::engine::decision::calibration::{
    fit_types, parse_label_lines, records_only, AnswerType, TypeSamples,
};
use serde_json::Value;

const ENV_MODEL: &str = "BLAZR_DECISION_MODEL_DIR";
const ENV_REF: &str = "BLAZR_CALIBRATE_REF";

fn env_path(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os(name)?);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skip: {name}={} does not exist", path.display());
        None
    }
}

/// The mean NLL of `kind`'s answers at temperature `t`.
fn nll_at(samples: &TypeSamples, kind: AnswerType, t: f64) -> f64 {
    let beta = 1.0 / t;
    match kind {
        AnswerType::Choice => evaluate(&samples.choice, beta).nll,
        AnswerType::Noul => evaluate(&samples.noul, beta).nll,
        AnswerType::Score => evaluate_mixed(&samples.score, &samples.score_isolated, beta).nll,
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn calibrate_matches_the_reference_fit_by_type() {
    let (Some(model), Some(reference)) = (env_path(ENV_MODEL), env_path(ENV_REF)) else {
        eprintln!("skip: set {ENV_MODEL} and {ENV_REF} (see the module docs)");
        return;
    };
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string(&reference).expect("reference reads"))
            .expect("reference is JSON");
    let records = PathBuf::from(reference["records"].as_str().expect("records path"));
    let min_rows = reference["min_rows"].as_u64().expect("min_rows") as usize;
    let text = std::fs::read_to_string(&records).expect("records read");

    let lines = parse_label_lines(&text).expect("records parse");
    let collected = records_only(&lines).expect("records only");
    let reports =
        fit_types(&collected.samples, min_rows, TemperatureBounds::default()).expect("fits");

    let unrounded = reference["unrounded"].as_object().expect("unrounded fits");
    for report in &reports {
        let kind = report.kind.name();
        let Some(want) = unrounded.get(kind) else {
            assert!(
                report.fit.is_none(),
                "{kind}: the reference left it unfitted"
            );
            continue;
        };
        let fit = report.fit.expect("fitted");
        let t_ref = want["t"].as_f64().expect("t");
        assert_eq!(report.n as u64, want["n"].as_u64().expect("n"), "{kind}: n");
        let before = want["nll_t1"].as_f64().expect("nll_t1");
        assert!(
            rel(fit.nll_before, before) < 1e-9,
            "{kind}: NLL at T=1 {} vs reference {before}",
            fit.nll_before
        );
        let at_ref = nll_at(&collected.samples, report.kind, t_ref);
        let ref_after = want["nll_fit"].as_f64().expect("nll_fit");
        assert!(
            rel(at_ref, ref_after) < 1e-9,
            "{kind}: NLL at the reference T {at_ref} vs reference {ref_after}"
        );
        assert!(
            fit.nll_after <= ref_after + 1e-12,
            "{kind}: blazr's NLL {} is above the reference's {ref_after}",
            fit.nll_after
        );
        assert!(
            rel(fit.t, t_ref) < 1e-3,
            "{kind}: T {} vs reference {t_ref}",
            fit.t
        );
        let rounded = reference["fit_by_type"]["temperature_by_type"][kind]
            .as_f64()
            .expect("rounded T");
        assert!(
            (fit.t - rounded).abs() <= 5e-4 + 1e-9,
            "{kind}: rounded T {rounded}"
        );
        eprintln!("{kind}: n {} T {} reference {t_ref}", report.n, fit.t);
    }

    // The command end to end on the same file: it writes what the library fits.
    let out_dir = tempfile::tempdir().expect("temp dir");
    let out = out_dir.path().join("blazr_calibration.json");
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_blazr"))
        .args(["decision", "calibrate", "--model"])
        .arg(&model)
        .arg("--labels")
        .arg(&records)
        .args(["--min-rows", &min_rows.to_string(), "--out"])
        .arg(&out)
        .status()
        .expect("blazr runs");
    assert!(status.success(), "blazr decision calibrate: {status}");
    let written: Value =
        serde_json::from_str(&std::fs::read_to_string(&out).expect("written")).expect("JSON");
    assert_eq!(written["schema"], "blazr.decision.calibration.v1");
    assert_eq!(written["family"], "decider");
    assert_eq!(written["source"], "blazr decision calibrate");
    for report in &reports {
        let kind = report.kind.name();
        let t = written["temperatures"][kind].as_f64();
        assert_eq!(t, report.fit.map(|f| f.t), "{kind}: written T");
        if let Some(fit) = report.fit {
            assert_eq!(written["fits"][kind]["n"].as_u64(), Some(report.n as u64));
            assert_eq!(written["fits"][kind]["t"].as_f64(), Some(fit.t));
        }
    }
}
