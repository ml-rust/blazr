//! Wire answers against the family's reference answers, field by field.
//!
//! Each value may differ by the run's probability tolerance carried through
//! the formula that produced it, plus one unit of the family's rounding:
//! both sides are rounded, so a value near a rounding edge can land one
//! unit apart. A choice must match where the reference's top two
//! probabilities are further apart than the tolerance's gate.

use std::path::Path;

use boostr::format::Gguf;
use serde_json::{Map, Value};

/// How far a run may sit from the reference.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    /// Each probability.
    pub prob: f64,
    /// The reference top-two gap above which the chosen option must match.
    pub gate: f64,
}

/// The formatter alone, fed the reference's own distributions.
pub const EXACT: Tolerance = Tolerance {
    prob: 0.0,
    gate: 0.0,
};

/// CPU, every weight F32, against the F32 reference: the probability band
/// of `tests/decision_loader/compare.rs`, where only the order of float
/// accumulation differs.
#[cfg_attr(any(feature = "cuda", feature = "wgpu"), allow(dead_code))]
pub const CPU_F32: Tolerance = Tolerance {
    prob: 1e-3,
    gate: 2e-3,
};

/// The same request merged with another against it alone: the same
/// weights, rows padded into different lanes.
pub const MERGED: Tolerance = Tolerance {
    prob: 2e-5,
    gate: 4e-5,
};

/// The tolerance of a GGUF build against the F32 reference, by
/// `general.file_type` or file name: the bands of
/// `tests/decision_loader/compare.rs`.
pub fn gguf_tolerance(path: &Path) -> Tolerance {
    let gguf = Gguf::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let file_type = gguf.metadata().get_u32("general.file_type");
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_uppercase())
        .unwrap_or_default();
    if file_type == Some(7) || name.contains("Q8_0") {
        Tolerance {
            prob: 0.1,
            gate: 0.2,
        }
    } else if file_type == Some(15) || name.contains("Q4_K_M") {
        Tolerance {
            prob: 0.3,
            gate: 0.6,
        }
    } else {
        panic!(
            "{}: general.file_type is {file_type:?}; the route tests hold tolerances for Q8_0 \
             and Q4_K_M only",
            path.display()
        )
    }
}

/// One unit of the family's rounding: probabilities and confidences, then
/// `score`. Jebadiah's server derives `confidence` and `score` from values
/// already rounded to 6 digits, its reference script from unrounded ones,
/// so its unit spans two digits' worth.
fn units(family: &str) -> (f64, f64) {
    match family {
        "decider" => (1.0e-4 + 1e-12, 1.0e-2 + 1e-12),
        _ => (2.0e-6, 2.0e-6),
    }
}

fn number(v: &Value, what: &str) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("{what}: {v} is not a number"))
}

fn close(got: &Value, want: &Value, band: f64, what: &str) {
    let (g, w) = (number(got, what), number(want, what));
    assert!(
        (g - w).abs() <= band,
        "{what}: {g} vs reference {w} (band {band})"
    );
}

/// The reference's top-two probability gap.
fn gap(probs: &Map<String, Value>, what: &str) -> f64 {
    let mut p: Vec<f64> = probs.values().map(|v| number(v, what)).collect();
    p.sort_by(|a, b| b.total_cmp(a));
    p[0] - p.get(1).copied().unwrap_or(0.0)
}

fn probabilities(got: &Value, want: &Value, band: f64, what: &str) -> Map<String, Value> {
    let g = got
        .as_object()
        .unwrap_or_else(|| panic!("{what}: no probabilities"));
    let w = want
        .as_object()
        .unwrap_or_else(|| panic!("{what}: reference has none"));
    let gk: Vec<&String> = g.keys().collect();
    let wk: Vec<&String> = w.keys().collect();
    assert_eq!(gk, wk, "{what}: probability keys");
    for (k, wv) in w {
        close(&g[k], wv, band, &format!("{what} p[{k}]"));
    }
    w.clone()
}

/// One answer against the reference's.
fn answer(family: &str, got: &Value, want: &Value, tol: Tolerance, what: &str) {
    let (unit, score_unit) = units(family);
    let band = tol.prob + unit;
    assert_eq!(got["type"], want["type"], "{what}: type");
    let keys = |v: &Value| -> Vec<String> {
        v.as_object()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    };
    assert_eq!(keys(got), keys(want), "{what}: fields");
    let decider = family == "decider";
    match want["type"].as_str() {
        Some("noul") => close(&got["noul"], &want["noul"], band, &format!("{what} noul")),
        Some(kind @ ("choice" | "score")) => {
            let p = probabilities(&got["probabilities"], &want["probabilities"], band, what);
            let gated = gap(&p, what) > tol.gate;
            let levels = p.len() as f64;
            if kind == "choice" {
                if gated {
                    assert_eq!(got["choice"], want["choice"], "{what}: choice");
                }
                close(
                    &got["confidence"],
                    &want["confidence"],
                    2.0 * tol.prob + unit,
                    what,
                );
            } else {
                assert_eq!(got["legend"], want["legend"], "{what}: legend");
                let spread: f64 = (0..p.len()).map(|i| i as f64).sum::<f64>() * band;
                close(
                    &got["score"],
                    &want["score"],
                    score_unit + spread,
                    &format!("{what} score"),
                );
                let confidence_band = if decider { 10.0 } else { 2.0 } * tol.prob + unit;
                if gated || tol.prob == 0.0 {
                    close(
                        &got["confidence"],
                        &want["confidence"],
                        confidence_band,
                        what,
                    );
                }
            }
            if decider {
                close(
                    &got["x_p_max"],
                    &want["x_p_max"],
                    band,
                    &format!("{what} x_p_max"),
                );
                let certainty_band = (10.0 * tol.prob).min(1.0) + unit;
                close(
                    &got["certainty"],
                    &want["certainty"],
                    certainty_band,
                    &format!("{what} certainty"),
                );
            }
            if want.get("fit_mass").is_some() {
                probabilities(
                    &got["level_fit"],
                    &want["level_fit"],
                    band,
                    &format!("{what} level_fit"),
                );
                close(
                    &got["fit_mass"],
                    &want["fit_mass"],
                    levels * band,
                    &format!("{what} fit_mass"),
                );
            }
        }
        other => panic!("{what}: unknown answer type {other:?}"),
    }
}

/// Every answer of `got` against `want`, in order.
pub fn answers(
    family: &str,
    got: &Map<String, Value>,
    want: &Map<String, Value>,
    tol: Tolerance,
    what: &str,
) {
    let gk: Vec<&String> = got.keys().collect();
    let wk: Vec<&String> = want.keys().collect();
    assert_eq!(gk, wk, "{what}: question ids");
    for (id, w) in want {
        answer(family, &got[id], w, tol, &format!("{what} question {id}"));
    }
}
