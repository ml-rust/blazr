//! Clef wire answers against the reference's, field by field.
//!
//! Both sides round to 4 digits, so each value may differ by the run's
//! probability band carried through its formula, plus one rounding unit:
//!
//! - `noul`, each probability, each `confidence`: the band.
//! - Score `score` `sum(i * p_i)`: the band times `sum(i)`.
//!
//! A `choice` must match where the reference's top two probabilities are
//! further apart than twice the band. Keys, their order, types and legends
//! must match exactly.

use serde_json::{Map, Value};

/// One rounding unit of 4 digits, with room for the binary value.
const UNIT: f64 = 1e-4 + 1e-12;

fn num(v: &Value, what: &str) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("{what}: {v} is not a number"))
}

fn close(got: &Value, want: &Value, band: f64, what: &str) {
    let (g, w) = (num(got, what), num(want, what));
    assert!(
        (g - w).abs() <= band + UNIT,
        "{what}: {g} vs {w}, band {band} plus one rounding unit"
    );
}

fn keys(v: &Value, what: &str) -> Vec<String> {
    v.as_object()
        .unwrap_or_else(|| panic!("{what}: {v} is not an object"))
        .keys()
        .cloned()
        .collect()
}

/// One answer within `band`.
fn answer(got: &Value, want: &Value, band: f64, what: &str) {
    assert_eq!(keys(got, what), keys(want, what), "{what}: fields");
    assert_eq!(got["type"], want["type"], "{what}: type");
    match want["type"].as_str() {
        Some("noul") => close(&got["noul"], &want["noul"], band, what),
        Some(kind @ ("choice" | "score")) => {
            let names = keys(&want["probabilities"], what);
            assert_eq!(
                keys(&got["probabilities"], what),
                names,
                "{what}: probability keys and order"
            );
            let mut p: Vec<f64> = Vec::with_capacity(names.len());
            for k in &names {
                let at = format!("{what} p[{k}]");
                close(
                    &got["probabilities"][k],
                    &want["probabilities"][k],
                    band,
                    &at,
                );
                p.push(num(&want["probabilities"][k], &at));
            }
            close(&got["confidence"], &want["confidence"], band, what);
            if kind == "choice" {
                let mut sorted = p.clone();
                sorted.sort_by(|a, b| b.total_cmp(a));
                let gap = if sorted.len() > 1 {
                    sorted[0] - sorted[1]
                } else {
                    1.0
                };
                if gap > 2.0 * band {
                    assert_eq!(got["choice"], want["choice"], "{what}: choice");
                }
            } else {
                let spread = (0..p.len()).sum::<usize>() as f64;
                close(&got["score"], &want["score"], band * spread, what);
                assert_eq!(got["legend"], want["legend"], "{what}: legend");
            }
        }
        other => panic!("{what}: unknown answer type {other:?}"),
    }
}

/// Every answer of `got` against `want`, in the same order, within `band`.
pub fn answers(got: &Map<String, Value>, want: &Map<String, Value>, band: f64, what: &str) {
    let order = |m: &Map<String, Value>| m.keys().cloned().collect::<Vec<_>>();
    assert_eq!(order(got), order(want), "{what}: question order");
    for (id, w) in want {
        answer(&got[id], w, band, &format!("{what} {id}"));
    }
}
