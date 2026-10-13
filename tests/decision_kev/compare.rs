//! Kev wire answers against Kev's own, field by field.
//!
//! Both sides round to 4 digits, so each value may differ by the run's
//! probability band carried through its formula, plus one rounding unit:
//!
//! - `noul`, each probability: the band.
//! - Choice `confidence` `(p_max - 1/K) / (1 - 1/K)`: the band times
//!   `K / (K - 1)`, at most twice the band.
//! - Score `score` `sum(i * p_i)`: the band times `sum(i)`.
//! - Score `confidence` `1 - E|i - mode| / D`: the band times
//!   `sum|i - mode| / D`, when the mode is not in doubt.
//!
//! A choice, or a Score's mode, must match where the reference's top two
//! probabilities are further apart than twice the band.

use serde_json::{Map, Value};

/// One rounding unit of Kev's 4 digits, with room for the binary value.
const UNIT: f64 = 1e-4 + 1e-12;

fn num(v: &Value, what: &str) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("{what}: {v} is not a number"))
}

fn close(got: f64, want: f64, band: f64, what: &str) {
    assert!(
        (got - want).abs() <= band + UNIT,
        "{what}: {got} vs {want}, band {band} plus one rounding unit"
    );
}

/// The reference probabilities in key order, and the gap between its top
/// two.
fn probs(want: &Value, keys: &[String], what: &str) -> (Vec<f64>, f64) {
    let p: Vec<f64> = keys
        .iter()
        .map(|k| num(&want["probabilities"][k], &format!("{what} p[{k}]")))
        .collect();
    let mut sorted = p.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let gap = if sorted.len() > 1 {
        sorted[0] - sorted[1]
    } else {
        1.0
    };
    (p, gap)
}

/// The keys of an answer's `probabilities`, in order.
fn keys_of(answer: &Value, what: &str) -> Vec<String> {
    answer["probabilities"]
        .as_object()
        .unwrap_or_else(|| panic!("{what}: no probabilities in {answer}"))
        .keys()
        .cloned()
        .collect()
}

/// `got` against `want` for one question, within `band`.
fn answer(got: &Value, want: &Value, band: f64, what: &str) {
    assert_eq!(got["type"], want["type"], "{what}: type");
    match want["type"].as_str() {
        Some("noul") => close(
            num(&got["noul"], what),
            num(&want["noul"], what),
            band,
            what,
        ),
        Some("choice") => {
            let keys = keys_of(want, what);
            assert_eq!(keys_of(got, what), keys, "{what}: option keys");
            let (p, gap) = probs(want, &keys, what);
            for (k, w) in keys.iter().zip(&p) {
                close(
                    num(&got["probabilities"][k], what),
                    *w,
                    band,
                    &format!("{what} p[{k}]"),
                );
            }
            let k = keys.len() as f64;
            let conf_band = if k > 1.0 { band * k / (k - 1.0) } else { 0.0 };
            close(
                num(&got["confidence"], what),
                num(&want["confidence"], what),
                conf_band,
                &format!("{what} confidence"),
            );
            if gap > 2.0 * band {
                assert_eq!(got["choice"], want["choice"], "{what}: choice");
            }
        }
        Some("score") => {
            assert_eq!(got["legend"], want["legend"], "{what}: legend");
            let keys = keys_of(want, what);
            assert_eq!(keys_of(got, what), keys, "{what}: level keys");
            let (p, gap) = probs(want, &keys, what);
            for (k, w) in keys.iter().zip(&p) {
                close(
                    num(&got["probabilities"][k], what),
                    *w,
                    band,
                    &format!("{what} p[{k}]"),
                );
            }
            let levels = p.len();
            let index_sum = (levels * levels.saturating_sub(1) / 2) as f64;
            close(
                num(&got["score"], what),
                num(&want["score"], what),
                band * index_sum,
                &format!("{what} score"),
            );
            if gap > 2.0 * band && levels > 1 {
                let mode = p
                    .iter()
                    .enumerate()
                    .fold(0, |best, (i, &x)| if x > p[best] { i } else { best });
                let mid = (levels as f64 - 1.0) / 2.0;
                let d = (0..levels).map(|i| (i as f64 - mid).abs()).sum::<f64>() / levels as f64;
                let reach = (0..levels).map(|i| i.abs_diff(mode) as f64).sum::<f64>() / d;
                close(
                    num(&got["confidence"], what),
                    num(&want["confidence"], what),
                    band * reach,
                    &format!("{what} confidence"),
                );
            }
        }
        other => panic!("{what}: unknown answer type {other:?}"),
    }
}

/// Every answer of `got` against `want`, the same ids in the same order.
pub fn answers(got: &Map<String, Value>, want: &Map<String, Value>, band: f64, what: &str) {
    let got_ids: Vec<&String> = got.keys().collect();
    let want_ids: Vec<&String> = want.keys().collect();
    assert_eq!(got_ids, want_ids, "{what}: question ids");
    for (id, w) in want {
        answer(&got[id], w, band, &format!("{what} {id}"));
    }
}
