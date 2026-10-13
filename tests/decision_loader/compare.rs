//! Tolerances, and every fixture through `run_decision` compared with its
//! reference: raw label logits, probabilities and argmax. The bands are
//! those of boostr's `decision_label_token_real_file` test, which derives
//! them.

use std::path::Path;

use boostr::format::Gguf;
use boostr::inference::decision::{
    run_decision, tempered_softmax, Answer, AnswerOptions, DecisionResult, RunnerConfig,
};
use boostr::model::decision::LabelTokenHead;
use boostr::model::{ModelClient, Qwen35Model};
use boostr::{DType, Runtime};

use super::fixture::{request, Fixture};

/// Where the argmax of a run must equal the reference's.
#[derive(Clone, Copy)]
pub enum ArgmaxGate {
    /// Everywhere.
    Always,
    /// Where the reference's top two are further apart than twice the
    /// tolerance of the compared values.
    #[cfg_attr(not(all(feature = "cuda", feature = "f16")), allow(dead_code))]
    TwiceTolerance,
    /// Where the reference's top two probabilities differ by more than this.
    ProbGap(f64),
}

/// How close the run must come to the reference.
#[derive(Clone, Copy)]
pub struct Tolerance {
    pub raw: f32,
    pub prob: f64,
    pub argmax: ArgmaxGate,
}

impl Tolerance {
    fn gates(&self, gap: f64, band: f64, prob_gap: f64) -> bool {
        match self.argmax {
            ArgmaxGate::Always => true,
            ArgmaxGate::TwiceTolerance => gap > 2.0 * band,
            ArgmaxGate::ProbGap(min) => prob_gap > min,
        }
    }
}

/// CPU, every weight F32, against the F32 reference. Both sides run the same
/// F32 weights, so only the order of float accumulation differs.
pub const CPU_F32: Tolerance = Tolerance {
    raw: 5e-3,
    prob: 1e-3,
    argmax: ArgmaxGate::Always,
};

/// CUDA, the checkpoint in BF16.
#[cfg(all(feature = "cuda", feature = "f16"))]
pub const CUDA_BF16: Tolerance = Tolerance {
    raw: 0.5,
    prob: 0.1,
    argmax: ArgmaxGate::TwiceTolerance,
};

/// The llama.cpp file types the GGUF tests hold tolerances for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// `general.file_type` 7.
    Q8_0,
    /// `general.file_type` 15.
    Q4KM,
}

impl Quant {
    /// The quant `general.file_type` names, else the one the file name
    /// carries. Panics when neither names Q8_0 or Q4_K_M.
    pub fn of(path: &Path) -> Self {
        let gguf = Gguf::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let file_type = gguf.metadata().get_u32("general.file_type");
        match file_type {
            Some(7) => return Self::Q8_0,
            Some(15) => return Self::Q4KM,
            _ => {}
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_ascii_uppercase())
            .unwrap_or_default();
        if name.contains("Q8_0") {
            Self::Q8_0
        } else if name.contains("Q4_K_M") {
            Self::Q4KM
        } else {
            panic!(
                "{}: general.file_type is {file_type:?} and the file name names no quant; \
                 the decision tests hold tolerances for Q8_0 and Q4_K_M only",
                path.display()
            )
        }
    }

    /// The tolerance of this build against the F32 reference.
    pub fn tolerance(self) -> Tolerance {
        match self {
            Self::Q8_0 => Tolerance {
                raw: 0.5,
                prob: 0.1,
                argmax: ArgmaxGate::ProbGap(0.2),
            },
            Self::Q4KM => Tolerance {
                raw: 5.0,
                prob: 0.3,
                argmax: ArgmaxGate::ProbGap(0.6),
            },
        }
    }
}

/// The first index of the largest value, and the gap to the runner-up.
fn top<T: Copy + Into<f64>>(values: &[T]) -> (usize, f64) {
    let v: Vec<f64> = values.iter().map(|&x| x.into()).collect();
    let best = (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    let second = (0..v.len())
        .filter(|&i| i != best)
        .map(|i| v[i])
        .fold(f64::NEG_INFINITY, f64::max);
    (best, v[best] - second)
}

/// The argmax of `got` equals the reference's where `tol` gates it.
/// Returns whether it was compared.
fn same_argmax<T: Copy + Into<f64>>(
    got: &[T],
    want: &[T],
    band: f64,
    prob_gap: f64,
    tol: Tolerance,
    what: &str,
) -> bool {
    let (g, _) = top(got);
    let (w, gap) = top(want);
    let gated = tol.gates(gap, band, prob_gap);
    if gated {
        assert_eq!(
            g, w,
            "{what}: argmax {g} vs reference {w} (reference gap {gap})"
        );
    }
    gated
}

fn close_f64(got: &[f64], want: &[f64], tol: f64, what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(d <= tol, "{what} entry {i}: {g} vs reference {w}");
        worst = worst.max(d);
    }
    worst
}

/// Every raw logit, probability and argmax of `out` against `fx`.
fn compare(fx: &Fixture, out: &DecisionResult, tol: Tolerance) {
    let what = fx.label();
    assert_eq!(out.raw_logits.len(), fx.scoring_rows.len(), "{what}: rows");
    let (mut raw_worst, mut prob_worst) = (0.0f32, 0.0f64);
    let (mut checked, mut total) = (0usize, 0usize);
    for (s, sr) in fx.scoring_rows.iter().enumerate() {
        let here = format!("{what} scoring row {s}");
        assert!(
            (out.temperatures[s] - sr.temperature).abs() < 1e-12,
            "{here}: temperature {} vs reference {}",
            out.temperatures[s],
            sr.temperature
        );
        let got = &out.raw_logits[s];
        assert_eq!(got.len(), sr.raw_logits.len(), "{here}: logit count");
        for (k, (g, w)) in got.iter().zip(&sr.raw_logits).enumerate() {
            let d = (g - w).abs();
            assert!(
                d <= tol.raw,
                "{here} label {k}: raw logit {g} vs reference {w}"
            );
            raw_worst = raw_worst.max(d);
        }
        total += 1;
        let prob_gap = top(sr.probs.as_slice()).1;
        checked += usize::from(same_argmax(
            got.as_slice(),
            sr.raw_logits.as_slice(),
            f64::from(tol.raw),
            prob_gap,
            tol,
            &here,
        ));
        let probs = tempered_softmax(got, sr.temperature)
            .unwrap_or_else(|e| panic!("{here}: tempered_softmax: {e}"));
        prob_worst = prob_worst.max(close_f64(&probs, &sr.probs, tol.prob, &here));
    }
    assert_eq!(out.answers.len(), fx.questions.len(), "{what}: answers");
    for (answer, q) in out.answers.iter().zip(&fx.questions) {
        let here = format!("{what} question {}", q.id);
        total += 1;
        match answer {
            Answer::Choice { index, probs, .. } => {
                assert_eq!(q.kind, "choice", "{here}: kind");
                prob_worst = prob_worst.max(close_f64(probs, &q.probs, tol.prob, &here));
                let (w, gap) = top(q.probs.as_slice());
                if tol.gates(gap, tol.prob, gap) {
                    checked += 1;
                    assert_eq!(*index, w, "{here}: choice {index} vs reference {w}");
                }
            }
            Answer::Noul { p_yes, .. } => {
                assert_eq!(q.kind, "noul", "{here}: kind");
                let want = q.p_yes.unwrap_or_else(|| panic!("{here}: no p_yes"));
                prob_worst = prob_worst.max(close_f64(&[*p_yes], &[want], tol.prob, &here));
                let gap = (2.0 * want - 1.0).abs();
                if tol.gates(gap, tol.prob, gap) {
                    checked += 1;
                    assert_eq!(
                        *p_yes >= 0.5,
                        want >= 0.5,
                        "{here}: p_yes {p_yes} vs {want}"
                    );
                }
            }
            Answer::Score {
                probs, fit_mass, ..
            } => {
                assert_eq!(q.kind, "score", "{here}: kind");
                prob_worst = prob_worst.max(close_f64(probs, &q.probs, tol.prob, &here));
                let prob_gap = top(q.probs.as_slice()).1;
                checked += usize::from(same_argmax(
                    probs.as_slice(),
                    q.probs.as_slice(),
                    tol.prob,
                    prob_gap,
                    tol,
                    &here,
                ));
                match (fit_mass, q.fit_mass) {
                    (Some(g), Some(w)) => {
                        let d = (g - w).abs();
                        let band = tol.prob * q.options as f64;
                        assert!(d <= band, "{here}: fit mass {g} vs {w}");
                    }
                    (None, None) => {}
                    (g, w) => panic!("{here}: fit mass {g:?} vs reference {w:?}"),
                }
            }
            other => panic!("{here}: unexpected answer {other:?}"),
        }
    }
    println!(
        "{what}: {} rows, {} scoring rows, prefix {}: max raw diff {raw_worst}, \
         max prob diff {prob_worst}, argmax compared {checked} of {total}",
        fx.rows.len(),
        fx.scoring_rows.len(),
        fx.shared_prefix_len
    );
}

/// Every fixture through `run_decision` with `LabelTokenHead` on `model`.
pub fn check_all<R>(
    client: &R::Client,
    model: &Qwen35Model<R>,
    fixtures: &[Fixture],
    tol: Tolerance,
) where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R>,
{
    let opts = AnswerOptions::default();
    let cfg = RunnerConfig::default();
    for fx in fixtures {
        let (plan, items, meta, temp) = request(fx);
        let out = run_decision(
            client,
            model,
            &LabelTokenHead,
            &plan,
            None,
            &items,
            &meta,
            &temp,
            &opts,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("{}: run_decision: {e}", fx.label()));
        compare(fx, &out, tol);
    }
}
