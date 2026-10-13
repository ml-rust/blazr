//! One temperature per answer type, fitted by boostr on the labelled
//! answers of that type. Score fits its listed and isolated answers
//! together. A type with fewer than the minimum answers is not fitted.

use boostr::inference::decision::{
    fit_temperature, fit_temperature_mixed, FitOutcome, TemperatureBounds, TemperatureFit,
};

use super::labels::AnswerType;
use super::samples::TypeSamples;

/// The default minimum answers a type needs to be fitted.
pub const DEFAULT_MIN_ROWS: usize = 50;

/// One type's result.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeReport {
    /// The answer type.
    pub kind: AnswerType,
    /// Its labelled answers.
    pub n: usize,
    /// The fit. `None` when `n` is below the minimum.
    pub fit: Option<TemperatureFit>,
}

/// The name a fit outcome is written as.
pub fn outcome_name(outcome: FitOutcome) -> &'static str {
    match outcome {
        FitOutcome::Converged => "converged",
        FitOutcome::Bound => "bound",
        FitOutcome::Flat => "flat",
        FitOutcome::IterationLimit => "iteration_limit",
        _ => "unknown",
    }
}

/// Fit every type with at least `min_rows` answers (at least 1) within
/// `bounds`. Every type is reported, fitted or not, in
/// [`AnswerType::ALL`] order.
///
/// # Errors
///
/// boostr's message when a fit refuses its samples or the bounds.
pub fn fit_types(
    samples: &TypeSamples,
    min_rows: usize,
    bounds: TemperatureBounds,
) -> Result<Vec<TypeReport>, String> {
    let min_rows = min_rows.max(1);
    AnswerType::ALL
        .into_iter()
        .map(|kind| {
            let n = samples.count(kind);
            if n < min_rows {
                return Ok(TypeReport { kind, n, fit: None });
            }
            let fit = match kind {
                AnswerType::Choice => fit_temperature(&samples.choice, bounds),
                AnswerType::Noul => fit_temperature(&samples.noul, bounds),
                AnswerType::Score => {
                    fit_temperature_mixed(&samples.score, &samples.score_isolated, bounds)
                }
            }
            .map_err(|e| format!("the {} fit failed: {e}", kind.name()))?;
            Ok(TypeReport {
                kind,
                n,
                fit: Some(fit),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::inference::decision::{CalibrationSample, IsolatedSample, Target};

    fn listed(n: usize, k: usize) -> Vec<CalibrationSample> {
        (0..n)
            .map(|i| CalibrationSample {
                logits: (0..k).map(|j| ((i * 7 + j * 3) % 5) as f64 - 2.0).collect(),
                target: Target::Hard(i % k),
            })
            .collect()
    }

    #[test]
    fn types_below_the_minimum_are_not_fitted() {
        let samples = TypeSamples {
            choice: listed(60, 3),
            noul: listed(10, 2),
            score: listed(30, 4),
            score_isolated: (0..30)
                .map(|i| IsolatedSample {
                    levels: vec![[0.0, (i % 3) as f64], [1.0, 0.5], [0.2, -1.0]],
                    target: Target::Hard(i % 3),
                })
                .collect(),
        };
        let report = fit_types(&samples, 50, TemperatureBounds::default()).expect("fits");
        let kinds: Vec<_> = report
            .iter()
            .map(|r| (r.kind, r.n, r.fit.is_some()))
            .collect();
        assert_eq!(
            kinds,
            [
                (AnswerType::Choice, 60, true),
                (AnswerType::Noul, 10, false),
                (AnswerType::Score, 60, true)
            ]
        );
        let lowered = fit_types(&samples, 10, TemperatureBounds::default()).expect("fits");
        assert!(lowered.iter().all(|r| r.fit.is_some()));
        let none = fit_types(&TypeSamples::default(), 0, TemperatureBounds::default())
            .expect("nothing to fit");
        assert!(none.iter().all(|r| r.n == 0 && r.fit.is_none()));
    }

    #[test]
    fn the_score_fit_pools_both_readouts() {
        let isolated: Vec<IsolatedSample> = (0..20)
            .map(|i| IsolatedSample {
                levels: vec![[0.0, (i % 4) as f64 - 1.0], [0.5, 0.0]],
                target: Target::Hard(i % 2),
            })
            .collect();
        let samples = TypeSamples {
            score: listed(20, 3),
            score_isolated: isolated.clone(),
            ..TypeSamples::default()
        };
        let bounds = TemperatureBounds::default();
        let report = fit_types(&samples, 1, bounds).expect("fits");
        let want = fit_temperature_mixed(&samples.score, &isolated, bounds).expect("fits");
        assert_eq!(report[2].fit, Some(want));
        assert_eq!(outcome_name(FitOutcome::Bound), "bound");
    }
}
