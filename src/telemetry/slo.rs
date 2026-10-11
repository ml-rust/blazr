//! Latency SLO (Service Level Objective) checks.
//!
//! The checker keeps rolling windows of TTFT, ITL and end-to-end latency.
//! Each new sample reads the configured percentiles of its window. Each
//! percentile above its `server.slo` threshold logs a warning and adds 1 to
//! `blazr_slo_violations_total{metric, percentile}`. The P95 TTFT and ITL
//! targets also drive [`SloBatchPolicy`], which holds batch growth.

use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Instant;

use super::names::SLO_VIOLATIONS_TOTAL;
use crate::config::LatencySlo;
use crate::engine::latency_window::LatencyWindow;
use crate::engine::SloBatchPolicy;

/// Samples each checker window holds.
const WINDOW_SIZE: usize = 1000;

/// The process's SLO state, set once at server start from `server.slo`.
static STATE: OnceLock<SloState> = OnceLock::new();

struct SloState {
    checker: SloChecker,
    policy: Option<Arc<SloBatchPolicy>>,
}

/// Check every later latency sample against `slo`, and return the batch
/// policy its P95 targets set, if any.
///
/// `server.slo` is a restart-only key. A second call with the same
/// thresholds returns the first policy. A call with other thresholds keeps
/// the first ones and logs a warning.
pub fn install(slo: LatencySlo) -> Option<Arc<SloBatchPolicy>> {
    let state = STATE.get_or_init(|| SloState {
        policy: SloBatchPolicy::from_slo(&slo).map(Arc::new),
        checker: SloChecker::new(slo.clone()),
    });
    if state.checker.config != slo {
        tracing::warn!("server.slo is already set for this process. The first thresholds stay");
    }
    state.policy.clone()
}

/// Register a zero series for each configured threshold. Call this once the
/// process recorder is installed, so alerts see the series before a breach.
pub fn publish() {
    if let Some(state) = STATE.get() {
        state.checker.publish();
    }
}

/// Record a time to first token, when an SLO is installed.
pub fn observe_ttft_ms(ms: f64) {
    if let Some(state) = STATE.get() {
        state.checker.record_ttft_ms(ms);
        if let Some(policy) = &state.policy {
            policy.observe_ttft_ms(ms);
        }
    }
}

/// Record an inter-token latency, when an SLO is installed.
pub fn observe_itl_ms(ms: f64) {
    if let Some(state) = STATE.get() {
        state.checker.record_itl_ms(ms);
        if let Some(policy) = &state.policy {
            policy.observe_itl_ms(ms);
        }
    }
}

/// Record an end-to-end request latency, when an SLO is installed.
pub fn observe_e2e_ms(ms: f64) {
    if let Some(state) = STATE.get() {
        state.checker.record_e2e_ms(ms);
    }
}

/// Lock a window. A panic in another holder leaves plain samples behind,
/// so the data stays usable.
fn lock(window: &Mutex<LatencyWindow>) -> MutexGuard<'_, LatencyWindow> {
    window.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The latency kinds and the percentiles each one checks.
const TTFT: &str = "ttft";
const ITL: &str = "itl";
const E2E: &str = "e2e";

/// Checks each latency sample against the configured thresholds.
pub struct SloChecker {
    config: LatencySlo,
    ttft_samples: Mutex<LatencyWindow>,
    itl_samples: Mutex<LatencyWindow>,
    e2e_samples: Mutex<LatencyWindow>,
}

impl SloChecker {
    pub fn new(config: LatencySlo) -> Self {
        Self {
            config,
            ttft_samples: Mutex::new(LatencyWindow::new(WINDOW_SIZE, None)),
            itl_samples: Mutex::new(LatencyWindow::new(WINDOW_SIZE, None)),
            e2e_samples: Mutex::new(LatencyWindow::new(WINDOW_SIZE, None)),
        }
    }

    /// Every (metric, percentile, threshold) the config sets.
    fn thresholds(&self) -> Vec<(&'static str, usize, u64)> {
        let c = &self.config;
        [
            (TTFT, 50, c.ttft_p50_ms),
            (TTFT, 95, c.ttft_p95_ms),
            (TTFT, 99, c.ttft_p99_ms),
            (ITL, 50, c.itl_p50_ms),
            (ITL, 95, c.itl_p95_ms),
            (ITL, 99, c.itl_p99_ms),
            (E2E, 99, c.e2e_p99_ms),
        ]
        .into_iter()
        .filter_map(|(metric, p, ms)| ms.map(|ms| (metric, p, ms)))
        .collect()
    }

    /// Register a zero counter series for each configured threshold.
    pub fn publish(&self) {
        for (metric, percentile, _) in self.thresholds() {
            violations(metric, percentile).increment(0);
        }
    }

    /// Record a TTFT sample in milliseconds and check its thresholds.
    pub fn record_ttft_ms(&self, ttft_ms: f64) {
        self.record(&self.ttft_samples, TTFT, ttft_ms);
    }

    /// Record an ITL sample in milliseconds and check its thresholds.
    pub fn record_itl_ms(&self, itl_ms: f64) {
        self.record(&self.itl_samples, ITL, itl_ms);
    }

    /// Record an end-to-end latency sample in milliseconds and check it.
    pub fn record_e2e_ms(&self, e2e_ms: f64) {
        self.record(&self.e2e_samples, E2E, e2e_ms);
    }

    fn record(&self, window: &Mutex<LatencyWindow>, metric: &str, sample_ms: f64) {
        let mut window = lock(window);
        window.push(Instant::now(), sample_ms);
        for (kind, percentile, threshold_ms) in self.thresholds() {
            if kind != metric {
                continue;
            }
            let Some(actual) = window.percentile(percentile) else {
                continue;
            };
            if actual > threshold_ms as f64 {
                tracing::warn!(
                    metric,
                    percentile = format!("p{percentile}").as_str(),
                    actual_ms = format!("{actual:.1}").as_str(),
                    threshold_ms,
                    "SLO violation"
                );
                violations(kind, percentile).increment(1);
            }
        }
    }
}

/// The violation counter series for one metric and percentile.
fn violations(metric: &'static str, percentile: usize) -> metrics::Counter {
    metrics::counter!(
        SLO_VIOLATIONS_TOTAL,
        "metric" => metric,
        "percentile" => format!("p{percentile}")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::test_recorder::TestRecorder;

    fn slo() -> LatencySlo {
        LatencySlo {
            ttft_p50_ms: None,
            ttft_p95_ms: Some(100),
            ttft_p99_ms: None,
            itl_p50_ms: None,
            itl_p95_ms: Some(20),
            itl_p99_ms: None,
            e2e_p99_ms: Some(1000),
        }
    }

    fn count(recorder: &TestRecorder, metric: &str, percentile: &str) -> Option<f64> {
        recorder.value(
            SLO_VIOLATIONS_TOTAL,
            &[("metric", metric), ("percentile", percentile)],
        )
    }

    #[test]
    fn samples_under_the_thresholds_count_nothing() {
        let recorder = TestRecorder::install();
        let checker = SloChecker::new(slo());
        checker.publish();
        for _ in 0..10 {
            checker.record_ttft_ms(50.0);
            checker.record_itl_ms(5.0);
            checker.record_e2e_ms(500.0);
        }
        assert_eq!(count(&recorder, "ttft", "p95"), Some(0.0));
        assert_eq!(count(&recorder, "itl", "p95"), Some(0.0));
        assert_eq!(count(&recorder, "e2e", "p99"), Some(0.0));
        // Unconfigured thresholds have no series.
        assert_eq!(count(&recorder, "ttft", "p50"), None);
    }

    #[test]
    fn the_first_breaching_request_counts_for_each_metric() {
        let recorder = TestRecorder::install();
        let checker = SloChecker::new(slo());
        checker.record_ttft_ms(250.0);
        checker.record_itl_ms(45.0);
        checker.record_e2e_ms(4000.0);
        assert_eq!(count(&recorder, "ttft", "p95"), Some(1.0));
        assert_eq!(count(&recorder, "itl", "p95"), Some(1.0));
        assert_eq!(count(&recorder, "e2e", "p99"), Some(1.0));
    }

    #[test]
    fn one_slow_sample_among_many_fast_ones_keeps_p95_under_target() {
        let recorder = TestRecorder::install();
        let checker = SloChecker::new(slo());
        for _ in 0..99 {
            checker.record_ttft_ms(10.0);
        }
        checker.record_ttft_ms(500.0);
        assert_eq!(count(&recorder, "ttft", "p95"), None);
    }
}
