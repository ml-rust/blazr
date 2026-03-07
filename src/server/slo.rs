//! Latency SLO (Service Level Objective) enforcement.
//!
//! Tracks rolling percentiles for TTFT, ITL, and end-to-end latency.
//! Logs warnings and increments Prometheus counters on SLO violations.

use std::sync::Mutex;

use crate::config::LatencySlo;

/// SLO metric name for Prometheus counter
const SLO_VIOLATIONS_TOTAL: &str = "blazr_slo_violations_total";

/// Rolling window size for percentile estimation
const WINDOW_SIZE: usize = 1000;

/// SLO checker that tracks recent latency samples and checks against thresholds.
pub struct SloChecker {
    config: LatencySlo,
    ttft_samples: Mutex<RollingWindow>,
    itl_samples: Mutex<RollingWindow>,
    e2e_samples: Mutex<RollingWindow>,
}

impl SloChecker {
    pub fn new(config: LatencySlo) -> Self {
        // Register the SLO violations counter
        metrics::describe_counter!(SLO_VIOLATIONS_TOTAL, "Total SLO violations detected");

        Self {
            config,
            ttft_samples: Mutex::new(RollingWindow::new(WINDOW_SIZE)),
            itl_samples: Mutex::new(RollingWindow::new(WINDOW_SIZE)),
            e2e_samples: Mutex::new(RollingWindow::new(WINDOW_SIZE)),
        }
    }

    /// Record a TTFT sample (in milliseconds) and check SLO
    pub fn record_ttft_ms(&self, ttft_ms: f64) {
        let mut window = self.ttft_samples.lock().unwrap();
        window.push(ttft_ms);

        if window.len() >= 100 {
            self.check_percentile(&window, ttft_ms, "ttft", self.config.ttft_p50_ms, 50);
            self.check_percentile(&window, ttft_ms, "ttft", self.config.ttft_p95_ms, 95);
            self.check_percentile(&window, ttft_ms, "ttft", self.config.ttft_p99_ms, 99);
        }
    }

    /// Record an ITL sample (in milliseconds) and check SLO
    pub fn record_itl_ms(&self, itl_ms: f64) {
        let mut window = self.itl_samples.lock().unwrap();
        window.push(itl_ms);

        if window.len() >= 100 {
            self.check_percentile(&window, itl_ms, "itl", self.config.itl_p50_ms, 50);
            self.check_percentile(&window, itl_ms, "itl", self.config.itl_p95_ms, 95);
            self.check_percentile(&window, itl_ms, "itl", self.config.itl_p99_ms, 99);
        }
    }

    /// Record an end-to-end latency sample (in milliseconds) and check SLO
    pub fn record_e2e_ms(&self, e2e_ms: f64) {
        let mut window = self.e2e_samples.lock().unwrap();
        window.push(e2e_ms);

        if window.len() >= 100 {
            self.check_percentile(&window, e2e_ms, "e2e", self.config.e2e_p99_ms, 99);
        }
    }

    fn check_percentile(
        &self,
        window: &RollingWindow,
        _sample: f64,
        metric: &str,
        threshold: Option<u64>,
        percentile: usize,
    ) {
        if let Some(threshold_ms) = threshold {
            let actual = window.percentile(percentile);
            if actual > threshold_ms as f64 {
                tracing::warn!(
                    metric = metric,
                    percentile = format!("p{}", percentile).as_str(),
                    actual_ms = format!("{:.1}", actual).as_str(),
                    threshold_ms = threshold_ms,
                    "SLO violation"
                );
                metrics::counter!(
                    SLO_VIOLATIONS_TOTAL,
                    "metric" => metric.to_string(),
                    "percentile" => format!("p{}", percentile)
                )
                .increment(1);
            }
        }
    }
}

/// Fixed-size rolling window for percentile estimation
struct RollingWindow {
    samples: Vec<f64>,
    max_size: usize,
}

impl RollingWindow {
    fn new(max_size: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_size),
            max_size,
        }
    }

    fn push(&mut self, value: f64) {
        if self.samples.len() >= self.max_size {
            self.samples.remove(0);
        }
        self.samples.push(value);
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    /// Compute approximate percentile (0-100)
    fn percentile(&self, p: usize) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (p as f64 / 100.0 * (sorted.len() - 1) as f64) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_window_percentile() {
        let mut w = RollingWindow::new(100);
        for i in 1..=100 {
            w.push(i as f64);
        }
        assert!((w.percentile(50) - 50.0).abs() < 2.0);
        assert!((w.percentile(95) - 95.0).abs() < 2.0);
        assert!((w.percentile(99) - 99.0).abs() < 2.0);
    }

    #[test]
    fn test_rolling_window_eviction() {
        let mut w = RollingWindow::new(10);
        for i in 0..20 {
            w.push(i as f64);
        }
        assert_eq!(w.len(), 10);
        // Should contain 10..20
        assert_eq!(w.samples[0], 10.0);
    }

    #[test]
    fn test_rolling_window_empty() {
        let w = RollingWindow::new(10);
        assert_eq!(w.percentile(50), 0.0);
    }

    #[test]
    fn test_slo_checker_no_violation() {
        // Install a test recorder
        let _ = metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder();

        let config = LatencySlo {
            ttft_p50_ms: Some(1000),
            ttft_p95_ms: None,
            ttft_p99_ms: None,
            itl_p50_ms: None,
            itl_p95_ms: None,
            itl_p99_ms: None,
            e2e_p99_ms: None,
        };
        let checker = SloChecker::new(config);
        // Record 100 samples all under threshold
        for _ in 0..100 {
            checker.record_ttft_ms(50.0);
        }
        // No panic = no violation logged (we can't easily assert on tracing output)
    }
}
