//! SLO-aware batch growth for the continuous-batching schedulers.
//!
//! The policy keeps short rolling windows of time to first token (TTFT) and
//! inter-token latency (ITL). While the rolling P95 of either one exceeds
//! its `server.slo` target, the batch stops growing. Running sequences keep
//! decoding, and a waiting request joins only as a running one leaves. The
//! batch grows again once both P95 values are back under their targets.

use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use super::latency_window::LatencyWindow;
use crate::config::LatencySlo;
use crate::telemetry::names::SLO_BATCH_HELD;

/// Samples each policy window holds.
const POLICY_WINDOW: usize = 256;

/// Age after which a sample no longer counts. Without fresh slow samples,
/// growth resumes after this time.
const POLICY_MAX_AGE: Duration = Duration::from_secs(30);

/// Batch growth limits from the P95 TTFT and ITL targets.
#[derive(Debug)]
pub struct SloBatchPolicy {
    ttft_p95_ms: Option<f64>,
    itl_p95_ms: Option<f64>,
    state: Mutex<PolicyState>,
}

#[derive(Debug)]
struct PolicyState {
    ttft: LatencyWindow,
    itl: LatencyWindow,
    /// Whether the last read found a breach. It drives the gauge and log.
    held: bool,
}

impl SloBatchPolicy {
    /// The policy for `slo`, or `None` when it sets no P95 TTFT or ITL target.
    pub fn from_slo(slo: &LatencySlo) -> Option<Self> {
        if slo.ttft_p95_ms.is_none() && slo.itl_p95_ms.is_none() {
            return None;
        }
        Some(Self {
            ttft_p95_ms: slo.ttft_p95_ms.map(|ms| ms as f64),
            itl_p95_ms: slo.itl_p95_ms.map(|ms| ms as f64),
            state: Mutex::new(PolicyState {
                ttft: LatencyWindow::new(POLICY_WINDOW, Some(POLICY_MAX_AGE)),
                itl: LatencyWindow::new(POLICY_WINDOW, Some(POLICY_MAX_AGE)),
                held: false,
            }),
        })
    }

    /// Record a time to first token in milliseconds.
    pub fn observe_ttft_ms(&self, ms: f64) {
        self.observe_ttft_ms_at(Instant::now(), ms);
    }

    /// Record an inter-token latency in milliseconds.
    pub fn observe_itl_ms(&self, ms: f64) {
        self.observe_itl_ms_at(Instant::now(), ms);
    }

    /// Record a time to first token taken at `at`.
    pub fn observe_ttft_ms_at(&self, at: Instant, ms: f64) {
        self.lock().ttft.push(at, ms);
    }

    /// Record an inter-token latency taken at `at`.
    pub fn observe_itl_ms_at(&self, at: Instant, ms: f64) {
        self.lock().itl.push(at, ms);
    }

    /// The most sequences the batch can hold after this scheduling step.
    ///
    /// `running` counts the sequences already decoding. `max_batch` is the
    /// configured batch size. Under a breach the limit is `running`, with a
    /// floor of 1 so an idle scheduler still admits one request.
    pub fn batch_limit(&self, running: usize, max_batch: usize) -> usize {
        self.batch_limit_at(Instant::now(), running, max_batch)
    }

    /// [`SloBatchPolicy::batch_limit`] read at time `now`.
    pub fn batch_limit_at(&self, now: Instant, running: usize, max_batch: usize) -> usize {
        if self.breached_at(now) {
            running.clamp(1, max_batch.max(1))
        } else {
            max_batch
        }
    }

    /// Whether the rolling P95 TTFT or ITL exceeds its target at `now`.
    pub fn breached_at(&self, now: Instant) -> bool {
        let mut state = self.lock();
        state.ttft.expire(now);
        state.itl.expire(now);
        let ttft = over(&state.ttft, self.ttft_p95_ms);
        let itl = over(&state.itl, self.itl_p95_ms);
        let held = ttft.is_some() || itl.is_some();
        if held != state.held {
            state.held = held;
            metrics::gauge!(SLO_BATCH_HELD).set(if held { 1.0 } else { 0.0 });
            if held {
                tracing::warn!(
                    ttft_p95_ms = ttft.map(|v| format!("{v:.1}")).unwrap_or_default(),
                    itl_p95_ms = itl.map(|v| format!("{v:.1}")).unwrap_or_default(),
                    "SLO breach: batch growth held until P95 latency is back under target"
                );
            } else {
                tracing::info!("P95 latency is under the SLO targets. Batch growth resumes");
            }
        }
        held
    }

    fn lock(&self) -> MutexGuard<'_, PolicyState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The P95 of `window` when it exceeds `target`.
fn over(window: &LatencyWindow, target: Option<f64>) -> Option<f64> {
    let target = target?;
    window.percentile(95).filter(|&p95| p95 > target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slo(ttft: Option<u64>, itl: Option<u64>) -> LatencySlo {
        LatencySlo {
            ttft_p50_ms: None,
            ttft_p95_ms: ttft,
            ttft_p99_ms: None,
            itl_p50_ms: None,
            itl_p95_ms: itl,
            itl_p99_ms: None,
            e2e_p99_ms: None,
        }
    }

    #[test]
    fn no_p95_target_gives_no_policy() {
        assert!(SloBatchPolicy::from_slo(&slo(None, None)).is_none());
    }

    #[test]
    fn healthy_latency_allows_the_full_batch() {
        let policy = SloBatchPolicy::from_slo(&slo(Some(100), Some(20))).expect("policy");
        assert_eq!(policy.batch_limit(3, 8), 8);
        policy.observe_ttft_ms(50.0);
        policy.observe_itl_ms(10.0);
        assert_eq!(policy.batch_limit(3, 8), 8);
    }

    #[test]
    fn a_ttft_breach_holds_the_batch_at_its_running_size() {
        let policy = SloBatchPolicy::from_slo(&slo(Some(100), None)).expect("policy");
        policy.observe_ttft_ms(250.0);
        assert_eq!(policy.batch_limit(3, 8), 3);
        assert_eq!(policy.batch_limit(0, 8), 1);
        assert_eq!(policy.batch_limit(12, 8), 8);
    }

    #[test]
    fn an_itl_breach_holds_the_batch() {
        let policy = SloBatchPolicy::from_slo(&slo(None, Some(20))).expect("policy");
        for _ in 0..10 {
            policy.observe_itl_ms(45.0);
        }
        assert_eq!(policy.batch_limit(2, 8), 2);
    }

    #[test]
    fn growth_resumes_when_p95_drops_or_samples_age_out() {
        let policy = SloBatchPolicy::from_slo(&slo(Some(100), None)).expect("policy");
        let now = Instant::now();
        policy.observe_ttft_ms_at(now, 400.0);
        assert_eq!(policy.batch_limit_at(now, 2, 8), 2);

        // Fast samples push the P95 under the target.
        for _ in 0..40 {
            policy.observe_ttft_ms_at(now, 10.0);
        }
        assert_eq!(policy.batch_limit_at(now, 2, 8), 8);

        // A slow burst holds again, and it ages out after the max age.
        for _ in 0..40 {
            policy.observe_ttft_ms_at(now, 900.0);
        }
        assert_eq!(policy.batch_limit_at(now, 2, 8), 2);
        let later = now + POLICY_MAX_AGE + Duration::from_secs(1);
        assert_eq!(policy.batch_limit_at(later, 2, 8), 8);
    }
}
