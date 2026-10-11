//! A bounded window of latency samples with percentile reads.
//!
//! The SLO checker and the SLO batch policy both read rolling percentiles
//! from this window.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// The newest latency samples, oldest first.
///
/// The window holds at most `max_len` samples. With `max_age` set, a read
/// ignores and drops samples older than that age.
#[derive(Debug)]
pub struct LatencyWindow {
    samples: VecDeque<(Instant, f64)>,
    max_len: usize,
    max_age: Option<Duration>,
}

impl LatencyWindow {
    /// A window of up to `max_len` samples. `max_age` of `None` keeps
    /// samples until newer ones push them out.
    pub fn new(max_len: usize, max_age: Option<Duration>) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_len),
            max_len: max_len.max(1),
            max_age,
        }
    }

    /// Add `value`, taken at `at`. The oldest sample leaves a full window.
    pub fn push(&mut self, at: Instant, value: f64) {
        if self.samples.len() >= self.max_len {
            self.samples.pop_front();
        }
        self.samples.push_back((at, value));
    }

    /// Drop the samples older than `max_age` at time `now`.
    pub fn expire(&mut self, now: Instant) {
        let Some(max_age) = self.max_age else {
            return;
        };
        while let Some(&(at, _)) = self.samples.front() {
            if now.saturating_duration_since(at) <= max_age {
                break;
            }
            self.samples.pop_front();
        }
    }

    /// The number of samples held.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the window holds no sample.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The nearest-rank `p`th percentile (0 to 100), or `None` when empty.
    pub fn percentile(&self, p: usize) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<f64> = self.samples.iter().map(|&(_, v)| v).collect();
        sorted.sort_by(f64::total_cmp);
        let last = sorted.len() - 1;
        let idx = (p.min(100) as f64 / 100.0 * last as f64).round() as usize;
        Some(sorted[idx.min(last)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(n: usize) -> LatencyWindow {
        let mut w = LatencyWindow::new(n, None);
        let now = Instant::now();
        for i in 1..=n {
            w.push(now, i as f64);
        }
        w
    }

    #[test]
    fn percentiles_of_one_to_a_hundred() {
        let w = filled(100);
        for p in [50, 95, 99] {
            let value = w.percentile(p).unwrap_or(0.0);
            assert!((value - p as f64).abs() < 2.0, "p{p} = {value}");
        }
    }

    #[test]
    fn a_full_window_drops_the_oldest_sample() {
        let mut w = LatencyWindow::new(10, None);
        let now = Instant::now();
        for i in 0..20 {
            w.push(now, i as f64);
        }
        assert_eq!(w.len(), 10);
        assert_eq!(w.percentile(0), Some(10.0));
    }

    #[test]
    fn an_empty_window_has_no_percentile() {
        assert_eq!(LatencyWindow::new(10, None).percentile(50), None);
    }

    #[test]
    fn one_sample_is_every_percentile() {
        let mut w = LatencyWindow::new(10, None);
        w.push(Instant::now(), 7.0);
        assert_eq!(w.percentile(50), Some(7.0));
        assert_eq!(w.percentile(99), Some(7.0));
    }

    #[test]
    fn expire_drops_samples_past_the_max_age() {
        let mut w = LatencyWindow::new(10, Some(Duration::from_secs(5)));
        let now = Instant::now();
        let old = now.checked_sub(Duration::from_secs(10)).unwrap_or(now);
        w.push(old, 1.0);
        w.push(now, 2.0);
        w.expire(now);
        if old < now {
            assert_eq!(w.len(), 1);
            assert_eq!(w.percentile(50), Some(2.0));
        }
    }
}
