//! The `blazr_queue_depth` gauge.
//!
//! The gauge counts requests that wait for a decode slot or for token-budget
//! admission. Each queue owns one [`QueueDepth`] and reports its own count.
//! The gauge holds the sum over every queue in the process.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::telemetry::names::QUEUE_DEPTH;

/// One queue's share of `blazr_queue_depth`.
#[derive(Debug, Default)]
pub struct QueueDepth {
    waiting: AtomicUsize,
}

impl QueueDepth {
    /// A share of 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set this queue's waiting count. The gauge moves by the change.
    pub fn set(&self, waiting: usize) {
        let before = self.waiting.swap(waiting, Ordering::AcqRel);
        move_gauge(before, waiting);
    }

    /// Count one more waiting request until the returned guard drops.
    pub fn enter(&self) -> Waiting<'_> {
        self.waiting.fetch_add(1, Ordering::AcqRel);
        metrics::gauge!(QUEUE_DEPTH).increment(1.0);
        Waiting { depth: self }
    }

    /// This queue's waiting count.
    pub fn get(&self) -> usize {
        self.waiting.load(Ordering::Acquire)
    }
}

impl Drop for QueueDepth {
    fn drop(&mut self) {
        self.set(0);
    }
}

/// One request counted in a [`QueueDepth`]. Dropping it ends the wait.
#[derive(Debug)]
pub struct Waiting<'a> {
    depth: &'a QueueDepth,
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.depth.waiting.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!(QUEUE_DEPTH).decrement(1.0);
    }
}

/// Move the gauge from `before` to `after`.
fn move_gauge(before: usize, after: usize) {
    if after > before {
        metrics::gauge!(QUEUE_DEPTH).increment((after - before) as f64);
    } else if before > after {
        metrics::gauge!(QUEUE_DEPTH).decrement((before - after) as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::test_recorder::TestRecorder;

    #[test]
    fn set_and_enter_sum_into_one_gauge() {
        let recorder = TestRecorder::install();
        let a = QueueDepth::new();
        let b = QueueDepth::new();
        a.set(3);
        b.set(2);
        a.set(1);
        let guard = b.enter();
        assert_eq!(b.get(), 3);
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(4.0));
        drop(guard);
        assert_eq!(b.get(), 2);
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(3.0));
        drop(a);
        drop(b);
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(0.0));
    }
}
