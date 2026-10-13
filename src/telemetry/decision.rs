//! Recording helpers of the decision worker. Each metric carries the model
//! name as its `model` label.

use super::names;

/// Record one decision request the worker finished: its prompt rows, the
/// tokens its run prefilled, the seconds from queueing to its answer, and
/// whether it was answered.
pub fn record_decision(model: &str, rows: usize, prefill_tokens: usize, secs: f64, answered: bool) {
    let model = model.to_string();
    let outcome = if answered { "answered" } else { "failed" };
    metrics::counter!(names::DECISION_REQUESTS_TOTAL, "model" => model.clone(), "outcome" => outcome)
        .increment(1);
    metrics::counter!(names::DECISION_ROWS_TOTAL, "model" => model.clone()).increment(rows as u64);
    metrics::counter!(names::DECISION_PREFILL_TOKENS_TOTAL, "model" => model.clone())
        .increment(prefill_tokens as u64);
    metrics::histogram!(names::DECISION_LATENCY, "model" => model).record(secs);
}

/// Set the rows queued or running on a model's decision worker.
pub fn set_decision_queue_rows(model: &str, rows: usize) {
    metrics::gauge!(names::DECISION_QUEUE_ROWS, "model" => model.to_string()).set(rows as f64);
}

/// Record what a model's prefix-state cache did since the last record:
/// `hits`, `misses` and `evictions` added, and the `bytes` it holds now.
pub fn record_prefix_cache(model: &str, hits: u64, misses: u64, evictions: u64, bytes: usize) {
    let model = model.to_string();
    metrics::counter!(names::DECISION_PREFIX_CACHE_HITS_TOTAL, "model" => model.clone())
        .increment(hits);
    metrics::counter!(names::DECISION_PREFIX_CACHE_MISSES_TOTAL, "model" => model.clone())
        .increment(misses);
    metrics::counter!(names::DECISION_PREFIX_CACHE_EVICTIONS_TOTAL, "model" => model.clone())
        .increment(evictions);
    metrics::gauge!(names::DECISION_PREFIX_CACHE_BYTES, "model" => model).set(bytes as f64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::test_recorder::TestRecorder;

    #[test]
    fn records_each_decision_series_by_model() {
        let recorder = TestRecorder::install();
        record_decision("d", 3, 120, 0.25, true);
        record_decision("d", 1, 40, 0.5, false);
        set_decision_queue_rows("d", 7);
        let value = |name: &str, labels: &[(&str, &str)]| recorder.value(name, labels);
        let answered = [("model", "d"), ("outcome", "answered")];
        assert_eq!(value(names::DECISION_REQUESTS_TOTAL, &answered), Some(1.0));
        assert_eq!(
            value(names::DECISION_ROWS_TOTAL, &[("model", "d")]),
            Some(4.0)
        );
        assert_eq!(
            value(names::DECISION_PREFILL_TOKENS_TOTAL, &[("model", "d")]),
            Some(160.0)
        );
        assert_eq!(
            value(names::DECISION_QUEUE_ROWS, &[("model", "d")]),
            Some(7.0)
        );
        let latency = format!("{}_count", names::DECISION_LATENCY);
        assert_eq!(value(latency.as_str(), &[("model", "d")]), Some(2.0));
    }

    #[test]
    fn records_the_prefix_cache_by_model() {
        let recorder = TestRecorder::install();
        record_prefix_cache("d", 2, 1, 0, 300);
        record_prefix_cache("d", 1, 0, 1, 100);
        let value = |name: &str| recorder.value(name, &[("model", "d")]);
        assert_eq!(value(names::DECISION_PREFIX_CACHE_HITS_TOTAL), Some(3.0));
        assert_eq!(value(names::DECISION_PREFIX_CACHE_MISSES_TOTAL), Some(1.0));
        let evictions = names::DECISION_PREFIX_CACHE_EVICTIONS_TOTAL;
        assert_eq!(value(evictions), Some(1.0));
        assert_eq!(value(names::DECISION_PREFIX_CACHE_BYTES), Some(100.0));
    }
}
