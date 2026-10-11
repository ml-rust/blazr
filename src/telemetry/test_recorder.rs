//! A Prometheus recorder scoped to one test thread.
//!
//! Tests read the metrics that a code path records without the process
//! recorder. A `#[tokio::test]` runs on one thread, so async code records
//! into the same recorder.

use metrics::LocalRecorderGuard;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle, PrometheusRecorder};

/// The recorder of the current thread until this value drops.
pub struct TestRecorder {
    handle: PrometheusHandle,
    _guard: LocalRecorderGuard<'static>,
}

impl TestRecorder {
    /// Install a fresh recorder on this thread.
    pub fn install() -> Self {
        // The guard borrows the recorder for its whole life. One small
        // recorder per test stays allocated until the test process exits.
        let recorder: &'static PrometheusRecorder =
            Box::leak(Box::new(PrometheusBuilder::new().build_recorder()));
        let handle = recorder.handle();
        let guard = metrics::set_default_local_recorder(recorder);
        Self {
            handle,
            _guard: guard,
        }
    }

    /// The Prometheus text of every recorded metric.
    pub fn render(&self) -> String {
        self.handle.render()
    }

    /// The value of the series `name` whose labels include every pair in
    /// `labels`. An empty `labels` matches only a series with no labels.
    pub fn value(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        series_value(&self.render(), name, labels)
    }
}

/// The value of one series in Prometheus text.
pub fn series_value(text: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    text.lines().find_map(|line| {
        let rest = line.strip_prefix(name)?;
        let (label_text, value) = if let Some(inner) = rest.strip_prefix('{') {
            let (inner, value) = inner.split_once('}')?;
            (inner, value)
        } else if rest.starts_with(' ') {
            ("", rest)
        } else {
            return None;
        };
        let wanted = labels
            .iter()
            .all(|(k, v)| label_text.contains(&format!("{k}=\"{v}\"")));
        let exact = !labels.is_empty() || label_text.is_empty();
        if wanted && exact {
            value.trim().parse().ok()
        } else {
            None
        }
    })
}
