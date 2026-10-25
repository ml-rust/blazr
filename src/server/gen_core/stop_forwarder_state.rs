//! The state of a [`StopForwarder`]: its stop matcher, metrics and token counts.

use super::stream_metrics::StreamMetrics;
use super::stream_usage::StreamUsage;
use crate::config::GenerationConfig;
use crate::engine::StopMatcher;

/// Maps engine tokens to client items, cutting the text at the first stop
/// sequence.
///
/// With `gen_config.logprobs` set, each item carries its engine token's
/// logprobs. A token the matcher holds back then sends an empty-text item,
/// so its logprob goes out when it arrives. Without logprobs it sends nothing.
///
/// A stream that ends with no finish token closes with `Length`, as the
/// non-streaming collector reads it.
///
/// The item that carries the finish reason carries the token counts: the
/// prompt ids and every engine token, held-back stop text included. A
/// closed channel gets no final item.
pub(super) struct StopForwarder {
    pub(super) matcher: StopMatcher,
    /// Taken when the stream ends.
    pub(super) metrics: Option<StreamMetrics>,
    pub(super) prompt_len: usize,
    pub(super) logprobs: bool,
    /// Names the path in the error log.
    pub(super) label: &'static str,
    /// An engine error ended the stream, so it records no end metrics.
    pub(super) errored: bool,
}

impl StopForwarder {
    pub(super) fn new(
        prompt_len: usize,
        gen_config: &GenerationConfig,
        label: &'static str,
    ) -> Self {
        Self {
            matcher: StopMatcher::new(&gen_config.stop_sequences),
            metrics: Some(StreamMetrics::start(prompt_len)),
            prompt_len,
            logprobs: gen_config.logprobs,
            label,
            errored: false,
        }
    }

    /// The token counts so far.
    pub(super) fn usage(&self) -> StreamUsage {
        match &self.metrics {
            Some(metrics) => metrics.usage(),
            None => StreamUsage::new(self.prompt_len, 0),
        }
    }
}
