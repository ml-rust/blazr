//! Latency and token metrics of one streamed generation.
//!
//! The executor stream and the batching scheduler stream both record
//! through [`StreamMetrics`], so a dashboard reads the same series for
//! either path.

use std::time::Instant;

use crate::server::metrics;

/// The clock and token count of one streamed request.
pub struct StreamMetrics {
    start: Instant,
    last_token: Instant,
    prompt_tokens: usize,
    completion_tokens: usize,
}

impl StreamMetrics {
    /// Start the clock for a request of `prompt_tokens` prompt ids.
    pub fn start(prompt_tokens: usize) -> Self {
        let now = Instant::now();
        Self {
            start: now,
            last_token: now,
            prompt_tokens,
            completion_tokens: 0,
        }
    }

    /// Record one generated token.
    ///
    /// The first token closes the TTFT. Each later token is one ITL sample.
    pub fn token(&mut self) {
        self.completion_tokens += 1;
        let now = Instant::now();
        if self.completion_tokens == 1 {
            metrics::record_ttft(now.duration_since(self.start).as_secs_f64());
        } else {
            metrics::record_itl(now.duration_since(self.last_token).as_secs_f64());
        }
        self.last_token = now;
    }

    /// Record the end of the stream: end-to-end time, token counts and
    /// throughput. A stream with no token records nothing.
    pub fn finish(self) {
        if self.completion_tokens == 0 {
            return;
        }
        let elapsed = self.start.elapsed().as_secs_f64();
        metrics::record_e2e(elapsed);
        metrics::record_tokens(self.prompt_tokens, self.completion_tokens);
        if elapsed > 0.0 {
            metrics::record_tokens_per_second(self.completion_tokens as f64 / elapsed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::metrics::names::{INTER_TOKEN_LATENCY, TOKENS_GENERATED, TOKENS_PROMPTED};
    use crate::server::test_recorder::{series_value, TestRecorder};

    #[test]
    fn three_tokens_record_two_itl_samples_and_the_token_counts() {
        let recorder = TestRecorder::install();
        let mut stream = StreamMetrics::start(5);
        for _ in 0..3 {
            std::thread::sleep(std::time::Duration::from_millis(1));
            stream.token();
        }
        stream.finish();
        let rendered = recorder.render();
        let itl = series_value(&rendered, &format!("{INTER_TOKEN_LATENCY}_count"), &[]);
        assert_eq!(itl, Some(2.0), "{rendered}");
        assert_eq!(
            recorder.value(TOKENS_PROMPTED, &[]),
            Some(5.0),
            "{rendered}"
        );
        assert_eq!(
            recorder.value(TOKENS_GENERATED, &[]),
            Some(3.0),
            "{rendered}"
        );
    }

    #[test]
    fn a_stream_with_no_token_records_no_counts() {
        let recorder = TestRecorder::install();
        StreamMetrics::start(5).finish();
        assert_eq!(recorder.value(TOKENS_PROMPTED, &[]), None);
    }
}
