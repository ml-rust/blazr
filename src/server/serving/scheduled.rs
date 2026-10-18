//! Non-streaming text generation on a request's route.
//!
//! A route with a batching scheduler submits the prompt there and collects
//! the tokens. A route without one generates on its executor.

use std::time::Instant;

use super::metrics;
use crate::config::GenerationConfig;
use crate::engine::{FinishReason, GeneratedToken, GenerationResult, GenerationScheduler};
use crate::server::handlers::Route;

/// Result of a batched generation via the configured scheduler
pub struct BatchedGenerationResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
    /// Time from submit to the first token.
    pub prompt_eval_duration_ms: u64,
    /// Every token with its logprobs, when the request asked for them.
    pub token_logprobs: Option<Vec<GeneratedToken>>,
}

/// The batching scheduler that serves text on `route`, or `None` when its
/// executor generates one sequence at a time.
///
/// The streaming and non-streaming text paths both route by this. A route
/// carries a scheduler only for the start model with batching on. The
/// graph decoders run only on the executor path.
pub fn batching_scheduler(route: &Route) -> Option<&dyn GenerationScheduler> {
    route.scheduler.as_deref()
}

/// Generate `prompt_tokens` on `route` and return the whole text.
///
/// # Errors
///
/// An error when the scheduler refuses the request or generation fails.
pub async fn generate_text_routed(
    route: &Route,
    prompt_tokens: &[u32],
    gen_config: &GenerationConfig,
) -> anyhow::Result<GenerationResult> {
    match batching_scheduler(route) {
        Some(scheduler) => {
            let r = generate_via_scheduler(scheduler, prompt_tokens.to_vec(), gen_config)
                .await
                .map_err(anyhow::Error::msg)?;
            Ok(GenerationResult {
                text: r.text,
                prompt_tokens: r.prompt_tokens,
                completion_tokens: r.completion_tokens,
                finish_reason: r.finish_reason,
                prompt_eval_duration_ms: r.prompt_eval_duration_ms,
                token_logprobs: r.token_logprobs,
            })
        }
        None => {
            route
                .executor
                .generate_text(prompt_tokens, gen_config)
                .await
        }
    }
}

/// Submit a request to `request_scheduler` and collect every token.
///
/// The first token's arrival sets the prompt eval time. Each later token
/// records one inter-token latency sample.
///
/// The scheduler's engine applies the stop sequences and builds the
/// logprobs, so the text arrives cut and no second stop pass runs here.
///
/// Dropping this future, on a timeout or a disconnect, drops the request's
/// handle. The handle's `Drop` cancels the request in the scheduler.
///
/// # Errors
///
/// An error string when the submit or the generation fails.
pub async fn generate_via_scheduler(
    request_scheduler: &dyn GenerationScheduler,
    prompt_tokens: Vec<u32>,
    gen_config: &GenerationConfig,
) -> Result<BatchedGenerationResult, String> {
    let prompt_len = prompt_tokens.len();
    let start = Instant::now();

    let mut handle = request_scheduler
        .submit(prompt_tokens, gen_config.clone())
        .map_err(|e| format!("scheduler submit failed: {}", e))?;

    let mut text = String::new();
    let mut completion_tokens = 0;
    let mut finish = FinishReason::Length;
    let mut prompt_eval_ms = 0u64;
    let mut last_token = start;
    let mut token_logprobs = gen_config.logprobs.then(Vec::new);

    while let Some(result) = handle.token_rx.recv().await {
        match result {
            Ok(token) => {
                let now = Instant::now();
                if completion_tokens == 0 {
                    prompt_eval_ms = now.duration_since(start).as_millis() as u64;
                } else {
                    metrics::record_itl(now.duration_since(last_token).as_secs_f64());
                }
                last_token = now;
                text.push_str(&token.text);
                completion_tokens += 1;
                let finish_reason = token.finish_reason;
                if let Some(tokens) = token_logprobs.as_mut() {
                    tokens.push(token);
                }
                if let Some(reason) = finish_reason {
                    finish = reason;
                    break;
                }
            }
            Err(e) => {
                return Err(format!("generation error: {}", e));
            }
        }
    }

    Ok(BatchedGenerationResult {
        text,
        prompt_tokens: prompt_len,
        completion_tokens,
        finish_reason: finish,
        prompt_eval_duration_ms: prompt_eval_ms,
        token_logprobs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    use boostr::inference::memory::CpuBlockAllocator;
    use boostr::inference::scheduler::SchedulerConfig;

    use crate::config::LatencySlo;
    use crate::engine::{GeneratedToken, RequestScheduler};
    use crate::server::generation::record_generation_metrics;
    use crate::server::metrics::names::{INTER_TOKEN_LATENCY, SLO_VIOLATIONS_TOTAL};
    use crate::server::test_recorder::TestRecorder;

    fn token(id: u32, finish: Option<FinishReason>) -> GeneratedToken {
        GeneratedToken {
            token_id: id,
            text: format!("t{id}"),
            logprob: None,
            top_logprobs: None,
            finish_reason: finish,
        }
    }

    /// A batched request: the first token after 40 ms, then two more 30 ms
    /// apart. TTFT is the first gap, not the whole request.
    #[tokio::test]
    async fn the_batched_path_measures_ttft_at_the_first_token_and_records_itl() {
        let recorder = TestRecorder::install();
        let config = SchedulerConfig {
            max_batch_size: 2,
            block_size: 4,
            ..Default::default()
        };
        let scheduler = RequestScheduler::new(CpuBlockAllocator::new(64, 4), config);
        let driver = Arc::clone(&scheduler);
        let feed = async move {
            // Wait for the submit, then play the engine's part.
            let seq_id = loop {
                if let Some(batch) = driver.schedule_batch().ok().flatten() {
                    break batch.prefill_sequences[0];
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            };
            let tx = driver.get_token_sender(seq_id).expect("sender");
            tokio::time::sleep(Duration::from_millis(40)).await;
            let _ = tx.send(Ok(token(1, None))).await;
            tokio::time::sleep(Duration::from_millis(30)).await;
            let _ = tx.send(Ok(token(2, None))).await;
            tokio::time::sleep(Duration::from_millis(30)).await;
            let _ = tx.send(Ok(token(3, Some(FinishReason::Eos)))).await;
        };
        let gen_config = GenerationConfig::default();
        let started = Instant::now();
        let (result, ()) = tokio::join!(
            generate_via_scheduler(scheduler.as_ref(), vec![5, 6, 7], &gen_config),
            feed
        );
        let total_ms = started.elapsed().as_millis() as u64;
        let result = result.expect("generation");
        assert_eq!(result.text, "t1t2t3");
        assert_eq!(result.completion_tokens, 3);
        // Sleeps only run long, so the two later gaps keep 60 ms after TTFT.
        let ttft = result.prompt_eval_duration_ms;
        assert!(ttft >= 40, "TTFT {ttft} ms");
        assert!(total_ms >= ttft + 55, "TTFT {ttft} ms of {total_ms} ms");
        let rendered = recorder.render();
        let count = crate::server::test_recorder::series_value(
            &rendered,
            &format!("{INTER_TOKEN_LATENCY}_count"),
            &[],
        );
        assert_eq!(count, Some(2.0), "{rendered}");
    }

    /// A future the timeout drops cancels its request: queued, it leaves the
    /// queue at once. Running, it leaves at the engine's next schedule. Both
    /// times every KV block is free again.
    #[tokio::test]
    async fn a_timed_out_request_is_cancelled_and_frees_its_blocks() {
        const BLOCKS: usize = 64;
        let config = SchedulerConfig {
            max_batch_size: 2,
            block_size: 4,
            ..Default::default()
        };
        let scheduler = RequestScheduler::new(CpuBlockAllocator::new(BLOCKS, 4), config);
        let free = |s: &RequestScheduler| s.block_stats().map(|b| b.free_blocks);
        let gen_config = GenerationConfig {
            max_tokens: 32,
            ..GenerationConfig::default()
        };

        // Queued: nothing schedules it, so the timeout drops it in the queue.
        let queued = tokio::time::timeout(
            Duration::from_millis(20),
            generate_via_scheduler(scheduler.as_ref(), vec![5, 6, 7], &gen_config),
        )
        .await;
        assert!(queued.is_err(), "no token arrives, so the timeout fires");
        assert!(!scheduler.has_work(), "the queued request is gone");
        assert_eq!(scheduler.queue_depth(), 0);
        assert_eq!(free(scheduler.as_ref()), Some(BLOCKS));

        // Running: admitted with blocks, one token streamed, then dropped.
        let driver = Arc::clone(&scheduler);
        let feed = tokio::spawn(async move {
            let seq_id = loop {
                if let Some(batch) = driver.schedule_batch().ok().flatten() {
                    break batch.prefill_sequences[0];
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            };
            if let Some(tx) = driver.get_token_sender(seq_id) {
                let _ = tx.send(Ok(token(1, None))).await;
            }
            seq_id
        });
        let running = tokio::time::timeout(
            Duration::from_millis(50),
            generate_via_scheduler(scheduler.as_ref(), vec![5, 6, 7], &gen_config),
        )
        .await;
        let seq_id = feed.await.expect("the feed task runs");
        assert!(
            running.is_err(),
            "the stream never finishes, so the timeout fires"
        );
        assert!(
            !scheduler.is_live(seq_id),
            "the drop marks the request cancelled"
        );
        assert!(
            free(scheduler.as_ref()).is_some_and(|f| f < BLOCKS),
            "the engine still holds its blocks"
        );

        // The engine's next schedule drops it before it builds a batch.
        let next = scheduler.schedule_batch().expect("schedule");
        assert!(next.is_none(), "the cancelled request takes no slot");
        assert!(!scheduler.has_work());
        assert_eq!(free(scheduler.as_ref()), Some(BLOCKS));
    }

    /// A request whose TTFT, ITL and end-to-end time break `server.slo`
    /// increments one violation series per metric, through the same calls
    /// the handlers make.
    #[test]
    fn a_breaching_request_increments_each_violation_series() {
        let recorder = TestRecorder::install();
        let slo = LatencySlo {
            ttft_p50_ms: None,
            ttft_p95_ms: Some(100),
            ttft_p99_ms: None,
            itl_p50_ms: None,
            itl_p95_ms: Some(10),
            itl_p99_ms: None,
            e2e_p99_ms: Some(1000),
        };
        let _policy = crate::server::slo::install(slo);
        crate::server::slo::publish();
        // TTFT 0.5 s and end-to-end 3 s, as a non-streaming handler reports.
        record_generation_metrics("m", &None, 4, 10, 500, Duration::from_secs(3));
        // An inter-token gap of 50 ms, as the token loops report.
        metrics::record_itl(0.05);

        for (metric, percentile) in [("ttft", "p95"), ("itl", "p95"), ("e2e", "p99")] {
            let value = recorder.value(
                SLO_VIOLATIONS_TOTAL,
                &[("metric", metric), ("percentile", percentile)],
            );
            assert_eq!(
                value,
                Some(1.0),
                "{metric} {percentile}\n{}",
                recorder.render()
            );
        }
    }
}
