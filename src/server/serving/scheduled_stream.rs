//! Streaming text generation through a batching scheduler.
//!
//! The scheduler takes the prompt, and each token goes on as it arrives.
//! `routed_stream` picks this path or the executor's for a request.

use tokio::sync::mpsc;

use crate::config::GenerationConfig;
use crate::engine::{FinishReason, GenerationScheduler};
use crate::server::stream_metrics::StreamMetrics;
use crate::server::streaming::StreamToken;

/// Submit a request to `scheduler` and forward each token to `tx`.
///
/// The scheduler's engine applies the stop sequences, the `max_tokens`
/// clamp and the finish reasons, so no second stop pass runs here. With
/// `gen_config.logprobs` set, each item carries its token's logprobs.
/// The item that carries the finish reason carries the token counts.
///
/// A closed `tx` means the client left. This function then returns, which
/// drops the request's handle, and the handle's `Drop` cancels the request.
/// The metrics match the executor stream's.
pub async fn stream_via_scheduler(
    scheduler: &dyn GenerationScheduler,
    prompt_tokens: Vec<u32>,
    gen_config: GenerationConfig,
    tx: mpsc::Sender<StreamToken>,
) {
    let logprobs = gen_config.logprobs;
    let mut metrics = StreamMetrics::start(prompt_tokens.len());
    let mut handle = match scheduler.submit(prompt_tokens, gen_config) {
        Ok(handle) => handle,
        Err(e) => {
            tracing::error!("Batched generation submit error: {}", e);
            let message = format!("scheduler submit failed: {}", e);
            let _ = tx.send(StreamToken::error(message)).await;
            return;
        }
    };

    loop {
        let next = tokio::select! {
            biased;
            next = handle.token_rx.recv() => next,
            () = tx.closed() => break,
        };
        match next {
            Some(Ok(token)) => {
                metrics.token();
                let finished = token.finish_reason.is_some();
                let mut item = StreamToken::text(token.text.clone(), token.finish_reason)
                    .with_logprobs(&token, logprobs);
                if finished {
                    item = item.with_usage(metrics.usage());
                }
                if tx.send(item).await.is_err() || finished {
                    break;
                }
            }
            Some(Err(e)) => {
                tracing::error!("Batched generation error during streaming: {}", e);
                let _ = tx.send(StreamToken::error(e.to_string())).await;
                return;
            }
            None => {
                // The engine closed the stream with no finish token.
                let close = StreamToken::text(String::new(), Some(FinishReason::Length))
                    .with_usage(metrics.usage());
                let _ = tx.send(close).await;
                break;
            }
        }
    }
    metrics.finish();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::{anyhow, Result};

    use crate::engine::batch_test_rig::{
        alone_tokens, drain, free_blocks, greedy, rig, tick, Rig, POOL,
    };
    use crate::engine::{GeneratedToken, RequestHandle, RequestRegistry};
    use crate::server::completion_stream::create_completion_stream;
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    use crate::server::routed_stream::stream_routed;
    use crate::server::stream_usage::StreamUsage;
    use crate::server::streaming::sse_test;

    /// Logprobs of the two paths agree to this much.
    const LOGPROB_TOL: f32 = 1e-5;
    /// Yields a test allows the stream task before it calls the submit lost.
    const YIELD_LIMIT: usize = 64;

    /// Yield until the stream task has submitted its request to `rig`.
    async fn submitted(rig: &Rig) -> Result<()> {
        for _ in 0..YIELD_LIMIT {
            if rig.scheduler.has_work() {
                return Ok(());
            }
            tokio::task::yield_now().await;
        }
        Err(anyhow!("the stream task never submitted its request"))
    }

    /// Stream `prompt` through `rig`'s scheduler and collect every item.
    async fn stream(
        rig: &mut Rig,
        prompt: &[u32],
        cfg: GenerationConfig,
    ) -> Result<Vec<StreamToken>> {
        let (tx, mut rx) = mpsc::channel(64);
        let scheduler = Arc::clone(&rig.scheduler);
        let prompt = prompt.to_vec();
        let task = tokio::spawn(async move {
            stream_via_scheduler(scheduler.as_ref(), prompt, cfg, tx).await;
        });
        submitted(rig).await?;
        drain(rig).await?;
        task.await?;
        let mut items = Vec::new();
        while let Some(item) = rx.recv().await {
            items.push(item);
        }
        Ok(items)
    }

    /// The text and finish reason a client reads from `items`.
    fn read(items: &[StreamToken]) -> Result<(String, FinishReason)> {
        if let Some(e) = items.iter().find_map(|t| t.error.as_ref()) {
            return Err(anyhow!("the stream sent an error: {e}"));
        }
        let text = items.iter().map(|t| t.text.as_str()).collect();
        let finish = items
            .last()
            .and_then(|t| t.finish_reason)
            .ok_or_else(|| anyhow!("the stream sent no finish reason"))?;
        Ok((text, finish))
    }

    /// The non-streaming single-sequence result for `prompt`.
    async fn non_streaming(
        rig: &Rig,
        prompt: &[u32],
        cfg: &GenerationConfig,
    ) -> Result<(String, FinishReason)> {
        let result = rig.executor.generate_text(prompt, cfg).await?;
        Ok((result.text, result.finish_reason))
    }

    #[tokio::test]
    async fn streamed_text_and_finish_match_the_non_streaming_result() -> Result<()> {
        let mut rig = rig(POOL)?;
        let prompt = [1, 2, 3];
        for cfg in [greedy(6), greedy(1)] {
            let expected = non_streaming(&rig, &prompt, &cfg).await?;
            let items = stream(&mut rig, &prompt, cfg).await?;
            assert_eq!(read(&items)?, expected);
        }
        let one = stream(&mut rig, &prompt, greedy(1)).await?;
        assert_eq!(read(&one)?.1, FinishReason::Length);

        // Every id under 32 is one ASCII character, so a byte range is a token range.
        let alone = alone_tokens(&rig.executor, &prompt, &greedy(6)).await?;
        let text: String = alone.iter().map(|t| t.text.as_str()).collect();
        if text.len() != alone.len() || !text.is_ascii() {
            return Err(anyhow!("{text:?} is not one ASCII character per token"));
        }
        let split_stop = GenerationConfig {
            stop_sequences: vec![text[1..3].to_string()],
            ..greedy(6)
        };
        let expected = non_streaming(&rig, &prompt, &split_stop).await?;
        assert_eq!(expected.1, FinishReason::Stop);
        let items = stream(&mut rig, &prompt, split_stop).await?;
        assert_eq!(read(&items)?, expected);
        assert!(!rig.scheduler.has_work());
        Ok(())
    }

    /// The final item's counts equal the non-streaming `usage`, on a length
    /// finish and on a stop sequence. The SSE stream sends them as its usage chunk.
    #[tokio::test]
    async fn streamed_usage_matches_the_non_streaming_counts() -> Result<()> {
        let mut rig = rig(POOL)?;
        let prompt = [1, 2, 3];
        let alone = alone_tokens(&rig.executor, &prompt, &greedy(6)).await?;
        let text: String = alone.iter().map(|t| t.text.as_str()).collect();
        if text.len() != alone.len() || !text.is_ascii() {
            return Err(anyhow!("{text:?} is not one ASCII character per token"));
        }
        let split_stop = GenerationConfig {
            stop_sequences: vec![text[1..3].to_string()],
            ..greedy(6)
        };
        for cfg in [greedy(6), greedy(1), split_stop] {
            let result = rig.executor.generate_text(&prompt, &cfg).await?;
            let want = StreamUsage::new(result.prompt_tokens, result.completion_tokens);
            let items = stream(&mut rig, &prompt, cfg).await?;
            let (last, rest) = items
                .split_last()
                .ok_or_else(|| anyhow!("the stream sent no item"))?;
            assert_eq!(last.usage, Some(want), "{:?}", result.finish_reason);
            assert!(rest.iter().all(|t| t.usage.is_none()));

            let tokens = Box::pin(futures::stream::iter(items));
            let sse = create_completion_stream("c".into(), "m".into(), tokens, true);
            let events = sse_test::events(sse).await?;
            sse_test::assert_usage_tail(&events, "text_completion.chunk", want);
        }
        Ok(())
    }

    #[tokio::test]
    async fn streamed_logprobs_match_the_single_path_and_only_when_asked() -> Result<()> {
        let mut rig = rig(POOL)?;
        let prompt = [1, 2, 3];
        let cfg = GenerationConfig {
            logprobs: true,
            top_logprobs: 3,
            ..greedy(4)
        };
        let expected = alone_tokens(&rig.executor, &prompt, &cfg).await?;
        let items = stream(&mut rig, &prompt, cfg).await?;
        assert_eq!(items.len(), expected.len());
        for (item, want) in items.iter().zip(&expected) {
            let got = item.logprob.as_ref().ok_or_else(|| anyhow!("no logprob"))?;
            let want_lp = want.logprob.ok_or_else(|| anyhow!("no expected logprob"))?;
            assert_eq!(got.token_id, want.token_id);
            assert!(
                (got.logprob - want_lp).abs() <= LOGPROB_TOL,
                "{} vs {want_lp}",
                got.logprob
            );
            let (got_top, want_top) = (item.top_logprobs.as_deref(), want.top_logprobs.as_deref());
            let (Some(got_top), Some(want_top)) = (got_top, want_top) else {
                return Err(anyhow!("missing top logprobs: {got_top:?} vs {want_top:?}"));
            };
            assert_eq!(got_top.len(), 3);
            assert_eq!(got_top.len(), want_top.len());
            for (g, w) in got_top.iter().zip(want_top) {
                assert_eq!(g.token_id, w.token_id);
                assert!((g.logprob - w.logprob).abs() <= LOGPROB_TOL);
            }
        }

        let plain = stream(&mut rig, &prompt, greedy(4)).await?;
        assert!(plain
            .iter()
            .all(|t| t.logprob.is_none() && t.top_logprobs.is_none()));
        Ok(())
    }

    #[tokio::test]
    async fn a_dropped_receiver_cancels_the_request_and_frees_its_blocks() -> Result<()> {
        let mut rig = rig(POOL)?;
        let start = free_blocks(&rig)?;
        let (tx, mut rx) = mpsc::channel(64);
        let scheduler = Arc::clone(&rig.scheduler);
        let task = tokio::spawn(async move {
            stream_via_scheduler(scheduler.as_ref(), vec![1, 2, 3], greedy(12), tx).await;
        });
        submitted(&rig).await?;
        tick(&mut rig).await;
        let first = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await?;
        assert!(
            first.is_some_and(|t| t.finish_reason.is_none()),
            "one token arrives"
        );
        assert!(
            free_blocks(&rig)? < start,
            "the running request holds blocks"
        );

        drop(rx);
        tokio::time::timeout(Duration::from_secs(5), task).await??;
        drain(&mut rig).await?;
        assert!(!rig.scheduler.has_work());
        assert_eq!(free_blocks(&rig)?, start);
        Ok(())
    }

    /// A scheduler whose engine sends one token, then an error.
    struct FailingScheduler {
        registry: RequestRegistry,
    }

    impl GenerationScheduler for FailingScheduler {
        fn submit(&self, prompt: Vec<u32>, cfg: GenerationConfig) -> Result<RequestHandle> {
            let id = self.registry.next_id();
            let handle = self.registry.insert(id, prompt, cfg)?;
            let tx = self
                .registry
                .token_sender(id)
                .ok_or_else(|| anyhow!("the request is gone"))?;
            let token = GeneratedToken {
                token_id: 7,
                text: "a".to_string(),
                logprob: None,
                top_logprobs: None,
                finish_reason: None,
            };
            tx.try_send(Ok(token))
                .map_err(|_| anyhow!("token channel full"))?;
            tx.try_send(Err(anyhow!("decode step error")))
                .map_err(|_| anyhow!("token channel full"))?;
            Ok(handle)
        }
    }

    #[tokio::test]
    async fn an_engine_error_mid_stream_becomes_an_sse_error_event() -> Result<()> {
        let scheduler = FailingScheduler {
            registry: RequestRegistry::new(),
        };
        let (tx, rx) = mpsc::channel(64);
        stream_via_scheduler(&scheduler, vec![1, 2], greedy(8), tx).await;
        let tokens = Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx));
        let sse = create_completion_stream("c".into(), "m".into(), tokens, false);
        let events = sse_test::events(sse).await?;

        assert_eq!(events.len(), 3, "{events:?}");
        assert_eq!(events[0]["choices"][0]["delta"]["text"], "a");
        assert_eq!(events[1]["error"]["message"], "decode step error");
        assert_eq!(events[1]["error"]["type"], "server_error");
        assert_eq!(events[2], serde_json::json!("[DONE]"));
        Ok(())
    }

    /// `stream_routed` takes the scheduler when the route has one, else the
    /// executor. The scheduler's queue shows which path ran.
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    #[tokio::test]
    async fn stream_routed_takes_the_scheduler_only_when_batching_is_on() -> Result<()> {
        use crate::server::handlers::{ServeReplica, Serving};

        let mut rig = rig(POOL)?;
        let prompt = [1, 2, 3];
        let expected = non_streaming(&rig, &prompt, &greedy(4)).await?;
        let executor = Arc::clone(&rig.executor);
        let route = |scheduler: Option<Arc<dyn GenerationScheduler>>| {
            let replica = ServeReplica {
                executor: Arc::clone(&executor),
                scheduler,
                device: "cpu".to_string(),
            };
            Serving::Single(replica).route(Arc::clone(&executor), &prompt)
        };

        // Batching on: the request waits in the scheduler until the engine ticks.
        let batching: Arc<dyn GenerationScheduler> = rig.scheduler.clone();
        let batched = route(Some(batching));
        let (tx, mut rx) = mpsc::channel(64);
        let task = tokio::spawn(async move {
            stream_routed(&batched, prompt.to_vec(), greedy(4), tx).await;
        });
        submitted(&rig).await?;
        drain(&mut rig).await?;
        task.await?;
        let mut items = Vec::new();
        while let Some(item) = rx.recv().await {
            items.push(item);
        }
        assert_eq!(read(&items)?, expected);

        // Batching off: the executor streams alone and the scheduler sees nothing.
        let single = route(None);
        let (tx, mut rx) = mpsc::channel(64);
        stream_routed(&single, prompt.to_vec(), greedy(4), tx).await;
        assert!(!rig.scheduler.has_work());
        assert_eq!(rig.scheduler.stats().map(|s| s.waiting_count), Some(0));
        let mut items = Vec::new();
        while let Some(item) = rx.recv().await {
            items.push(item);
        }
        assert_eq!(read(&items)?, expected);
        Ok(())
    }
}
