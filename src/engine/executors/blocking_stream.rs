//! Single-sequence generation on a dedicated decode thread.
//!
//! A forward pass blocks its thread for the whole step. A stream polled on
//! an async worker pins that worker, so health probes, SSE writes and
//! accepts on it wait. Each generation here runs on its own named OS
//! thread, `blazr-decode`. The thread builds the stream, drives it, and
//! sends each item on a bounded channel. The stream never leaves that
//! thread, so its KV lease, cache and tensor-parallel guard drop there.
//!
//! The thread drives the stream on a current-thread Tokio runtime with the
//! timer on. The send timeout needs the timer. The `tokio::sync` guards the
//! stream awaits need only a waker. CUDA needs no setup on the new thread:
//! cudarc makes its context current in every safe call, and numr does the
//! same before each raw driver call.
//!
//! A WebAssembly target starts no threads, so there the stream runs on the
//! caller.

use anyhow::Result;
use tokio::sync::mpsc;

use boostr::{DType, Runtime};

use super::decode_stream::decode_output::{DecodeOutput, RawTokens};
use super::executor::{EngineClient, Executor};
use super::generation::generate_text::collect_generation;
use super::types::GenerationResult;

pub(crate) use super::decode_stream::decode_job::DecodeJob;
#[cfg(test)]
pub(crate) use super::decode_stream::decode_job::StepHook;

/// Items a decode channel buffers: the capacity of the handlers' stream channels.
pub(crate) const DECODE_CHANNEL_CAPACITY: usize = 32;

/// Run `job` on a decode thread and send each item `out` maps to `tx`.
///
/// Returns once the thread exits. A guard the caller holds across this
/// call, such as an admission permit, outlives the decode. A caller that
/// drops this future instead leaves the thread to stop within one token,
/// when it finds `tx`'s receiver gone.
pub(crate) async fn stream_on_decode_thread<R, O>(
    job: DecodeJob<R>,
    out: O,
    tx: mpsc::Sender<O::Item>,
) where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R>,
    Executor<R>: Send + Sync + 'static,
    O: DecodeOutput,
{
    #[cfg(target_family = "wasm")]
    {
        // No thread and no timer: the caller drives the stream.
        let mut out = out;
        let job = DecodeJob { idle: None, ..job };
        job.pump(&mut out, &tx).await;
    }
    #[cfg(not(target_family = "wasm"))]
    {
        let failure_tx = tx.clone();
        match super::decode_stream::decode_thread::spawn_decode(job, out, tx) {
            Ok(exited) => {
                drop(failure_tx);
                // The thread drops its exit sender last, after the stream.
                let _ = exited.await;
            }
            Err(e) => {
                let message = format!("the decode thread did not start: {e}");
                tracing::error!("{message}");
                let _ = failure_tx.send(O::failure(message)).await;
            }
        }
    }
}

/// Run `job` on a decode thread and collect its tokens into one
/// `GenerationResult`, as [`collect_generation`] reads them.
///
/// # Errors
///
/// The stream's error, a panic on the decode thread, or a decode thread
/// that did not start.
pub(crate) async fn collect_on_decode_thread<R>(job: DecodeJob<R>) -> Result<GenerationResult>
where
    R: Runtime<DType = DType>,
    R::Client: EngineClient<R>,
    Executor<R>: Send + Sync + 'static,
{
    let prompt_len = job.prompt.len();
    let config = job.config.clone();
    let (tx, mut rx) = mpsc::channel(DECODE_CHANNEL_CAPACITY);
    let tokens = futures::stream::poll_fn(move |cx| rx.poll_recv(cx));
    // A collect that ends early, on a stop sequence, drops the receiver,
    // so the thread stops within one token.
    let (result, ()) = futures::join!(
        collect_generation(tokens, prompt_len, &config),
        stream_on_decode_thread(job, RawTokens, tx),
    );
    result
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use anyhow::anyhow;
    use boostr::CpuRuntime;

    use super::*;
    use crate::engine::decode_stream::decode_output::{Emit, StreamEnd};
    use crate::engine::requests::batch_test_rig::{greedy, POOL, SEED};
    use crate::engine::test_support::{
        free_pool_blocks as free_blocks, seeded_executor, ticker, wait_past, WAIT_LIMIT,
    };
    use crate::engine::{GeneratedToken, WorkHold};

    /// Engine tokens tagged with the stream they came from.
    struct Tagged(usize);

    impl DecodeOutput for Tagged {
        type Item = (usize, Result<GeneratedToken>);

        fn token(&mut self, token: Result<GeneratedToken>) -> Emit<Self::Item> {
            Emit::Send((self.0, token))
        }

        fn end(&mut self, _why: StreamEnd) -> Option<Self::Item> {
            None
        }

        fn failure(message: String) -> Self::Item {
            (usize::MAX, Err(anyhow!(message)))
        }
    }

    /// Every item `rx` receives until its senders drop.
    async fn drain<T>(rx: &mut mpsc::Receiver<T>) -> Vec<T> {
        let mut items = Vec::new();
        while let Some(item) = rx.recv().await {
            items.push(item);
        }
        items
    }

    fn executor(paged: bool) -> Result<Arc<Executor<CpuRuntime>>> {
        Ok(Arc::new(seeded_executor(SEED, paged.then_some(POOL))?))
    }

    /// Each decode step waits on its thread for the ticker to move. The
    /// ticker runs on the one runtime worker, so it moves only while no
    /// forward pass holds that worker.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_slow_decode_leaves_the_runtime_worker_free() -> Result<()> {
        let ticks = Arc::new(AtomicUsize::new(0));
        let ticking = ticker(&ticks);
        let saw_tick = Arc::new(Mutex::new(Vec::new()));
        let hook: StepHook = {
            let (ticks, saw_tick) = (Arc::clone(&ticks), Arc::clone(&saw_tick));
            Arc::new(move |_| {
                let passed = wait_past(&ticks, ticks.load(Ordering::SeqCst));
                if let Ok(mut saw) = saw_tick.lock() {
                    saw.push(passed);
                }
            })
        };
        let job = DecodeJob::text(executor(false)?, vec![1, 2, 3], greedy(3)).with_step_hook(hook);
        let (tx, mut rx) = mpsc::channel(DECODE_CHANNEL_CAPACITY);
        // Spawned like a handler's stream task, which runs on the worker.
        let decode = tokio::spawn(stream_on_decode_thread(job, RawTokens, tx));
        let tokens = drain(&mut rx).await;
        decode.await?;
        ticking.abort();

        assert_eq!(tokens.len(), 3);
        assert!(tokens.iter().all(Result::is_ok));
        let saw = saw_tick.lock().map_err(|e| anyhow!("{e}"))?.clone();
        assert_eq!(saw, [true; 3], "the ticker stalled during a decode step");
        Ok(())
    }

    /// Two decodes step in lockstep: each step waits for the other stream
    /// to reach the same step. Their tokens arrive interleaved.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn two_streams_progress_together() -> Result<()> {
        let exec = executor(false)?;
        let steps = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
        let waited = Arc::new(Mutex::new(Vec::new()));
        let (tx, mut rx) = mpsc::channel(DECODE_CHANNEL_CAPACITY);
        let mut decodes = Vec::new();
        for me in 0..2 {
            let (own, other) = (Arc::clone(&steps[me]), Arc::clone(&steps[1 - me]));
            let waited = Arc::clone(&waited);
            let hook: StepHook = Arc::new(move |step| {
                own.store(step + 1, Ordering::SeqCst);
                let passed = wait_past(&other, step);
                if let Ok(mut waited) = waited.lock() {
                    waited.push(passed);
                }
            });
            let job = DecodeJob::text(Arc::clone(&exec), vec![1, 2, 3], greedy(4));
            let out = Tagged(me);
            let run = stream_on_decode_thread(job.with_step_hook(hook), out, tx.clone());
            decodes.push(tokio::spawn(run));
        }
        drop(tx);
        let order: Vec<usize> = drain(&mut rx)
            .await
            .into_iter()
            .map(|(tag, _)| tag)
            .collect();
        for decode in decodes {
            decode.await?;
        }

        let waited = waited.lock().map_err(|e| anyhow!("{e}"))?.clone();
        assert_eq!(waited, [true; 8], "a stream waited out its peer");
        assert_eq!(order.len(), 8);
        let first = |tag| order.iter().position(|&t| t == tag);
        let last = |tag| order.iter().rposition(|&t| t == tag);
        assert!(first(1) < last(0) && first(0) < last(1), "{order:?}");
        Ok(())
    }

    /// The reader takes one token and leaves. The decode stops at its next
    /// step, the thread exits, and the KV lease returns its blocks.
    #[tokio::test]
    async fn a_dropped_receiver_stops_the_decode_and_frees_its_blocks() -> Result<()> {
        let exec = executor(true)?;
        let steps = Arc::new(AtomicUsize::new(0));
        let left = Arc::new(AtomicUsize::new(0));
        let hook: StepHook = {
            let (steps, left) = (Arc::clone(&steps), Arc::clone(&left));
            Arc::new(move |step| {
                steps.fetch_add(1, Ordering::SeqCst);
                if step == 1 {
                    wait_past(&left, 0);
                }
            })
        };
        let job = DecodeJob::text(Arc::clone(&exec), vec![1, 2, 3], greedy(10));
        let (tx, mut rx) = mpsc::channel(DECODE_CHANNEL_CAPACITY);
        let decode = tokio::spawn(stream_on_decode_thread(
            job.with_step_hook(hook),
            RawTokens,
            tx,
        ));
        let first = rx.recv().await.ok_or_else(|| anyhow!("no first token"))?;
        first?;
        assert!(
            free_blocks(&exec)? < POOL.blocks,
            "the decode holds no lease"
        );
        drop(rx);
        left.fetch_add(1, Ordering::SeqCst);
        tokio::time::timeout(WAIT_LIMIT, decode).await??;

        assert_eq!(
            steps.load(Ordering::SeqCst),
            2,
            "the decode ran past the drop"
        );
        assert_eq!(free_blocks(&exec)?, POOL.blocks);
        Ok(())
    }

    /// The caller's future drops after the first token, as a handler's does
    /// when its client leaves. The decode thread keeps the caller's hold
    /// until it exits, one step later.
    #[tokio::test]
    async fn the_decode_thread_keeps_the_hold_until_it_exits() -> Result<()> {
        let exec = executor(false)?;
        let left = Arc::new(AtomicUsize::new(0));
        let hook: StepHook = {
            let left = Arc::clone(&left);
            Arc::new(move |step| {
                if step == 1 {
                    wait_past(&left, 0);
                }
            })
        };
        let probe = Arc::new(());
        let hold = WorkHold::new(Arc::clone(&probe));
        let job = DecodeJob::text(exec, vec![1, 2, 3], greedy(10)).with_step_hook(hook);
        let (tx, mut rx) = mpsc::channel(DECODE_CHANNEL_CAPACITY);
        let caller = tokio::spawn(WorkHold::within(
            Some(hold),
            stream_on_decode_thread(job, RawTokens, tx),
        ));
        rx.recv().await.ok_or_else(|| anyhow!("no first token"))??;

        caller.abort();
        let _ = caller.await;
        drop(rx);
        assert_eq!(
            Arc::strong_count(&probe),
            2,
            "the hold dropped while the decode still ran"
        );
        left.fetch_add(1, Ordering::SeqCst);
        let deadline = std::time::Instant::now() + WAIT_LIMIT;
        while Arc::strong_count(&probe) > 1 {
            if std::time::Instant::now() >= deadline {
                return Err(anyhow!("the decode thread never dropped the hold"));
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        Ok(())
    }

    /// A reader that never reads: the first token fills the channel, the
    /// second waits out the idle timeout, and the thread exits.
    #[tokio::test]
    async fn a_reader_that_takes_nothing_frees_the_thread() -> Result<()> {
        let exec = executor(true)?;
        let steps = Arc::new(AtomicUsize::new(0));
        let hook: StepHook = {
            let steps = Arc::clone(&steps);
            Arc::new(move |_| {
                steps.fetch_add(1, Ordering::SeqCst);
            })
        };
        let job = DecodeJob::text(Arc::clone(&exec), vec![1, 2, 3], greedy(10))
            .with_step_hook(hook)
            .with_idle_timeout(Some(Duration::from_millis(20)));
        let (tx, rx) = mpsc::channel(1);
        tokio::time::timeout(WAIT_LIMIT, stream_on_decode_thread(job, RawTokens, tx)).await?;

        assert_eq!(steps.load(Ordering::SeqCst), 2);
        assert_eq!(rx.len(), 1, "the first token waits in the channel");
        assert_eq!(free_blocks(&exec)?, POOL.blocks);
        Ok(())
    }

    /// A panic mid-decode ends the stream with an error item. The unwind
    /// drops the lease, and the runtime runs the next request.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_decode_panic_sends_an_error_and_the_runtime_lives() -> Result<()> {
        let exec = executor(true)?;
        let hook: StepHook = Arc::new(|step| {
            if step == 1 {
                panic!("injected decode panic");
            }
        });
        let job = DecodeJob::text(Arc::clone(&exec), vec![1, 2, 3], greedy(4)).with_step_hook(hook);
        let (tx, mut rx) = mpsc::channel(DECODE_CHANNEL_CAPACITY);
        tokio::spawn(stream_on_decode_thread(job, RawTokens, tx)).await?;
        let items = drain(&mut rx).await;

        assert_eq!(items.len(), 2, "one token, then the error");
        assert!(items[0].is_ok());
        let error = items[1].as_ref().err().map(ToString::to_string);
        assert!(
            error
                .as_deref()
                .is_some_and(|e| e.contains("injected decode panic")),
            "{error:?}"
        );
        assert_eq!(free_blocks(&exec)?, POOL.blocks);
        let again = exec.generate_text(&[1, 2, 3], &greedy(2)).await?;
        assert_eq!(again.completion_tokens, 2);
        Ok(())
    }
}
