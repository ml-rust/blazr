//! A batching loop on a dedicated OS thread.
//!
//! [`BatchEngine::run`](super::BatchEngine::run) and the lane engine's
//! `run` block their thread for each forward pass. On an async worker each
//! pins it for the server's whole life. [`EngineThread`] runs one loop
//! on its own named thread, on a current-thread Tokio runtime with the
//! timer on. The loop's channels reach the server's runtime as they would
//! from any task.
//!
//! [`ThreadedScheduler`] ties the thread to its scheduler. When the last
//! handle to the scheduler drops, at server stop, the thread gets its stop
//! signal and is joined.
//!
//! A loop that panics, or ends before its stop signal, leaves no engine to
//! serve the scheduler. The thread then fails the scheduler through
//! [`FailRequests`]: every request it carries gets an error item and loses
//! its state, the hold of its slot included, and every later submit is
//! refused. `GET /health` answers 503 with the reason.

use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use boostr::inference::memory::BlockAllocatorStats;
use tokio::sync::{mpsc::error::TrySendError, oneshot};

use super::request_registry::{GenerationScheduler, RequestHandle, TokenSender};
use crate::config::GenerationConfig;
use crate::engine::load::blocking::panic_message;

/// How long the error send waits on a reader whose channel is full.
const ERROR_SEND_LIMIT: Duration = Duration::from_secs(5);

/// A scheduler an engine thread drives, failed when its loop dies.
pub trait FailRequests: Send + Sync {
    /// Stop serving for `reason`: refuse every later submit and drop the
    /// state of every carried request, holds included. Returns each
    /// request's token channel, for the thread to send its error on.
    fn fail_all(&self, reason: &str) -> Vec<TokenSender>;
}

/// A batching loop running on its own thread.
///
/// Dropping the handle sends the stop signal and joins the thread. The loop
/// reads the signal while it waits for work and between steps, so the join
/// waits for at most one step.
pub struct EngineThread {
    stop: Option<oneshot::Sender<()>>,
    /// Set before the stop signal goes out, so a loop that ends after it
    /// counts as stopped.
    stopping: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl EngineThread {
    /// Start `name`, a thread that runs the future `work` builds. `work`
    /// gets the stop signal: a future that resolves once this handle drops.
    ///
    /// When the loop panics, ends before the stop signal, or cannot start,
    /// the thread fails `owner` with the reason.
    ///
    /// # Errors
    ///
    /// An error when the OS refuses the thread.
    pub fn spawn<F, Fut>(name: &str, owner: Arc<dyn FailRequests>, work: F) -> Result<Self>
    where
        F: FnOnce(std::pin::Pin<Box<dyn Future<Output = ()> + Send>>) -> Fut + Send + 'static,
        Fut: Future<Output = ()>,
    {
        let (stop, stopped) = oneshot::channel::<()>();
        let stopping = Arc::new(AtomicBool::new(false));
        let stop_seen = Arc::clone(&stopping);
        let label = name.to_string();
        let thread = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_time()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(e) => {
                        let reason =
                            format!("{label} did not start: its runtime did not build: {e}");
                        fail_owner(owner.as_ref(), &reason);
                        return;
                    }
                };
                // A dropped sender resolves the receiver too.
                let stop_signal = Box::pin(async move {
                    let _ = stopped.await;
                });
                let ran = catch_unwind(AssertUnwindSafe(|| runtime.block_on(work(stop_signal))));
                let reason = match ran {
                    Err(payload) => Some(format!(
                        "{label} panicked: {}",
                        panic_message(payload.as_ref())
                    )),
                    Ok(()) if !stop_seen.load(Ordering::SeqCst) => {
                        Some(format!("{label} ended before the server stopped"))
                    }
                    Ok(()) => None,
                };
                if let Some(reason) = reason {
                    fail_owner(owner.as_ref(), &reason);
                }
            })
            .map_err(|e| anyhow!("starting the {name} thread: {e}"))?;
        Ok(Self {
            stop: Some(stop),
            stopping,
            thread: Some(thread),
        })
    }
}

/// Fail `owner` for `reason` and send each request it carried one error.
///
/// A reader with room gets its error at once. A full channel gets up to
/// [`ERROR_SEND_LIMIT`] for its reader to make room. Each channel closes
/// after its error, so every stream ends.
fn fail_owner(owner: &dyn FailRequests, reason: &str) {
    tracing::error!("{reason}. Its scheduler serves no more requests; restart the server");
    let error = || anyhow!("the engine stopped: {reason}. Retry once the server restarts");
    let mut full = Vec::new();
    for tx in owner.fail_all(reason) {
        if let Err(TrySendError::Full(_)) = tx.try_send(Err(error())) {
            full.push(tx);
        }
    }
    if full.is_empty() {
        return;
    }
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            tracing::error!(
                "{} requests with full channels get no error item: no runtime for the send: {e}",
                full.len()
            );
            return;
        }
    };
    runtime.block_on(futures::future::join_all(full.iter().map(
        |tx| async move {
            let _ = tokio::time::timeout(ERROR_SEND_LIMIT, tx.send(Err(error()))).await;
        },
    )));
}

impl Drop for EngineThread {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::SeqCst);
        drop(self.stop.take());
        let Some(thread) = self.thread.take() else {
            return;
        };
        // The loop's own thread never joins itself.
        if thread.thread().id() == std::thread::current().id() {
            return;
        }
        if thread.join().is_err() {
            tracing::error!("an engine thread ended with a panic");
        }
    }
}

/// A scheduler whose engine loop runs on an [`EngineThread`]. It forwards
/// every call to the scheduler and stops the thread when it drops.
pub struct ThreadedScheduler {
    scheduler: Arc<dyn GenerationScheduler>,
    _engine: EngineThread,
}

impl ThreadedScheduler {
    /// `scheduler`, driven by the loop on `engine`.
    pub fn new(scheduler: Arc<dyn GenerationScheduler>, engine: EngineThread) -> Self {
        Self {
            scheduler,
            _engine: engine,
        }
    }
}

impl GenerationScheduler for ThreadedScheduler {
    fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        gen_config: GenerationConfig,
    ) -> Result<RequestHandle> {
        if let Some(reason) = self.fault() {
            bail!("the engine stopped: {reason}. Restart the server");
        }
        self.scheduler.submit(prompt_tokens, gen_config)
    }

    fn block_stats(&self) -> Option<BlockAllocatorStats> {
        self.scheduler.block_stats()
    }

    fn queue_depth(&self) -> usize {
        self.scheduler.queue_depth()
    }

    fn refusal(&self, gen_config: &GenerationConfig) -> Option<String> {
        self.scheduler.refusal(gen_config)
    }

    fn fault(&self) -> Option<String> {
        self.scheduler.fault()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::*;
    use crate::engine::requests::batch_test_rig::{greedy, rig, Rig, POOL};
    use crate::engine::test_support::{ticker, wait_past, WAIT_LIMIT};
    use crate::engine::{GeneratedToken, WorkHold};

    /// A thread over `engine`'s loop, failing `scheduler` when it dies.
    fn spawn_batch(
        scheduler: &Arc<super::super::RequestScheduler>,
        engine: super::super::BatchEngine<boostr::CpuRuntime>,
    ) -> Result<EngineThread> {
        let owner: Arc<dyn FailRequests> = scheduler.clone();
        EngineThread::spawn("blazr-batch", owner, move |stop| engine.run_until(stop))
    }

    /// A paged scheduler with no engine, for the tests that run their own loop.
    fn bare_scheduler() -> Arc<super::super::RequestScheduler> {
        let config = boostr::inference::scheduler::SchedulerConfig {
            max_batch_size: 2,
            max_batch_tokens: 4096,
            block_size: 4,
            ..Default::default()
        };
        let allocator = boostr::inference::memory::CpuBlockAllocator::new(16, 4);
        super::super::RequestScheduler::new(allocator, config)
    }

    /// Every item `handle` receives until its channel closes.
    async fn until_closed(handle: &mut RequestHandle) -> Result<Vec<Result<GeneratedToken>>> {
        let mut items = Vec::new();
        while let Some(item) = tokio::time::timeout(WAIT_LIMIT, handle.token_rx.recv()).await? {
            items.push(item);
        }
        Ok(items)
    }

    /// Each prefill chunk waits on the engine thread for the ticker to
    /// move. The ticker runs on the one runtime worker, so it moves only
    /// while the batch loop holds no worker. Dropping the scheduler then
    /// stops and joins the thread.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_batch_engine_thread_leaves_the_runtime_worker_free() -> Result<()> {
        let Rig {
            scheduler, engine, ..
        } = rig(POOL)?;
        let mut engine = engine.with_prefill_chunk_size(1);
        let ticks = Arc::new(AtomicUsize::new(0));
        let ticking = ticker(&ticks);
        let saw_tick = Arc::new(Mutex::new(Vec::new()));
        engine.after_chunk = Some({
            let (ticks, saw_tick) = (Arc::clone(&ticks), Arc::clone(&saw_tick));
            Box::new(move |_: usize| {
                let passed = wait_past(&ticks, ticks.load(Ordering::SeqCst));
                if let Ok(mut saw) = saw_tick.lock() {
                    saw.push(passed);
                }
            })
        });
        let thread = spawn_batch(&scheduler, engine)?;
        let threaded = Arc::new(ThreadedScheduler::new(scheduler, thread));

        let mut handle = threaded.submit(vec![1, 2, 3], greedy(2))?;
        let mut tokens = 0;
        while let Some(item) = tokio::time::timeout(WAIT_LIMIT, handle.token_rx.recv()).await? {
            let token = item?;
            tokens += 1;
            if token.finish_reason.is_some() {
                break;
            }
        }
        drop(handle);
        ticking.abort();

        assert_eq!(tokens, 2);
        let saw = saw_tick.lock().map_err(|e| anyhow!("{e}"))?.clone();
        assert_eq!(saw, [true; 3], "the ticker stalled during a prefill chunk");
        // The last handle drops off the runtime, since the drop joins.
        let joined = tokio::task::spawn_blocking(move || drop(threaded));
        tokio::time::timeout(WAIT_LIMIT, joined).await??;
        Ok(())
    }

    /// A loop that waits for work ends on the stop signal, and the drop
    /// returns once the thread has exited.
    #[test]
    fn dropping_the_handle_stops_an_idle_loop() -> Result<()> {
        let scheduler = bare_scheduler();
        let owner: Arc<dyn FailRequests> = scheduler.clone();
        let exited = Arc::new(AtomicUsize::new(0));
        let flag = Arc::clone(&exited);
        let thread = EngineThread::spawn("blazr-test-engine", owner, move |stop| async move {
            stop.await;
            flag.fetch_add(1, Ordering::SeqCst);
        })?;
        drop(thread);
        assert_eq!(exited.load(Ordering::SeqCst), 1);
        assert_eq!(scheduler.fault(), None, "a stopped loop is no fault");
        Ok(())
    }

    /// The first prefill chunk waits until both requests are queued, then
    /// panics. Each request ends with one error item, its hold drops, and
    /// the scheduler refuses the next submit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_panicking_engine_fails_every_request_and_refuses_new_ones() -> Result<()> {
        let Rig {
            scheduler, engine, ..
        } = rig(POOL)?;
        let mut engine = engine.with_prefill_chunk_size(1);
        let (go, gate) = std::sync::mpsc::channel::<()>();
        engine.after_chunk = Some(Box::new(move |_: usize| {
            let _ = gate.recv_timeout(WAIT_LIMIT);
            panic!("injected engine panic");
        }));
        let thread = spawn_batch(&scheduler, engine)?;
        let threaded = Arc::new(ThreadedScheduler::new(scheduler.clone(), thread));

        let probe = Arc::new(());
        let hold = WorkHold::new(Arc::clone(&probe));
        let (mut first, mut second) = WorkHold::within(Some(hold), async {
            let first = threaded.submit((1..=6).collect(), greedy(4))?;
            let second = threaded.submit(vec![7, 8, 9], greedy(4))?;
            anyhow::Ok((first, second))
        })
        .await?;
        assert_eq!(Arc::strong_count(&probe), 2, "the requests hold the guard");
        go.send(()).map_err(|e| anyhow!("{e}"))?;

        for handle in [&mut first, &mut second] {
            let items = until_closed(handle).await?;
            let Some(Err(e)) = items.last() else {
                return Err(anyhow!("the stream closed without an error item"));
            };
            let message = format!("{e:#}");
            assert!(message.contains("blazr-batch panicked"), "{message}");
            assert!(message.contains("injected engine panic"), "{message}");
            assert_eq!(items.iter().filter(|i| i.is_err()).count(), 1);
        }
        assert_eq!(Arc::strong_count(&probe), 1, "every hold dropped");
        assert!(!scheduler.registry.contains(first.seq_id));
        assert_eq!(scheduler.queue_depth(), 0);

        let fault = threaded.fault().ok_or_else(|| anyhow!("no fault is set"))?;
        assert!(fault.contains("injected engine panic"), "{fault}");
        let refused = threaded.submit(vec![1, 2], greedy(2)).err();
        assert!(
            refused.is_some_and(|e| e.to_string().contains("the engine stopped")),
            "a faulted scheduler refuses the submit"
        );
        let direct = scheduler.submit(vec![1, 2], greedy(2)).err();
        assert!(direct.is_some_and(|e| e.to_string().contains("stopped serving")));
        let joined = tokio::task::spawn_blocking(move || drop(threaded));
        tokio::time::timeout(WAIT_LIMIT, joined).await??;
        Ok(())
    }

    /// A loop that returns before the stop signal fails its scheduler too.
    #[test]
    fn a_loop_that_ends_on_its_own_fails_its_scheduler() -> Result<()> {
        let scheduler = bare_scheduler();
        let mut handle = scheduler.submit(vec![1, 2, 3], greedy(2))?;
        let owner: Arc<dyn FailRequests> = scheduler.clone();
        let thread = EngineThread::spawn("blazr-test-engine", owner, |_stop| async {})?;
        let item = handle.token_rx.blocking_recv();
        assert!(
            item.is_some_and(|i| i.is_err()),
            "the request gets an error"
        );
        assert!(handle.token_rx.blocking_recv().is_none());
        let fault = scheduler
            .fault()
            .ok_or_else(|| anyhow!("no fault is set"))?;
        assert!(fault.contains("ended before the server stopped"), "{fault}");
        drop(thread);
        Ok(())
    }
}
