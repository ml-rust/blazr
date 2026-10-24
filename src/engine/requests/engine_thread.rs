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

use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use std::thread::JoinHandle;

use anyhow::{anyhow, Result};
use boostr::inference::memory::BlockAllocatorStats;
use tokio::sync::oneshot;

use super::request_registry::{GenerationScheduler, RequestHandle};
use crate::config::GenerationConfig;
use crate::engine::load::blocking::panic_message;

/// A batching loop running on its own thread.
///
/// Dropping the handle sends the stop signal and joins the thread. The loop
/// reads the signal while it waits for work and between steps, so the join
/// waits for at most one step.
pub struct EngineThread {
    stop: Option<oneshot::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl EngineThread {
    /// Start `name`, a thread that runs the future `work` builds. `work`
    /// gets the stop signal: a future that resolves once this handle drops.
    ///
    /// # Errors
    ///
    /// An error when the OS refuses the thread.
    pub fn spawn<F, Fut>(name: &str, work: F) -> Result<Self>
    where
        F: FnOnce(std::pin::Pin<Box<dyn Future<Output = ()> + Send>>) -> Fut + Send + 'static,
        Fut: Future<Output = ()>,
    {
        let (stop, stopped) = oneshot::channel::<()>();
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
                        tracing::error!("{label} could not build its runtime: {e}");
                        return;
                    }
                };
                // A dropped sender resolves the receiver too.
                let stop_signal = Box::pin(async move {
                    let _ = stopped.await;
                });
                let ran = catch_unwind(AssertUnwindSafe(|| runtime.block_on(work(stop_signal))));
                if let Err(payload) = ran {
                    tracing::error!(
                        "{label} panicked: {}. Its scheduler serves no more requests; restart \
                         the server",
                        panic_message(payload.as_ref())
                    );
                }
            })
            .map_err(|e| anyhow!("starting the {name} thread: {e}"))?;
        Ok(Self {
            stop: Some(stop),
            thread: Some(thread),
        })
    }
}

impl Drop for EngineThread {
    fn drop(&mut self) {
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
        self.scheduler.submit(prompt_tokens, gen_config)
    }

    fn block_stats(&self) -> Option<BlockAllocatorStats> {
        self.scheduler.block_stats()
    }

    fn queue_depth(&self) -> usize {
        self.scheduler.queue_depth()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::*;
    use crate::engine::requests::batch_test_rig::{greedy, rig, Rig, POOL};
    use crate::engine::test_support::{ticker, wait_past, WAIT_LIMIT};

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
        let thread = EngineThread::spawn("blazr-batch", move |stop| engine.run_until(stop))?;
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
        let exited = Arc::new(AtomicUsize::new(0));
        let flag = Arc::clone(&exited);
        let thread = EngineThread::spawn("blazr-test-engine", move |stop| async move {
            stop.await;
            flag.fetch_add(1, Ordering::SeqCst);
        })?;
        drop(thread);
        assert_eq!(exited.load(Ordering::SeqCst), 1);
        Ok(())
    }
}
