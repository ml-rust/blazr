//! The swarm pipeline as the server's generation scheduler.
//!
//! [`SwarmScheduler`] takes a tokenized request from any HTTP route and
//! runs it through the pipeline stages, one task per request. The leader
//! samples with the single-process sampler, so the grammar, logprobs,
//! `logit_bias`, DRY, mirostat and the seed apply as on `blazr serve`. The
//! text, the stop sequences and the finish reasons apply as the batching
//! scheduler applies them.
//!
//! A stage error fails its request only. A lost stage rank, or any other
//! pipeline error, fails every request in flight and marks the swarm down:
//! `GET /health` then answers 503, and every new request is refused.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use boostr::inference::scheduler::SequenceId;
use nexar::Rank;
use tokio::sync::{oneshot, watch, Semaphore};
use tokio::task::{AbortHandle, JoinSet};

use super::swarm_request::{RequestJob, SwarmPipeline};
use super::swarm_stage::LocalStage;
use crate::config::GenerationConfig;
use crate::distributed::leader::SwarmLeader;
use crate::engine::{
    token_budget, CancelRequest, Executor, GenerationScheduler, RequestHandle, RequestRegistry,
    RowSampler,
};
use crate::runtime::ServerRuntime;

/// The running requests and their cancel senders.
#[derive(Default)]
struct Tasks {
    /// Dropping a request's sender cancels it.
    cancels: HashMap<SequenceId, oneshot::Sender<()>>,
    set: JoinSet<()>,
    /// Set by [`SwarmScheduler::close`]. A closed scheduler takes no request.
    closed: bool,
}

/// A [`GenerationScheduler`] over a [`SwarmLeader`] and the leader's stage.
pub struct SwarmScheduler {
    pipeline: Arc<SwarmPipeline>,
    /// Positions one request can reach: prompt plus output.
    context_len: usize,
    tasks: Mutex<Tasks>,
    /// The task that marks the swarm down when nexar reports a peer dead.
    watcher: AbortHandle,
    me: Weak<SwarmScheduler>,
}

/// The context a swarm serves with `executor` on the leader: the smaller of
/// the configured context and the model's own. The leader's executor holds
/// no weights, so both come from its config.
pub(crate) fn swarm_context(executor: &Executor<ServerRuntime>) -> usize {
    let config = executor.config();
    config.max_seq_len().min(config.model.max_seq_len)
}

impl SwarmScheduler {
    /// A scheduler that runs every request through `leader`, with the
    /// leader's own stage on `local`, and at most `max_requests` at once.
    ///
    /// `executor` holds the config and tokenizer of the model the leader
    /// serves, and needs no weights: the stages hold them. Its sampler
    /// samples, and its `inference.prefill_chunk_size` splits each prompt.
    ///
    /// # Errors
    ///
    /// `max_requests` of 0, or no tokio runtime.
    pub(crate) fn new(
        leader: Arc<SwarmLeader>,
        local: Option<LocalStage>,
        executor: Arc<Executor<ServerRuntime>>,
        max_requests: usize,
    ) -> Result<Arc<Self>> {
        if max_requests == 0 {
            bail!("a swarm needs --max-requests of 1 or more");
        }
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow!("the swarm scheduler must start inside a tokio runtime"))?;
        let context_len = swarm_context(&executor);
        let deaths = leader.client().failure_watch();
        let pipeline = Arc::new(SwarmPipeline {
            prefill_chunk: executor.config().inference.prefill_chunk_size,
            leader,
            local,
            executor,
            registry: Arc::new(RequestRegistry::new()),
            slots: Arc::new(Semaphore::new(max_requests)),
            waiting: Default::default(),
            fault: OnceLock::new(),
        });
        let watcher = runtime
            .spawn(watch_peers(deaths, Arc::downgrade(&pipeline)))
            .abort_handle();
        Ok(Arc::new_cyclic(|me| Self {
            pipeline,
            context_len,
            tasks: Mutex::new(Tasks::default()),
            watcher,
            me: me.clone(),
        }))
    }

    /// Cancels every request, waits up to `within` for each to release its
    /// stages, and takes no request after. Call it after the HTTP server
    /// stops and before the leader shuts down, so no task holds the leader.
    pub async fn close(&self, within: Duration) {
        let mut set = match self.tasks.lock() {
            Ok(mut tasks) => {
                tasks.closed = true;
                tasks.cancels.clear();
                std::mem::take(&mut tasks.set)
            }
            Err(e) => {
                tracing::error!("the swarm task map lock is poisoned: {e}");
                return;
            }
        };
        self.watcher.abort();
        let drained =
            tokio::time::timeout(within, async { while set.join_next().await.is_some() {} }).await;
        if drained.is_err() {
            tracing::warn!("swarm requests did not release in time; aborting them");
            set.abort_all();
            while set.join_next().await.is_some() {}
        }
    }

    /// Requests whose task still runs.
    #[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
    pub(super) fn running(&self) -> usize {
        self.tasks.lock().map_or(0, |t| t.cancels.len())
    }

    fn forget(&self, seq_id: SequenceId) {
        if let Ok(mut tasks) = self.tasks.lock() {
            tasks.cancels.remove(&seq_id);
        }
    }
}

impl Drop for SwarmScheduler {
    fn drop(&mut self) {
        self.watcher.abort();
    }
}

/// Marks the swarm down at the first peer nexar reports dead.
async fn watch_peers(mut deaths: watch::Receiver<Vec<Rank>>, pipeline: Weak<SwarmPipeline>) {
    loop {
        let dead = deaths.borrow_and_update().first().copied();
        if let Some(rank) = dead {
            if let Some(p) = pipeline.upgrade() {
                p.fail_swarm(format!("nexar reports swarm rank {rank} dead"));
            }
            return;
        }
        if deaths.changed().await.is_err() {
            return;
        }
    }
}

impl GenerationScheduler for SwarmScheduler {
    fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        gen_config: GenerationConfig,
    ) -> Result<RequestHandle> {
        if let Some(reason) = self.fault() {
            bail!("the swarm is down: {reason}. Restart every swarm node");
        }
        if prompt_tokens.is_empty() {
            bail!("the prompt has no tokens");
        }
        let (prompt_len, context) = (prompt_tokens.len(), self.context_len);
        let budget = token_budget(gen_config.max_tokens, context, prompt_len);
        if budget == 0 {
            bail!(
                "the request leaves no token to generate: max_tokens is {} and the prompt of \
                 {prompt_len} tokens leaves {} of the {context}-token context. Raise \
                 max_tokens or shorten the prompt",
                gen_config.max_tokens,
                context.saturating_sub(prompt_len)
            );
        }
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow!("the swarm scheduler must run inside a tokio runtime"))?;
        let p = &self.pipeline;
        let sampler = RowSampler::new(
            Arc::clone(&p.executor),
            &prompt_tokens,
            gen_config.clone(),
            budget,
        )?;
        let seq_id = p.registry.next_id();
        let canceller: Weak<dyn CancelRequest> = self.me.clone();
        let handle = p
            .registry
            .insert(seq_id, prompt_tokens.clone(), gen_config)?
            .cancel_on_drop(canceller);
        let (cancel, cancelled) = oneshot::channel();
        let job = RequestJob {
            seq_id,
            prompt: prompt_tokens,
            sampler,
            pipeline: Arc::clone(p),
        };
        let owner = self.me.clone();
        // The lock spans the spawn, so the task's own removal runs after
        // this insert.
        let mut tasks = self
            .tasks
            .lock()
            .map_err(|e| anyhow!("the swarm task map lock is poisoned: {e}; restart"))?;
        if tasks.closed {
            // The handle's drop cancels through this lock, so release it first.
            drop(tasks);
            bail!("the swarm scheduler is shutting down");
        }
        while tasks.set.try_join_next().is_some() {}
        tasks.set.spawn_on(
            async move {
                job.run(cancelled).await;
                if let Some(owner) = owner.upgrade() {
                    owner.forget(seq_id);
                }
            },
            &runtime,
        );
        tasks.cancels.insert(seq_id, cancel);
        Ok(handle)
    }

    fn queue_depth(&self) -> usize {
        self.pipeline.waiting.load(Ordering::Relaxed)
    }

    /// Why the swarm can no longer serve, or `None` while it can.
    fn fault(&self) -> Option<String> {
        self.pipeline.fault.get().cloned()
    }
}

impl CancelRequest for SwarmScheduler {
    fn cancel_request(&self, seq_id: SequenceId) {
        let _ = self.pipeline.registry.remove(seq_id);
        self.forget(seq_id);
    }
}

#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use std::sync::atomic::Ordering::SeqCst;

    use super::super::swarm_test_rig::{eventually, Probe, Swarm, LAYERS, MAX_REQUESTS, WAIT};
    use super::*;
    use crate::engine::test_support::full_vocab_front;
    use crate::engine::{FinishReason, GeneratedToken};

    async fn swarm(last: Probe) -> Result<Swarm> {
        let executor = Arc::new(full_vocab_front(LAYERS)?);
        Swarm::start(executor, vec![Probe::default(), Probe::default(), last]).await
    }

    fn slow() -> Probe {
        Probe {
            delay: Some(Duration::from_millis(10)),
            ..Probe::default()
        }
    }

    fn greedy(max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_tokens,
            ..GenerationConfig::greedy()
        }
    }

    async fn next(handle: &mut RequestHandle) -> Result<GeneratedToken> {
        tokio::time::timeout(WAIT, handle.token_rx.recv())
            .await?
            .ok_or_else(|| anyhow!("the stream closed"))?
    }

    async fn collect(handle: &mut RequestHandle) -> Result<Vec<GeneratedToken>> {
        let mut tokens = Vec::new();
        loop {
            let token = next(handle).await?;
            let done = token.finish_reason.is_some();
            tokens.push(token);
            if done {
                return Ok(tokens);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_the_handle_frees_the_request_on_every_stage() -> Result<()> {
        let s = swarm(slow()).await?;
        let mut handle = s.scheduler.submit(vec![1, 2, 3], greedy(50))?;
        for _ in 0..3 {
            next(&mut handle).await?;
        }
        assert!(s.probes.iter().all(|p| p.open() == 1));
        drop(handle);
        eventually("every stage to free the request", || s.idle()).await?;
        let frames = s.probes[2].frames.load(SeqCst);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(s.probes[2].frames.load(SeqCst), frames, "decoding went on");
        assert!(frames < 50, "{frames} frames ran");

        let mut again = s.scheduler.submit(vec![5], greedy(3))?;
        let tokens = collect(&mut again).await?;
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[2].finish_reason, Some(FinishReason::Length));
        eventually("the finished request to free", || s.idle()).await?;
        s.shutdown().await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn requests_past_the_cap_wait_for_a_slot() -> Result<()> {
        let s = swarm(slow()).await?;
        let mut handles = (0..MAX_REQUESTS + 2)
            .map(|i| s.scheduler.submit(vec![1 + i as u32, 2], greedy(4)))
            .collect::<Result<Vec<_>>>()?;
        eventually("two requests to wait", || s.scheduler.queue_depth() == 2).await?;
        for handle in &mut handles {
            assert_eq!(collect(handle).await?.len(), 4);
        }
        assert_eq!(s.scheduler.queue_depth(), 0);
        eventually("every request to free", || s.idle()).await?;
        s.shutdown().await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_request_with_no_room_is_refused_before_it_runs() -> Result<()> {
        let s = swarm(Probe::default()).await?;
        let full = vec![1; swarm_context(&full_vocab_front(LAYERS)?)];
        let err = s
            .scheduler
            .submit(full, greedy(4))
            .err()
            .map(|e| e.to_string());
        assert!(err.is_some_and(|e| e.contains("leaves no token")));
        assert!(s.scheduler.submit(Vec::new(), greedy(4)).is_err());
        assert!(s.probes.iter().all(|p| p.frames.load(SeqCst) == 0));
        s.shutdown().await
    }

    /// The order `blazr swarm` stops in: close the scheduler, then shut the
    /// leader down, and every worker loop ends clean.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn close_ends_running_requests_and_the_swarm_shuts_down_clean() -> Result<()> {
        let s = swarm(slow()).await?;
        let mut handle = s.scheduler.submit(vec![1, 2], greedy(50))?;
        next(&mut handle).await?;
        s.scheduler.close(WAIT).await;
        let err = loop {
            match next(&mut handle).await {
                Ok(_) => continue,
                Err(e) => break e.to_string(),
            }
        };
        assert!(err.contains("shut down"), "{err}");
        let refused = s.scheduler.submit(vec![1], greedy(2)).err();
        assert!(refused.is_some_and(|e| e.to_string().contains("shutting down")));
        eventually("every stage to free the request", || s.idle()).await?;
        s.shutdown().await
    }
}
