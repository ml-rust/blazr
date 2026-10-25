//! The disaggregated router as the server's generation scheduler.
//!
//! [`DisaggScheduler`] takes a tokenized request from any HTTP route,
//! routes it to a prefill and a decode worker, and turns each streamed
//! token into the server's [`GeneratedToken`]. The text, the stop
//! sequences and the finish reasons apply here, as the batching scheduler
//! applies them, so every route, the streaming ones included, behaves as
//! on `blazr serve`.
//!
//! A worker error fails its request only. A lost worker, or any other
//! router error, marks the cluster down: `GET /health` then answers 503,
//! and every new request is refused. A worker counts as lost when nexar
//! reports its rank dead, or when the router ends a request because the
//! worker's connection closed. The router itself fails the requests that
//! the lost worker held.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::{anyhow, bail, Result};
use boostr::inference::scheduler::SequenceId;
use nexar::Rank;
use splintr::AnyTokenizer;
use tokio::sync::watch;
use tokio::task::AbortHandle;

use super::disaggregated_params::{disagg_refusal, finish_reason, generation_params};
use crate::config::GenerationConfig;
use crate::distributed::disaggregated::{DisaggRouter, GenerationParams, Stage, WorkerError};
use crate::engine::{
    CancelRequest, GeneratedToken, GenerationScheduler, RequestHandle, RequestRegistry,
};

/// A [`GenerationScheduler`] over a [`DisaggRouter`].
///
/// Each request runs on one task that reads its token stream. Dropping the
/// request's handle aborts that task, and dropping its stream tells both
/// workers to stop and free the request.
pub struct DisaggScheduler {
    router: Arc<DisaggRouter>,
    tokenizer: Arc<AnyTokenizer>,
    /// Ids that end a request with `Eos`.
    eos_tokens: Vec<u32>,
    /// The model's context in tokens. A request's token cap clamps to it.
    context_len: usize,
    registry: Arc<RequestRegistry>,
    /// The stream task of each running request.
    tasks: Mutex<HashMap<SequenceId, AbortHandle>>,
    /// The ranks nexar reports dead, from the router's client.
    deaths: watch::Receiver<Vec<Rank>>,
    /// Why the cluster can no longer serve. Set once, never cleared.
    fault: Arc<OnceLock<String>>,
    me: Weak<DisaggScheduler>,
}

impl DisaggScheduler {
    /// A scheduler that routes through `router` and renders text with
    /// `tokenizer`.
    ///
    /// `deaths` is the `failure_watch` of the router's nexar client. A rank
    /// it reports dead marks the cluster down.
    pub fn new(
        router: Arc<DisaggRouter>,
        tokenizer: Arc<AnyTokenizer>,
        eos_tokens: Vec<u32>,
        context_len: usize,
        deaths: watch::Receiver<Vec<Rank>>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            router,
            tokenizer,
            eos_tokens,
            context_len,
            registry: Arc::new(RequestRegistry::new()),
            tasks: Mutex::new(HashMap::new()),
            deaths,
            fault: Arc::new(OnceLock::new()),
            me: me.clone(),
        })
    }

    /// Requests whose stream task still runs.
    #[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
    fn running(&self) -> usize {
        self.tasks.lock().map_or(0, |t| t.len())
    }

    fn forget_task(&self, seq_id: SequenceId) -> Option<AbortHandle> {
        self.tasks.lock().ok()?.remove(&seq_id)
    }
}

/// Records `reason` as the cluster's fault, once, and logs it.
fn mark_down(fault: &OnceLock<String>, reason: String) {
    if fault.set(reason.clone()).is_ok() {
        tracing::error!("the disaggregated cluster is down: {reason}");
    }
}

/// The request error of a router error.
///
/// A worker error names its stage and rank and fails its request only.
/// Any other error is a lost worker or a broken router: it marks the
/// cluster down through `fault`.
fn stream_error(fault: &OnceLock<String>, error: anyhow::Error) -> anyhow::Error {
    match error.downcast_ref::<WorkerError>() {
        Some(w) => {
            let stage = match w.stage {
                Stage::Prefill => "prefill",
                Stage::Decode => "decode",
            };
            anyhow!(
                "disaggregated {stage} worker (rank {}) failed the request: {}",
                w.rank,
                w.message
            )
        }
        None => {
            let reason = format!("{error:#}");
            mark_down(fault, reason.clone());
            anyhow!("the disaggregated cluster failed: {reason}. Restart every node")
        }
    }
}

/// What one request's stream task reads from and writes to.
struct StreamJob {
    seq_id: SequenceId,
    prompt: Vec<u32>,
    params: GenerationParams,
    router: Arc<DisaggRouter>,
    tokenizer: Arc<AnyTokenizer>,
    registry: Arc<RequestRegistry>,
    fault: Arc<OnceLock<String>>,
    owner: Weak<DisaggScheduler>,
}

impl StreamJob {
    async fn run(self) {
        if let Err(e) = self.stream().await {
            self.registry.send_error(self.seq_id, e).await;
        }
        let _ = self.registry.finish(self.seq_id);
        if let Some(owner) = self.owner.upgrade() {
            owner.forget_task(self.seq_id);
        }
    }

    /// Forward every token until one finishes the request, the stream
    /// fails, or the caller stops listening. Returning drops the token
    /// stream, which cancels the request on both workers if it has not ended.
    async fn stream(&self) -> Result<()> {
        let mut tokens = self
            .router
            .route_request_stream(&self.prompt, self.params.clone(), None)
            .await
            .map_err(|e| stream_error(&self.fault, e))?;
        while let Some(item) = tokens.next().await {
            let event = item.map_err(|e| stream_error(&self.fault, e))?;
            let finish = event.finish.map(finish_reason);
            let cut = self
                .registry
                .token_text(self.seq_id, &self.tokenizer, event.token, finish);
            let token = GeneratedToken {
                token_id: event.token,
                text: cut.text,
                logprob: None,
                top_logprobs: None,
                finish_reason: cut.finish,
            };
            if !self.registry.send_token(self.seq_id, token).await || cut.finish.is_some() {
                return Ok(());
            }
            self.registry.push_history(self.seq_id, event.token);
        }
        Err(anyhow!(
            "disaggregated stream {} ended without a finish reason",
            tokens.request_id()
        ))
    }
}

impl GenerationScheduler for DisaggScheduler {
    fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        gen_config: GenerationConfig,
    ) -> Result<RequestHandle> {
        if let Some(reason) = self.fault() {
            bail!("the disaggregated cluster is down: {reason}. Restart every node");
        }
        if prompt_tokens.is_empty() {
            bail!("the prompt has no tokens: pass at least one token id");
        }
        let params = generation_params(
            &gen_config,
            prompt_tokens.len(),
            self.context_len,
            &self.eos_tokens,
        )?;
        // The router checks the params too. Checking them here keeps a bad
        // request a request error, never a cluster fault.
        params.check()?;
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow!("the disaggregated scheduler must run inside a tokio runtime"))?;
        let seq_id = self.registry.next_id();
        let canceller: Weak<dyn CancelRequest> = self.me.clone();
        let handle = self
            .registry
            .insert(seq_id, prompt_tokens.clone(), gen_config)?
            .cancel_on_drop(canceller);
        let job = StreamJob {
            seq_id,
            prompt: prompt_tokens,
            params,
            router: Arc::clone(&self.router),
            tokenizer: Arc::clone(&self.tokenizer),
            registry: Arc::clone(&self.registry),
            fault: Arc::clone(&self.fault),
            owner: self.me.clone(),
        };
        // The map lock spans the spawn, so the task's own removal runs after
        // this insert.
        let mut tasks = self
            .tasks
            .lock()
            .map_err(|e| anyhow!("disaggregated task map lock is poisoned: {e}; restart"))?;
        let task = runtime.spawn(job.run());
        tasks.insert(seq_id, task.abort_handle());
        Ok(handle)
    }

    fn refusal(&self, gen_config: &GenerationConfig) -> Option<String> {
        disagg_refusal(gen_config)
    }

    /// Why the cluster can no longer serve, or `None` while it can.
    fn fault(&self) -> Option<String> {
        if let Some(rank) = self.deaths.borrow().first().copied() {
            mark_down(
                &self.fault,
                format!("nexar reports worker rank {rank} dead"),
            );
        }
        self.fault.get().cloned()
    }
}

impl CancelRequest for DisaggScheduler {
    fn cancel_request(&self, seq_id: SequenceId) {
        if let Some(task) = self.forget_task(seq_id) {
            task.abort();
        }
        let _ = self.registry.remove(seq_id);
    }
}

#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use super::super::disaggregated_test_rig::{next_of, Cluster, Mock};
    use super::*;
    use crate::config::Constraint;
    use crate::engine::FinishReason;

    /// How long a test waits for one token before it calls the run stuck.
    const WAIT: Duration = Duration::from_secs(20);

    fn llama3() -> Result<Arc<AnyTokenizer>> {
        Ok(Arc::new(crate::tokenizer::from_pretrained("llama3")?))
    }

    async fn scheduler(mock: &Mock, eos: Vec<u32>) -> Result<(Cluster, Arc<DisaggScheduler>)> {
        let cluster = Cluster::mock(mock).await?;
        let s = DisaggScheduler::new(
            Arc::clone(&cluster.router),
            llama3()?,
            eos,
            4096,
            cluster.deaths.clone(),
        );
        Ok((cluster, s))
    }

    fn greedy(max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_tokens,
            repeat_penalty: 1.0,
            ..GenerationConfig::greedy()
        }
    }

    /// Every token of `handle` up to the finishing one, or its error.
    async fn collect(handle: &mut RequestHandle) -> Result<Vec<GeneratedToken>> {
        let mut tokens = Vec::new();
        loop {
            let item = tokio::time::timeout(WAIT, handle.token_rx.recv())
                .await?
                .ok_or_else(|| anyhow!("the stream closed before a finish reason"))?;
            let token = item?;
            let done = token.finish_reason.is_some();
            tokens.push(token);
            if done {
                return Ok(tokens);
            }
        }
    }

    fn ids(tokens: &[GeneratedToken]) -> Vec<u32> {
        tokens.iter().map(|t| t.token_id).collect()
    }

    fn text(tokens: &[GeneratedToken]) -> String {
        tokens.iter().map(|t| t.text.as_str()).collect()
    }

    /// Waits until `cond` holds, or fails after [`WAIT`].
    async fn eventually(what: &str, cond: impl Fn() -> bool) -> Result<()> {
        let start = std::time::Instant::now();
        while !cond() {
            if start.elapsed() > WAIT {
                return Err(anyhow!("timed out waiting for {what}"));
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        Ok(())
    }

    /// llama3 ids 0 to 15 are the one-byte pieces `!` to `0`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tokens_text_and_finish_reason_reach_the_handle() -> Result<()> {
        let mock = Mock::default();
        let (_cluster, s) = scheduler(&mock, Vec::new()).await?;
        let mut handle = s.submit(vec![1, 2, 3], greedy(4))?;
        let tokens = collect(&mut handle).await?;
        assert_eq!(ids(&tokens), vec![4, 5, 6, 7]);
        assert_eq!(text(&tokens), "%&'(");
        let finishes: Vec<_> = tokens.iter().map(|t| t.finish_reason).collect();
        assert_eq!(finishes, vec![None, None, None, Some(FinishReason::Length)]);
        assert!(tokens.iter().all(|t| t.logprob.is_none()));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prompt_ids_reach_prefill_unchanged() -> Result<()> {
        let mock = Mock::default();
        let (_cluster, s) = scheduler(&mock, Vec::new()).await?;
        let prompt = vec![128_000, 7, 0, 99_999, 3];
        let mut handle = s.submit(prompt.clone(), greedy(2))?;
        let tokens = collect(&mut handle).await?;
        assert_eq!(ids(&tokens), vec![next_of(3), next_of(next_of(3))]);
        assert_eq!(mock.prompts(), vec![prompt]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_seed_fixes_the_sampled_tokens() -> Result<()> {
        let mock = Mock {
            flat: true,
            ..Mock::default()
        };
        let (_cluster, s) = scheduler(&mock, Vec::new()).await?;
        let sampled = |seed| GenerationConfig {
            temperature: 1.0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: Some(seed),
            max_tokens: 12,
            ..GenerationConfig::default()
        };
        let run = async |seed| -> Result<Vec<u32>> {
            let mut handle = s.submit(vec![1], sampled(seed))?;
            Ok(ids(&collect(&mut handle).await?))
        };
        let a = run(7).await?;
        let b = run(7).await?;
        let c = run(8).await?;
        assert_eq!(a.len(), 12);
        assert_eq!(a, b);
        assert_ne!(a, c);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_eos_id_ends_the_request_with_eos() -> Result<()> {
        let mock = Mock::default();
        let (_cluster, s) = scheduler(&mock, vec![6]).await?;
        let mut handle = s.submit(vec![3], greedy(50))?;
        let tokens = collect(&mut handle).await?;
        assert_eq!(ids(&tokens), vec![4, 5, 6]);
        assert_eq!(tokens[2].finish_reason, Some(FinishReason::Eos));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stop_sequence_cuts_the_text_and_frees_the_workers() -> Result<()> {
        let mock = Mock::default();
        let (_cluster, s) = scheduler(&mock, Vec::new()).await?;
        let cfg = GenerationConfig {
            stop_sequences: vec!["&'".to_string()],
            ..greedy(10)
        };
        let mut handle = s.submit(vec![3], cfg)?;
        let tokens = collect(&mut handle).await?;
        assert_eq!(text(&tokens), "%");
        assert_eq!(
            tokens.last().and_then(|t| t.finish_reason),
            Some(FinishReason::Stop)
        );
        eventually("the request to end", || s.running() == 0).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn each_unsupported_feature_is_refused() -> Result<()> {
        let mock = Mock::default();
        let (_cluster, s) = scheduler(&mock, Vec::new()).await?;
        let cases: Vec<(GenerationConfig, &str)> = vec![
            (
                GenerationConfig {
                    constraint: Some(Constraint::JsonObject),
                    ..greedy(4)
                },
                "grammar",
            ),
            (
                GenerationConfig {
                    mirostat_mode: 2,
                    ..greedy(4)
                },
                "mirostat",
            ),
            (
                GenerationConfig {
                    dry_multiplier: 1.0,
                    ..greedy(4)
                },
                "dry_multiplier",
            ),
            (
                GenerationConfig {
                    typical_p: 0.9,
                    ..greedy(4)
                },
                "typical_p",
            ),
            (
                GenerationConfig {
                    temperature: 0.8,
                    dynatemp_range: 0.2,
                    ..greedy(4)
                },
                "dynatemp_range",
            ),
            (
                GenerationConfig {
                    logit_bias: HashMap::from([(1, 2.0)]),
                    ..greedy(4)
                },
                "logit_bias",
            ),
            (
                GenerationConfig {
                    logprobs: true,
                    ..greedy(4)
                },
                "logprobs",
            ),
        ];
        for (cfg, field) in cases {
            let refusal = s.refusal(&cfg).unwrap_or_default();
            assert!(refusal.contains(field), "{field}: {refusal:?}");
            let err = s.submit(vec![1], cfg).err().map(|e| e.to_string());
            assert!(err.is_some_and(|e| e.contains(field)), "{field}");
        }
        assert!(
            mock.prompts().is_empty(),
            "no refused request reached a worker"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_worker_error_frame_becomes_the_request_error() -> Result<()> {
        let mock = Mock {
            prefill_fail: Some("out of device memory"),
            ..Mock::default()
        };
        let (_cluster, s) = scheduler(&mock, Vec::new()).await?;
        let mut handle = s.submit(vec![1], greedy(4))?;
        let err = collect(&mut handle).await.unwrap_err().to_string();
        assert!(err.contains("prefill worker (rank 1)"), "{err}");
        assert!(err.contains("out of device memory"), "{err}");

        let mock = Mock {
            fail_at_step: Some(2),
            ..Mock::default()
        };
        let (_cluster, s) = scheduler(&mock, Vec::new()).await?;
        let mut handle = s.submit(vec![1], greedy(8))?;
        let mut seen = Vec::new();
        let err = loop {
            match tokio::time::timeout(WAIT, handle.token_rx.recv()).await? {
                Some(Ok(token)) => seen.push(token.token_id),
                Some(Err(e)) => break e.to_string(),
                None => return Err(anyhow!("the stream closed without the error")),
            }
        };
        assert_eq!(seen, vec![2, 3], "prefill's token, then one decode step");
        assert!(err.contains("decode worker (rank 2)"), "{err}");
        assert!(err.contains("mock step failure"), "{err}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_worker_error_leaves_the_cluster_up() -> Result<()> {
        let mock = Mock {
            prefill_fail: Some("out of device memory"),
            ..Mock::default()
        };
        let (_cluster, s) = scheduler(&mock, Vec::new()).await?;
        let mut handle = s.submit(vec![1], greedy(4))?;
        assert!(collect(&mut handle).await.is_err());
        assert_eq!(s.fault(), None);
        assert!(s.submit(vec![1], greedy(4)).is_ok());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_lost_decode_worker_marks_the_cluster_down() -> Result<()> {
        use super::super::disaggregated_test_rig::app;
        use super::super::swarm_test_rig::get;
        use crate::engine::Scheduler;
        use crate::runtime::ServerRuntime;
        use crate::server::handlers::{ServeReplica, Serving};

        let mock = Mock {
            delay: Some(Duration::from_millis(20)),
            ..Mock::default()
        };
        let (cluster, s) = scheduler(&mock, Vec::new()).await?;
        assert_eq!(s.fault(), None);
        let mut handle = s.submit(vec![1], greedy(1000))?;
        tokio::time::timeout(WAIT, handle.token_rx.recv())
            .await?
            .ok_or_else(|| anyhow!("closed"))??;
        cluster.kill_decode();
        let err = loop {
            match tokio::time::timeout(WAIT, handle.token_rx.recv()).await? {
                Some(Ok(_)) => continue,
                Some(Err(e)) => break e.to_string(),
                None => return Err(anyhow!("the stream closed without the error")),
            }
        };
        assert!(err.contains("rank 2"), "{err}");
        eventually("the cluster to go down", || s.fault().is_some()).await?;
        let refused = s.submit(vec![1], greedy(2)).err().map(|e| e.to_string());
        assert!(
            refused.is_some_and(|e| e.contains("the disaggregated cluster is down")),
            "a new request is refused"
        );

        let dir = tempfile::tempdir()?;
        let models =
            Scheduler::<ServerRuntime>::new(dir.path().to_path_buf(), boostr::CpuDevice::new());
        let executor = Arc::new(crate::engine::test_support::seeded_full_vocab_executor(1)?);
        let disagg: Arc<dyn GenerationScheduler> = s;
        let serving = Serving::Single(ServeReplica {
            executor,
            scheduler: Some(disagg),
            device: "disaggregated".to_string(),
        });
        let routed = app(Arc::new(models.pinned()), serving);
        let (status, body) = get(&routed, "/health").await?;
        assert_eq!(status, 503, "{body}");
        assert!(body.contains("\"unavailable\""), "{body}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dead_rank_from_nexar_marks_the_cluster_down() -> Result<()> {
        let mock = Mock::default();
        let cluster = Cluster::mock(&mock).await?;
        let (report, deaths) = watch::channel(Vec::new());
        let s = DisaggScheduler::new(
            Arc::clone(&cluster.router),
            llama3()?,
            Vec::new(),
            4096,
            deaths,
        );
        assert_eq!(s.fault(), None);
        report.send(vec![1])?;
        let fault = s.fault().unwrap_or_default();
        assert!(fault.contains("worker rank 1 dead"), "{fault}");
        // A fault stays set, whatever nexar reports after.
        report.send(Vec::new())?;
        assert_eq!(s.fault(), Some(fault));
        assert!(s.submit(vec![1], greedy(2)).is_err());
        assert!(mock.prompts().is_empty(), "no request reached a worker");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_the_handle_cancels_and_frees_the_decode_worker() -> Result<()> {
        let mock = Mock {
            delay: Some(Duration::from_millis(10)),
            ..Mock::default()
        };
        let (_cluster, s) = scheduler(&mock, Vec::new()).await?;
        let mut handle = s.submit(vec![1], greedy(1000))?;
        for _ in 0..3 {
            tokio::time::timeout(WAIT, handle.token_rx.recv())
                .await?
                .ok_or_else(|| anyhow!("closed"))??;
        }
        drop(handle);
        assert_eq!(s.running(), 0, "the drop aborts the stream task");
        // The decode worker stops between steps once the cancel arrives.
        let settled = || {
            let before = mock.steps();
            std::thread::sleep(Duration::from_millis(100));
            mock.steps() == before
        };
        let mut stopped = false;
        for _ in 0..50 {
            if tokio::task::block_in_place(settled) {
                stopped = true;
                break;
            }
        }
        assert!(stopped, "the decode worker kept stepping");
        assert!(mock.steps() < 100, "{} steps ran", mock.steps());

        // The worker is free: the next request runs to its end.
        let mut next = s.submit(vec![5], greedy(3))?;
        assert_eq!(ids(&collect(&mut next).await?), vec![6, 7, 8]);
        Ok(())
    }
}
