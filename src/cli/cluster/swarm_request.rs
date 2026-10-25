//! One swarm request: its frames through the pipeline, and its tokens.
//!
//! The leader runs stage 0 on its [`LocalStage`] and sends the activation
//! into the worker stages. The last worker returns one logits row per
//! frame. The leader samples each row with the single-process sampler, and
//! feeds the token back as the next decode frame. The registry renders the
//! text and applies the stop sequences.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, Result};
use boostr::inference::scheduler::SequenceId;
use tokio::sync::{oneshot, Semaphore};

use super::swarm_stage::LocalStage;
use crate::distributed::leader::SwarmLeader;
use crate::distributed::stage::StageOutput;
use crate::distributed::transport::{GenRequestHeader, StageError, StageOp, StagePayload};
use crate::engine::{Executor, GeneratedToken, RequestRegistry, RowSampler};
use crate::runtime::ServerRuntime;

/// How long a request waits for its release to leave the leader.
const RELEASE_WAIT: Duration = Duration::from_secs(5);

/// What every request of one swarm shares.
pub(super) struct SwarmPipeline {
    pub leader: Arc<SwarmLeader>,
    /// The leader's own stage. `None` when every stage runs on a worker.
    pub local: Option<LocalStage>,
    /// The leader's executor: its tokenizer and sampler.
    pub executor: Arc<Executor<ServerRuntime>>,
    pub registry: Arc<RequestRegistry>,
    /// One permit per request the stages hold at once.
    pub slots: Arc<Semaphore>,
    /// Requests that wait for a permit.
    pub waiting: AtomicUsize,
    /// Why the swarm can no longer serve. Set once, never cleared.
    pub fault: OnceLock<String>,
    /// Prompt tokens per prefill frame. 0 sends the whole prompt in one frame.
    pub prefill_chunk: usize,
}

impl SwarmPipeline {
    /// Marks the swarm down for `reason`. The first reason stays.
    pub(super) fn fail_swarm(&self, reason: String) {
        if self.fault.set(reason.clone()).is_ok() {
            tracing::error!("the swarm is down: {reason}");
        }
    }

    /// Runs one frame of `tokens` through every stage and returns the
    /// logits row of its last position.
    ///
    /// # Errors
    ///
    /// The leader's stage error, a worker stage error, or a pipeline error.
    /// A pipeline error that is not a stage error marks the swarm down.
    async fn forward(&self, header: GenRequestHeader, tokens: Vec<i64>) -> Result<Vec<f32>> {
        let payload = StagePayload::Tokens(tokens);
        let payload = match &self.local {
            Some(local) => match local
                .forward(header, payload)
                .await
                .map_err(|e| e.context("the leader's pipeline stage failed the request"))?
            {
                StageOutput::Activation(activation) => StagePayload::Activation(activation),
                StageOutput::Logits(row) => return Ok(row),
            },
            None => payload,
        };
        self.leader
            .pipeline_forward(header, &payload)
            .await
            .map_err(|e| self.pipeline_error(e))
    }

    fn pipeline_error(&self, error: anyhow::Error) -> anyhow::Error {
        if let Some(stage) = error.downcast_ref::<StageError>() {
            return anyhow!(
                "swarm stage on rank {} failed the request: {}",
                stage.rank,
                stage.message
            );
        }
        let reason = format!("{error:#}");
        self.fail_swarm(reason.clone());
        anyhow!("the swarm pipeline failed: {reason}. Restart every swarm node")
    }

    /// Frees `request_id` on the leader's stage and on every worker stage.
    ///
    /// The worker release is skipped once the swarm is down, since a lost
    /// stage takes no frame.
    async fn release(&self, request_id: u64) {
        if let Some(local) = &self.local {
            local.release(request_id);
        }
        if self.fault.get().is_some() {
            return;
        }
        match tokio::time::timeout(RELEASE_WAIT, self.leader.release(request_id)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!(request_id, "the swarm release was not sent: {e:#}"),
            Err(_) => tracing::warn!(request_id, "the swarm release did not leave in time"),
        }
    }
}

fn header(
    request_id: u64,
    op: StageOp,
    position: usize,
    seq_len: usize,
) -> Result<GenRequestHeader> {
    Ok(GenRequestHeader {
        request_id,
        op,
        position: u32::try_from(position)?,
        seq_len: u32::try_from(seq_len)?,
    })
}

/// One request's task.
pub(super) struct RequestJob {
    pub seq_id: SequenceId,
    pub prompt: Vec<u32>,
    pub sampler: RowSampler<ServerRuntime>,
    pub pipeline: Arc<SwarmPipeline>,
}

impl RequestJob {
    /// Runs the request until it finishes, fails, or `cancel` resolves.
    ///
    /// The request holds a slot from its first frame until its release
    /// left, so no stage holds more requests than its cap. Every exit
    /// releases the request on every stage. A cancelled request sends its
    /// caller nothing more.
    pub(super) async fn run(self, mut cancel: oneshot::Receiver<()>) {
        let Self {
            seq_id,
            prompt,
            sampler,
            pipeline: p,
        } = self;
        p.waiting.fetch_add(1, Ordering::Relaxed);
        let permit = tokio::select! {
            biased;
            _ = &mut cancel => None,
            permit = Arc::clone(&p.slots).acquire_owned() => Some(permit),
        };
        p.waiting.fetch_sub(1, Ordering::Relaxed);
        let permit = match permit {
            None => return end_cancelled(&p, seq_id).await,
            Some(Ok(permit)) => permit,
            Some(Err(_)) => return end_cancelled(&p, seq_id).await,
        };
        let request_id = p.leader.next_request_id();
        let outcome = tokio::select! {
            biased;
            _ = &mut cancel => None,
            r = generate(&p, seq_id, &prompt, sampler, request_id) => Some(r),
        };
        p.release(request_id).await;
        drop(permit);
        match outcome {
            None => return end_cancelled(&p, seq_id).await,
            Some(Err(e)) => p.registry.send_error(seq_id, e).await,
            Some(Ok(())) => {}
        }
        let _ = p.registry.finish(seq_id);
    }
}

/// Ends a cancelled request. A dropped handle already removed it from the
/// registry. A request the scheduler's shutdown cancelled is still there,
/// and its caller gets an error.
async fn end_cancelled(p: &SwarmPipeline, seq_id: SequenceId) {
    if p.registry.contains(seq_id) {
        let closed = anyhow!("the swarm scheduler shut down before the request finished");
        p.registry.send_error(seq_id, closed).await;
        let _ = p.registry.finish(seq_id);
    }
}

/// Prefills `prompt` as request `request_id`, then samples and decodes one
/// token at a time until a token finishes the request or the caller stops
/// listening.
///
/// The prompt goes as one prefill frame, or with a prefill chunk as one
/// prefill frame and decode frames for the rest. A request has one frame in
/// the pipeline at a time.
async fn generate(
    p: &SwarmPipeline,
    seq_id: SequenceId,
    prompt: &[u32],
    mut sampler: RowSampler<ServerRuntime>,
    request_id: u64,
) -> Result<()> {
    let chunk = match p.prefill_chunk {
        0 => prompt.len(),
        n => n,
    };
    let mut position = 0;
    let mut row = Vec::new();
    for (i, piece) in prompt.chunks(chunk.max(1)).enumerate() {
        let op = if i == 0 {
            StageOp::Prefill
        } else {
            StageOp::Decode
        };
        let ids = piece.iter().map(|&t| i64::from(t)).collect();
        row = p
            .forward(header(request_id, op, position, piece.len())?, ids)
            .await?;
        position += piece.len();
    }
    loop {
        let (back, sampled) = tokio::task::spawn_blocking(move || {
            let sampled = sampler.sample(&row);
            (sampler, sampled)
        })
        .await
        .map_err(|e| anyhow!("the swarm sampler stopped: {e}"))?;
        sampler = back;
        let sampled = sampled?;
        let tokenizer = p.executor.tokenizer();
        let cut = p
            .registry
            .token_text(seq_id, tokenizer, sampled.token_id, sampled.finish);
        let token = GeneratedToken {
            token_id: sampled.token_id,
            text: cut.text,
            logprob: sampled.logprob,
            top_logprobs: sampled.top_logprobs,
            finish_reason: cut.finish,
        };
        if !p.registry.send_token(seq_id, token).await || cut.finish.is_some() {
            return Ok(());
        }
        p.registry.push_history(seq_id, sampled.token_id);
        let h = header(request_id, StageOp::Decode, position, 1)?;
        row = p.forward(h, vec![i64::from(sampled.token_id)]).await?;
        position += 1;
    }
}

#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use boostr::CpuDevice;

    use super::super::disaggregated_test_rig::app;
    use super::super::swarm_serving::swarm_serving;
    use super::super::swarm_test_rig::{eventually, get, Probe, Swarm, LAYERS, WAIT};
    use super::*;
    use crate::config::GenerationConfig;
    use crate::engine::test_support::full_vocab_front;
    use crate::engine::{GenerationScheduler, RequestHandle, Scheduler};

    fn greedy(max_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_tokens,
            ..GenerationConfig::greedy()
        }
    }

    /// The ids `handle` streams before its error, and the error.
    async fn until_error(handle: &mut RequestHandle) -> Result<(Vec<u32>, String)> {
        let mut seen = Vec::new();
        loop {
            match tokio::time::timeout(WAIT, handle.token_rx.recv()).await? {
                Some(Ok(token)) if token.finish_reason.is_none() => seen.push(token.token_id),
                Some(Ok(_)) => return Err(anyhow!("the request finished without an error")),
                Some(Err(e)) => return Ok((seen, format!("{e:#}"))),
                None => return Err(anyhow!("the stream closed without the error")),
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stage_error_fails_only_its_request() -> Result<()> {
        let executor = Arc::new(full_vocab_front(LAYERS)?);
        let failing = Probe {
            fail_at: Some(6),
            ..Probe::default()
        };
        let probes = vec![Probe::default(), Probe::default(), failing];
        let s = Swarm::start(executor, probes).await?;
        let mut handle = s.scheduler.submit(vec![1, 2, 3], greedy(20))?;
        let (seen, err) = until_error(&mut handle).await?;
        assert_eq!(
            seen.len(),
            4,
            "the prefill token, then decodes at 3, 4 and 5"
        );
        assert!(err.contains("swarm stage on rank 2"), "{err}");
        assert!(err.contains("probe failure at position 6"), "{err}");
        assert_eq!(
            s.scheduler.fault(),
            None,
            "a stage error leaves the swarm up"
        );
        eventually("every stage to free the request", || s.idle()).await?;

        let mut next = s.scheduler.submit(vec![7], greedy(2))?;
        let mut ids = Vec::new();
        while let Some(item) = tokio::time::timeout(WAIT, next.token_rx.recv()).await? {
            ids.push(item?.token_id);
        }
        assert_eq!(ids.len(), 2);
        s.shutdown().await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_lost_stage_fails_the_request_and_marks_the_swarm_down() -> Result<()> {
        let executor = Arc::new(full_vocab_front(LAYERS)?);
        let slow = Probe {
            delay: Some(std::time::Duration::from_millis(20)),
            ..Probe::default()
        };
        let probes = vec![Probe::default(), Probe::default(), slow];
        let s = Swarm::start(Arc::clone(&executor), probes).await?;
        let mut handle = s.scheduler.submit(vec![1, 2, 3], greedy(50))?;
        tokio::time::timeout(WAIT, handle.token_rx.recv()).await?;
        s.kill(2);
        let (_, err) = until_error(&mut handle).await?;
        assert!(err.contains("swarm"), "{err}");
        eventually("the swarm to go down", || s.scheduler.fault().is_some()).await?;
        let refused = s.scheduler.submit(vec![1], greedy(2)).err();
        assert!(refused.is_some_and(|e| e.to_string().contains("the swarm is down")));

        let dir = tempfile::tempdir()?;
        let models = Scheduler::<ServerRuntime>::new(dir.path().to_path_buf(), CpuDevice::new());
        let serving = swarm_serving(executor, Arc::clone(&s.scheduler));
        let routed = app(Arc::new(models.pinned()), serving);
        let (status, body) = get(&routed, "/health").await?;
        assert_eq!(status, 503, "{body}");
        assert!(body.contains("\"unavailable\""), "{body}");
        Ok(())
    }
    /// Chunks of two over a twelve-token prompt: six prefill frames. The
    /// handle drops once the first frame reached the last stage, so the
    /// request leaves at its next frame and every stage frees it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancel_mid_prefill_stops_at_the_next_chunk() -> Result<()> {
        use std::sync::atomic::Ordering::SeqCst;

        let mut front = full_vocab_front(LAYERS)?;
        front.config.inference.prefill_chunk_size = 2;
        let slow = Probe {
            delay: Some(std::time::Duration::from_millis(50)),
            ..Probe::default()
        };
        let s = Swarm::start(
            Arc::new(front),
            vec![Probe::default(), Probe::default(), slow],
        )
        .await?;
        let handle = s.scheduler.submit((1..=12).collect(), greedy(4))?;
        eventually("the first frame to reach the last stage", || {
            s.probes[2].frames.load(SeqCst) >= 1
        })
        .await?;
        drop(handle);
        eventually("every stage to free the request", || s.idle()).await?;
        let frames = s.probes[2].frames.load(SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert_eq!(
            s.probes[2].frames.load(SeqCst),
            frames,
            "the prefill went on"
        );
        assert!(frames < 6, "{frames} of 6 prefill frames ran");
        s.shutdown().await
    }
}
