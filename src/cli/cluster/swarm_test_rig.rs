//! An in-process swarm for the swarm serving tests: the leader on rank 0
//! and two workers. Each stage is a [`StageRunner`] over the layers of one
//! seeded model it holds, loaded as `blazr swarm` loads them.
//!
//! Each stage sits behind a [`Probe`], which records the requests the stage
//! holds and can slow or fail its frames.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use boostr::{CpuDevice, DType};
use nexar::NexarClient;
use tokio::task::JoinHandle;

use super::swarm_scheduler::{swarm_context, SwarmScheduler};
use super::swarm_stage::{build_stage, stage_config, LocalStage};
use crate::distributed::leader::SwarmLeader;
use crate::distributed::stage::{Stage, StageOutput};
use crate::distributed::transport::{self, GenRequestHeader, LayerAssignment, StagePayload};
use crate::distributed::worker::SwarmWorker;
use crate::engine::test_support::seeded_full_vocab_stage;
use crate::engine::Executor;
use crate::runtime::ServerRuntime;

/// Seed of the rig's model weights.
pub(super) const SEED: u64 = 0x5a12;
/// Layers of the rig's model: one per rank.
pub(super) const LAYERS: usize = 3;
/// Ranks of the rig: the leader and two workers.
const WORLD: u32 = 3;
/// Requests each stage and the scheduler hold at once.
pub(super) const MAX_REQUESTS: usize = 4;
/// How long a test waits before it calls the run stuck.
pub(super) const WAIT: Duration = Duration::from_secs(20);

/// What one stage holds, and how it misbehaves.
#[derive(Default, Clone)]
pub(super) struct Probe {
    pub(super) open: Arc<Mutex<HashSet<u64>>>,
    /// Frames the stage ran.
    pub frames: Arc<AtomicUsize>,
    /// A frame that starts at this position fails.
    pub fail_at: Option<u32>,
    /// A sleep before each frame.
    pub delay: Option<Duration>,
}

impl Probe {
    /// Requests the stage holds a KV cache for.
    pub(super) fn open(&self) -> usize {
        self.open.lock().map_or(usize::MAX, |o| o.len())
    }

    fn wrap(&self, inner: Box<dyn Stage>) -> Box<dyn Stage> {
        Box::new(Probed {
            inner,
            probe: self.clone(),
        })
    }

    fn set(&self, id: u64, open: bool) {
        if let Ok(mut ids) = self.open.lock() {
            if open {
                ids.insert(id);
            } else {
                ids.remove(&id);
            }
        }
    }
}

struct Probed {
    inner: Box<dyn Stage>,
    probe: Probe,
}

impl Stage for Probed {
    fn forward(&mut self, header: &GenRequestHeader, payload: StagePayload) -> Result<StageOutput> {
        self.probe.frames.fetch_add(1, Ordering::SeqCst);
        if let Some(delay) = self.probe.delay {
            std::thread::sleep(delay);
        }
        let id = header.request_id;
        if self.probe.fail_at == Some(header.position) {
            self.inner.release(id);
            self.probe.set(id, false);
            bail!("probe failure at position {}", header.position);
        }
        let out = self.inner.forward(header, payload);
        self.probe.set(id, out.is_ok());
        out
    }

    fn release(&mut self, request_id: u64) {
        self.inner.release(request_id);
        self.probe.set(request_id, false);
    }

    fn release_all(&mut self) {
        self.inner.release_all();
        if let Ok(mut ids) = self.probe.open.lock() {
            ids.clear();
        }
    }
}

/// The running swarm.
pub(super) struct Swarm {
    pub leader: Arc<SwarmLeader>,
    pub scheduler: Arc<SwarmScheduler>,
    /// The probe of each rank's stage, by rank.
    pub probes: Vec<Probe>,
    workers: Vec<JoinHandle<Result<()>>>,
}

/// The assignment of stage `rank` of `leader`'s schedule.
fn assignment(leader: &SwarmLeader, rank: usize) -> Result<LayerAssignment> {
    let stage = leader
        .schedule()
        .stages()
        .get(rank)
        .ok_or_else(|| anyhow!("no stage {rank}"))?;
    Ok(LayerAssignment {
        start_layer: u32::try_from(stage.start_layer)?,
        end_layer: u32::try_from(stage.end_layer)?,
        has_embedding: stage.has_embedding,
        has_lm_head: stage.has_lm_head,
    })
}

impl Swarm {
    /// A swarm that serves `executor`'s model, with `probes[r]` on rank `r`.
    ///
    /// `executor` samples, as the leader's does, and can hold no weights.
    /// Each stage holds its own layers of the seeded model of [`SEED`] only.
    pub(super) async fn start(
        executor: Arc<Executor<ServerRuntime>>,
        probes: Vec<Probe>,
    ) -> Result<Self> {
        if probes.len() != WORLD as usize {
            bail!("the rig takes one probe per rank");
        }
        crate::tls::install_crypto_provider();
        let mut clients = NexarClient::bootstrap_local(WORLD, transport::cpu_adapter())
            .await
            .map_err(|e| anyhow!("local nexar bootstrap failed: {e}"))?;
        clients.sort_by_key(|c| c.rank());
        let mut clients = clients.into_iter().map(Arc::new);
        let leader_client = clients.next().ok_or_else(|| anyhow!("no rank 0"))?;
        let leader = Arc::new(SwarmLeader::by_rank(
            leader_client,
            LAYERS,
            executor.vocab_size(),
        )?);
        let cfg = stage_config(MAX_REQUESTS, swarm_context(&executor), DType::F32);
        let stage_of = |a: LayerAssignment| -> Result<Box<dyn Stage>> {
            let layers = a.start_layer as usize..a.end_layer as usize;
            let model = Arc::new(seeded_full_vocab_stage(SEED, LAYERS, layers)?);
            build_stage(model, CpuDevice::new(), a, cfg)
        };
        let mut workers = Vec::new();
        for (rank, client) in (1..WORLD).zip(clients) {
            let a = assignment(&leader, rank as usize)?;
            let stage = stage_of(a)?;
            let next = (rank + 1 < WORLD).then_some(rank + 1);
            let stage = probes[rank as usize].wrap(stage);
            let mut worker = SwarmWorker::new(client, rank, a, Some(rank - 1), next, 0, stage);
            workers.push(tokio::spawn(async move {
                let run = worker.run_compute_loop().await;
                worker.shutdown(Duration::from_secs(10)).await?;
                run
            }));
        }
        let own = leader
            .local_assignment()
            .ok_or_else(|| anyhow!("the leader runs no stage"))?;
        let stage = probes[0].wrap(stage_of(own)?);
        let local = LocalStage::spawn(stage)?;
        let scheduler =
            SwarmScheduler::new(Arc::clone(&leader), Some(local), executor, MAX_REQUESTS)?;
        Ok(Self {
            leader,
            scheduler,
            probes,
            workers,
        })
    }

    /// Kills worker `rank`: its task stops and its nexar client drops.
    pub(super) fn kill(&self, rank: usize) {
        if let Some(worker) = self.workers.get(rank - 1) {
            worker.abort();
        }
    }

    /// Whether no stage holds a request.
    pub(super) fn idle(&self) -> bool {
        self.probes.iter().all(|p| p.open() == 0) && self.scheduler.running() == 0
    }

    /// Shuts down as `blazr swarm` does: end every request, stop the
    /// workers, then shut down the leader's client. Each worker's loop must
    /// end without an error.
    pub(super) async fn shutdown(self) -> Result<()> {
        self.scheduler.close(WAIT).await;
        drop(self.scheduler);
        let leader = Arc::try_unwrap(self.leader)
            .map_err(|_| anyhow!("a task still holds the leader after close"))?;
        leader.shutdown(Duration::from_secs(10)).await?;
        for worker in self.workers {
            tokio::time::timeout(WAIT, worker).await???;
        }
        Ok(())
    }
}

/// Waits until `cond` holds, or fails after [`WAIT`].
pub(super) async fn eventually(what: &str, cond: impl Fn() -> bool) -> Result<()> {
    let start = std::time::Instant::now();
    while !cond() {
        if start.elapsed() > WAIT {
            bail!("timed out waiting for {what}");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    Ok(())
}

/// The status and body of a `GET` of `path`.
pub(super) async fn get(app: &axum::Router, path: &str) -> Result<(u16, String)> {
    use tower::ServiceExt;
    let request = axum::http::Request::get(path).body(axum::body::Body::empty())?;
    let response = app.clone().oneshot(request).await?;
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
    Ok((status, String::from_utf8(bytes.to_vec())?))
}
