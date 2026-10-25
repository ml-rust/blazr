//! The pipeline stages of a swarm: each worker's stage, and the leader's
//! own stage on a thread of its own.
//!
//! - `stage_config`: the limits and KV dtype every stage of a node takes
//! - `build_stage`: a [`StageRunner`] over a loaded model, for one layer range
//! - `leader_assignment`: the leader's stage, known before any worker joins
//! - `LocalStage`: the leader's stage, which serves one frame at a time
//!
//! Each node loads the whole checkpoint and runs only its layer range, so
//! a stage's memory tracks the full model.

use std::panic::AssertUnwindSafe;
use std::sync::mpsc;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use boostr::model::LoadedModel;
use boostr::DType;
use tokio::sync::oneshot;

use crate::distributed::pipeline::PipelineSchedule;
use crate::distributed::stage::{Stage, StageConfig, StageOutput, StageRunner};
use crate::distributed::transport::{GenRequestHeader, LayerAssignment, StagePayload};
use crate::runtime::ServerRuntime;

/// Positions a new request's KV cache allocates up front. It grows to the
/// stage's `max_seq_len`.
const INITIAL_CAPACITY: usize = 256;

/// The config of a stage that holds `max_requests` requests of up to
/// `max_seq_len` positions in `kv_dtype`.
pub(crate) fn stage_config(
    max_requests: usize,
    max_seq_len: usize,
    kv_dtype: DType,
) -> StageConfig {
    StageConfig {
        max_requests,
        initial_capacity: INITIAL_CAPACITY.min(max_seq_len).max(1),
        max_seq_len,
        kv_dtype,
    }
}

/// The stage that runs `assignment` of `model` on `device`.
///
/// # Errors
///
/// A model with SSM or Gated DeltaNet state (Mamba, hybrid, qwen35), a
/// tensor-parallel shard, an assignment outside the model, or a config the
/// model refuses. Each error names the reason.
pub(crate) fn build_stage(
    model: Arc<LoadedModel<ServerRuntime>>,
    device: <ServerRuntime as boostr::Runtime>::Device,
    assignment: LayerAssignment,
    config: StageConfig,
) -> Result<Box<dyn Stage>> {
    Ok(Box::new(StageRunner::new(
        model, device, assignment, config,
    )?))
}

/// The leader's stage in a swarm of `world_size` ranks over `total_layers`
/// layers: stage 0 of [`PipelineSchedule::by_rank`].
///
/// # Errors
///
/// More ranks than layers.
pub(crate) fn leader_assignment(world_size: u32, total_layers: usize) -> Result<LayerAssignment> {
    let schedule = PipelineSchedule::by_rank(world_size, total_layers)?;
    let stage = schedule
        .stages()
        .first()
        .ok_or_else(|| anyhow!("a swarm of {world_size} ranks has no stage"))?;
    Ok(LayerAssignment {
        start_layer: u32::try_from(stage.start_layer)?,
        end_layer: u32::try_from(stage.end_layer)?,
        has_embedding: stage.has_embedding,
        has_lm_head: stage.has_lm_head,
    })
}

enum Job {
    Forward {
        header: GenRequestHeader,
        payload: StagePayload,
        reply: oneshot::Sender<Result<StageOutput>>,
    },
    Release(u64),
}

/// The leader's stage, served on its own thread.
///
/// Frames and releases run in the order they arrive. A release sent after
/// a forward of the same request therefore runs after that forward, even
/// when the caller stopped waiting for it. A forward that fails or panics
/// frees its request. The thread ends when this handle drops, and it frees
/// every request.
pub(crate) struct LocalStage {
    jobs: mpsc::Sender<Job>,
}

impl LocalStage {
    /// Starts the thread that serves `stage`.
    ///
    /// # Errors
    ///
    /// The thread does not start.
    pub(crate) fn spawn(stage: Box<dyn Stage>) -> Result<Self> {
        let (jobs, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("swarm-leader-stage".into())
            .spawn(move || serve(stage, rx))
            .map_err(|e| anyhow!("the leader's stage thread did not start: {e}"))?;
        Ok(Self { jobs })
    }

    /// Runs one prefill or decode frame on the leader's stage.
    ///
    /// # Errors
    ///
    /// The stage's error, or a stage thread that stopped.
    pub(crate) async fn forward(
        &self,
        header: GenRequestHeader,
        payload: StagePayload,
    ) -> Result<StageOutput> {
        let (reply, rx) = oneshot::channel();
        let job = Job::Forward {
            header,
            payload,
            reply,
        };
        self.jobs
            .send(job)
            .map_err(|_| anyhow!("the leader's stage thread stopped"))?;
        rx.await
            .map_err(|_| anyhow!("the leader's stage thread stopped during the frame"))?
    }

    /// Frees `request_id` on the leader's stage, after every frame of it
    /// already sent. It never blocks.
    pub(crate) fn release(&self, request_id: u64) {
        let _ = self.jobs.send(Job::Release(request_id));
    }
}

fn serve(mut stage: Box<dyn Stage>, jobs: mpsc::Receiver<Job>) {
    for job in jobs {
        match job {
            Job::Forward {
                header,
                payload,
                reply,
            } => {
                let run =
                    std::panic::catch_unwind(AssertUnwindSafe(|| stage.forward(&header, payload)));
                let out = run.unwrap_or_else(|panic| {
                    stage.release(header.request_id);
                    Err(anyhow!(
                        "the leader's stage panicked on request {}: {}",
                        header.request_id,
                        panic_text(panic.as_ref())
                    ))
                });
                let _ = reply.send(out);
            }
            Job::Release(id) => stage.release(id),
        }
    }
    stage.release_all();
}

fn panic_text(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".into())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Mutex;

    use anyhow::bail;

    use super::*;
    use crate::distributed::transport::StageOp;

    /// A stage that echoes token ids as logits and records the open ids.
    struct Echo {
        open: Arc<Mutex<HashSet<u64>>>,
    }

    impl Stage for Echo {
        fn forward(&mut self, h: &GenRequestHeader, payload: StagePayload) -> Result<StageOutput> {
            let StagePayload::Tokens(ids) = payload else {
                bail!("not tokens");
            };
            if ids.first() == Some(&-1) {
                panic!("asked to panic");
            }
            self.open.lock().unwrap().insert(h.request_id);
            Ok(StageOutput::Logits(ids.iter().map(|&i| i as f32).collect()))
        }

        fn release(&mut self, id: u64) {
            self.open.lock().unwrap().remove(&id);
        }

        fn release_all(&mut self) {
            self.open.lock().unwrap().clear();
        }
    }

    fn header(id: u64) -> GenRequestHeader {
        GenRequestHeader {
            request_id: id,
            op: StageOp::Prefill,
            position: 0,
            seq_len: 1,
        }
    }

    #[tokio::test]
    async fn frames_and_releases_run_in_order_and_a_panic_is_an_error() -> Result<()> {
        let open = Arc::new(Mutex::new(HashSet::new()));
        let stage = LocalStage::spawn(Box::new(Echo {
            open: Arc::clone(&open),
        }))?;
        let out = stage
            .forward(header(1), StagePayload::Tokens(vec![4]))
            .await?;
        assert_eq!(out, StageOutput::Logits(vec![4.0]));
        stage.release(1);
        let err = stage
            .forward(header(2), StagePayload::Tokens(vec![-1]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("asked to panic"), "{err}");
        stage
            .forward(header(3), StagePayload::Tokens(vec![5]))
            .await?;
        assert_eq!(*open.lock().unwrap(), HashSet::from([3]));
        drop(stage);
        for _ in 0..500 {
            if open.lock().unwrap().is_empty() {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        bail!("the dropped stage kept request 3")
    }

    #[test]
    fn the_leader_runs_stage_zero_and_a_short_model_is_refused() -> Result<()> {
        let a = leader_assignment(3, 6)?;
        assert_eq!((a.start_layer, a.end_layer), (0, 2));
        assert!(a.has_embedding && !a.has_lm_head);
        assert!(leader_assignment(4, 3).is_err());
        let cfg = stage_config(4, 64, DType::F32);
        assert_eq!((cfg.initial_capacity, cfg.max_requests), (64, 4));
        Ok(())
    }

    /// A one-layer Mamba2 with zero weights.
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    fn tiny_mamba2() -> Result<LoadedModel<ServerRuntime>> {
        use boostr::model::mamba::Mamba2Config;
        use boostr::nn::{VarBuilder, VarMap};
        use boostr::{CpuDevice, CpuRuntime, Tensor};

        let yaml = "model_type: mamba2\nvocab_size: 12\nhidden_size: 8\nnum_layers: 1\n\
                    max_seq_len: 16\ntie_word_embeddings: true\nssm:\n  variant: mamba2\n  \
                    num_heads: 2\n  head_dim: 8\n  state_size: 4\n  chunk_size: 4\n";
        let config: boostr::model::UniversalConfig = serde_saphyr::from_str(yaml)?;
        let m = Mamba2Config::from_universal(&config)?;
        let hidden = config.hidden_size;
        let shapes: Vec<(&str, Vec<usize>)> = vec![
            ("model.embed_tokens.weight", vec![config.vocab_size, hidden]),
            ("model.layers.0.input_layernorm.weight", vec![hidden]),
            (
                "model.layers.0.mixer.in_proj.weight",
                vec![m.proj_dim(), hidden],
            ),
            (
                "model.layers.0.mixer.conv1d.weight",
                vec![m.conv_channels(), 1, m.d_conv],
            ),
            (
                "model.layers.0.mixer.out_proj.weight",
                vec![hidden, m.d_inner()],
            ),
            ("model.layers.0.mixer.A_log", vec![m.nheads]),
            ("model.layers.0.mixer.dt_bias", vec![m.nheads]),
            ("model.layers.0.mixer.D", vec![m.nheads]),
            ("model.norm.weight", vec![hidden]),
        ];
        let device = CpuDevice::new();
        let mut map = VarMap::<CpuRuntime>::new();
        for (name, shape) in shapes {
            let zeros = Tensor::<CpuRuntime>::zeros(&shape, DType::F32, &device)?;
            map.insert(name.to_string(), zeros);
        }
        let mut vb = VarBuilder::new(&mut map, &device);
        Ok(LoadedModel::load(&config, &mut vb)?)
    }

    /// The stage refuses the model: no blazr-side check duplicates it.
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    #[test]
    fn a_mamba_model_cannot_run_as_a_stage() -> Result<()> {
        let model = Arc::new(tiny_mamba2()?);
        let assignment = LayerAssignment {
            start_layer: 0,
            end_layer: 1,
            has_embedding: true,
            has_lm_head: true,
        };
        let device = boostr::CpuDevice::new();
        match build_stage(model, device, assignment, stage_config(2, 16, DType::F32)) {
            Ok(_) => bail!("a Mamba2 model built a pipeline stage"),
            Err(e) => assert!(e.to_string().contains("SSM state"), "{e}"),
        }
        Ok(())
    }
}
