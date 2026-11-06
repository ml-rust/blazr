//! Loads the start model of `blazr serve`: one copy, or one copy per
//! data-parallel replica.
//!
//! Replica `i` runs on GPU `device + i` on a CUDA or wgpu build, and on its
//! own CPU copy on a CPU build. Each copy gets its own batching scheduler.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use colored::Colorize;

use crate::config::{tp_conflict, DecisionConfig, DtypeRequest, InferenceOverlay, VisionOverrides};
use crate::engine::{DataParallelGroup, Executor, Scheduler, SloBatchPolicy};
use crate::server::handlers::{ServeReplica, ServerDevice, ServerRuntime, Serving, StartReplica};

/// The settings every model scheduler of one server shares.
#[derive(Debug, Clone)]
pub struct SchedulerSettings {
    /// `--model` or `default_model`, which can be a path.
    pub operator_model: Option<String>,
    pub model_dir: PathBuf,
    pub num_ctx: usize,
    pub chat_template: Option<String>,
    pub inference: InferenceOverlay,
    pub tensor_parallel_size: Option<usize>,
    pub vision: VisionOverrides,
    /// `--dtype`, then `inference.dtype`, for every model loaded.
    pub dtype: Option<DtypeRequest>,
    /// `--decision-*`, then `inference.decision`, for every decision model.
    pub decision: DecisionConfig,
}

impl SchedulerSettings {
    /// A model scheduler on `device`.
    pub fn build(&self, device: ServerDevice) -> Scheduler<ServerRuntime> {
        Scheduler::new(self.model_dir.clone(), device)
            .with_operator_model(self.operator_model.clone())
            .with_num_ctx(self.num_ctx)
            .with_chat_template(self.chat_template.clone())
            .with_inference_overlay(self.inference.clone())
            .with_tensor_parallel_size(self.tensor_parallel_size)
            .with_vision(self.vision.clone())
            .with_dtype(self.dtype)
            .with_decision(self.decision.clone())
    }
}

/// Load `model` for serving.
///
/// `primary` is the server's model scheduler on GPU `base_index`.
/// It holds replica 0. Above 1 replica, each further copy loads on the next
/// device.
///
/// # Errors
///
/// An error when a copy does not load, when the host has too few devices,
/// when the model runs tensor parallel, or when a batching engine fails.
pub async fn load_serving(
    primary: &Scheduler<ServerRuntime>,
    settings: &SchedulerSettings,
    model: &str,
    base_index: usize,
    replicas: usize,
    slo_policy: Option<Arc<SloBatchPolicy>>,
) -> Result<Serving> {
    let first = primary.get_executor(model).await?;
    if replicas <= 1 {
        let label = device_label(base_index, 0);
        let replica = serve_replica(first, label, slo_policy).await?;
        // The model scheduler owns the copy, so an eviction frees the device.
        let start = StartReplica::new(&replica);
        if let Some(batching) = replica.scheduler {
            primary.attach_batching(model, batching).await;
        }
        return Ok(Serving::Start(start));
    }

    let tp = first.config().inference.tensor_parallel_size;
    if tp > 1 {
        return Err(anyhow!(tp_conflict(replicas, tp)));
    }

    let devices = replica_devices(base_index, replicas)?;
    let mut loaded = Vec::with_capacity(replicas);
    loaded.push(serve_replica(first, device_label(base_index, 0), slo_policy.clone()).await?);
    for (i, device) in devices.into_iter().enumerate().skip(1) {
        let label = device_label(base_index, i);
        let spinner = crate::cli::util::spinner(format!("Loading replica {i} on {label}..."));
        let executor = settings.build(device).get_executor(model).await?;
        if let Err(e) = executor.warmup() {
            tracing::warn!("replica {i} warmup failed. Its first request is slower: {e}");
        }
        spinner.finish_and_clear();
        eprintln!("  {} replica {} on {}", "Loaded".green(), i, label.bold());
        loaded.push(serve_replica(executor, label, slo_policy.clone()).await?);
    }

    eprintln!(
        "  Data parallel: {} replicas, cache-aware routing",
        replicas.to_string().green()
    );
    // The group holds every replica for the server's life, so an eviction of
    // replica 0 frees nothing.
    primary.hold(model).await;
    let group = DataParallelGroup::new(model.to_string(), loaded)?;
    Ok(Serving::DataParallel(Arc::new(group)))
}

/// One copy with its batching scheduler running.
async fn serve_replica(
    executor: Arc<Executor<ServerRuntime>>,
    device: String,
    slo_policy: Option<Arc<SloBatchPolicy>>,
) -> Result<ServeReplica> {
    let scheduler =
        super::serve_batching::configure_batching(Arc::clone(&executor), slo_policy, &device)
            .await?;
    Ok(ServeReplica {
        executor,
        scheduler,
        device,
    })
}

/// The name of replica `i`'s device.
fn device_label(base_index: usize, i: usize) -> String {
    match crate::runtime::GpuBackend::of_build() {
        Some(gpu) => format!("{}:{}", gpu.name(), base_index + i),
        None => format!("cpu:{i}"),
    }
}

/// One device per replica: CUDA devices `base_index..base_index + replicas`.
#[cfg(feature = "cuda")]
fn replica_devices(base_index: usize, replicas: usize) -> Result<Vec<ServerDevice>> {
    let count = boostr::CudaDevice::count()
        .map_err(|e| anyhow!("counting CUDA devices for inference.data_parallel: {e}"))?;
    let last = base_index + replicas;
    if last > count {
        return Err(anyhow!(
            "inference.data_parallel is {replicas} from device cuda:{base_index}, so it needs \
             CUDA devices {base_index} to {}. This host has {count}. Lower \
             inference.data_parallel or the device index",
            last - 1
        ));
    }
    Ok((base_index..last).map(boostr::CudaDevice::new).collect())
}

/// One device per replica: wgpu adapters `base_index..base_index + replicas`.
///
/// An adapter that does not open names the replica count to lower.
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
fn replica_devices(base_index: usize, replicas: usize) -> Result<Vec<ServerDevice>> {
    (base_index..base_index + replicas)
        .map(|index| {
            crate::runtime::open_wgpu_device(index).map_err(|e| {
                anyhow!(
                    "inference.data_parallel is {replicas} from device wgpu:{base_index}, so it                      needs adapter {index}: {e}. Lower inference.data_parallel or the device                      index"
                )
            })
        })
        .collect()
}

/// One device per replica: a CPU device for each copy.
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
fn replica_devices(base_index: usize, replicas: usize) -> Result<Vec<ServerDevice>> {
    let _ = base_index;
    Ok((0..replicas).map(|_| boostr::CpuDevice::new()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_replica_gets_one_device_with_a_label() {
        if crate::runtime::GpuBackend::of_build().is_some() {
            // A GPU host with fewer than three devices refuses the count.
            return;
        }
        let devices = replica_devices(0, 3);
        assert_eq!(devices.map(|d| d.len()).ok(), Some(3));
        assert_eq!(device_label(0, 2), "cpu:2");
    }

    /// Settings that give a tiny GGUF a paged batching engine on any runtime.
    fn settings(model_dir: &std::path::Path, start: &str) -> SchedulerSettings {
        let section = crate::config::InferenceConfig {
            paged_attention: true,
            max_batch_size: 2,
            block_size: 4,
            kv_pool_blocks: 16,
            ..Default::default()
        };
        let keys = [
            "paged_attention",
            "max_batch_size",
            "block_size",
            "kv_pool_blocks",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        SchedulerSettings {
            operator_model: Some(start.to_string()),
            model_dir: model_dir.to_path_buf(),
            num_ctx: 32,
            chat_template: None,
            inference: InferenceOverlay::from_section(&section, &keys).expect("overlay"),
            tensor_parallel_size: None,
            vision: VisionOverrides::default(),
            dtype: None,
            decision: DecisionConfig::default(),
        }
    }

    /// A model directory that holds `start.gguf` and `other.gguf`, both tiny
    /// real checkpoints, and a Whisper directory.
    fn model_dir() -> tempfile::TempDir {
        let (gguf, _) = crate::engine::load::bytes::tests::tiny_gguf();
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join("start.gguf"), &gguf).expect("write");
        std::fs::write(dir.path().join("other.gguf"), &gguf).expect("write");
        let whisper = dir.path().join("whisper-tiny");
        std::fs::create_dir(&whisper).expect("mkdir");
        std::fs::write(whisper.join("config.json"), r#"{"model_type": "whisper"}"#).expect("write");
        std::fs::write(whisper.join("model.safetensors"), b"").expect("write");
        dir
    }

    async fn loaded(
        dir: &std::path::Path,
        start: &str,
    ) -> (Arc<Scheduler<ServerRuntime>>, Serving) {
        let settings = settings(dir, start);
        let device = crate::runtime::open_device(0).expect("device 0 opens");
        let primary = Arc::new(settings.build(device));
        let serving = load_serving(&primary, &settings, start, 0, 1, None)
            .await
            .expect("the start model loads");
        (primary, serving)
    }

    /// What `blazr serve --model <path>` does: the entry owns the batching
    /// scheduler, so an eviction stops it and frees the model.
    #[tokio::test]
    async fn load_serving_leaves_one_evictable_batching_copy() {
        let dir = model_dir();
        let spelled = dir.path().join("start.gguf").display().to_string();
        let (primary, serving) = loaded(dir.path(), &spelled).await;
        assert!(matches!(serving, Serving::Start(_)));
        assert_eq!(serving.schedulers().len(), 1, "the start model batches");

        let executor = primary.get_executor(&spelled).await.expect("loaded");
        let weak = Arc::downgrade(&executor);
        assert!(serving
            .route(Arc::clone(&executor), &[1])
            .scheduler
            .is_some());
        drop(executor);

        let other = primary.get_executor("other.gguf").await.expect("loads");
        assert!(weak.upgrade().is_none(), "the start model is still held");
        assert!(serving.schedulers().is_empty(), "the engine stopped");
        assert!(serving.route(other, &[1]).scheduler.is_none());
    }

    /// `--model /abs/start.gguf`, and a client that sends the listed id or the
    /// bare name: the request stays on the batching start model.
    #[tokio::test]
    async fn the_listed_id_of_the_start_file_stays_on_the_batching_copy() {
        let dir = model_dir();
        let spelled = dir.path().join("start.gguf").display().to_string();
        let (primary, serving) = loaded(dir.path(), &spelled).await;
        let start = primary.get_executor(&spelled).await.expect("loaded");
        for name in ["start.gguf", "start"] {
            let executor = primary.get_executor(name).await.expect("routes");
            assert!(
                Arc::ptr_eq(&start, &executor),
                "{name} loaded a second copy"
            );
            assert!(
                serving.route(executor, &[1]).scheduler.is_some(),
                "{name} lost batching"
            );
        }
        assert_eq!(serving.schedulers().len(), 1);
        let ids: Vec<String> = primary
            .list_served()
            .expect("lists")
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert_eq!(ids, ["other.gguf", "start.gguf"]);
    }

    /// One bad id frees nothing, whether it resolves to nothing or to a model
    /// the text routes do not run.
    #[tokio::test]
    async fn a_refused_request_leaves_the_start_model_batching() {
        let dir = model_dir();
        std::fs::create_dir(dir.path().join("config-only")).expect("mkdir");
        std::fs::write(dir.path().join("config-only/config.json"), "{}").expect("write");
        std::fs::write(dir.path().join("mmproj-f16.gguf"), b"").expect("write");
        std::fs::write(dir.path().join("corrupt.gguf"), b"not a gguf").expect("write");
        let spelled = dir.path().join("start.gguf").display().to_string();
        let (primary, serving) = loaded(dir.path(), &spelled).await;

        for (name, status) in [
            ("/etc", "404"),
            ("../start.gguf", "404"),
            ("config-only", "404"),
            ("whisper-tiny", "400"),
            ("mmproj-f16.gguf", "400"),
            ("corrupt.gguf", "500"),
        ] {
            let error = primary.get_executor(name).await.err().expect(name);
            let failure = crate::server::handlers::classify_lookup(&error);
            assert_eq!(failure.status.as_str(), status, "{name}: {error:#}");
            assert!(primary.is_loaded(&spelled).await, "{name} freed the model");
            assert_eq!(serving.schedulers().len(), 1, "{name} stopped the engine");
        }
    }
}
