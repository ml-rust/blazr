//! Loads the start model of `blazr serve`: one copy, or one copy per
//! data-parallel replica.
//!
//! Replica `i` runs on CUDA device `device + i` on a CUDA build, and on its
//! own CPU copy on a CPU build. Each copy gets its own batching scheduler.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use colored::Colorize;

use crate::config::{tp_conflict, InferenceOverlay, VisionOverrides};
use crate::engine::{DataParallelGroup, Executor, Scheduler, SloBatchPolicy};
use crate::server::handlers::{ServeReplica, ServerDevice, ServerRuntime, Serving};

/// The settings every model scheduler of one server shares.
#[derive(Debug, Clone)]
pub struct SchedulerSettings {
    pub model_dir: PathBuf,
    pub num_ctx: usize,
    pub chat_template: Option<String>,
    pub inference: InferenceOverlay,
    pub tensor_parallel_size: Option<usize>,
    pub vision: VisionOverrides,
}

impl SchedulerSettings {
    /// A model scheduler on `device`.
    pub fn build(&self, device: ServerDevice) -> Scheduler<ServerRuntime> {
        Scheduler::new(self.model_dir.clone(), device)
            .with_num_ctx(self.num_ctx)
            .with_chat_template(self.chat_template.clone())
            .with_inference_overlay(self.inference.clone())
            .with_tensor_parallel_size(self.tensor_parallel_size)
            .with_vision(self.vision.clone())
    }
}

/// Load `model` for serving.
///
/// `primary` is the server's model scheduler on CUDA device `base_index`.
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
        return Ok(Serving::Single(replica));
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
        let spinner = super::util::spinner(format!("Loading replica {i} on {label}..."));
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
    if cfg!(feature = "cuda") {
        format!("cuda:{}", base_index + i)
    } else {
        format!("cpu:{i}")
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

/// One device per replica: a CPU device for each copy.
#[cfg(not(feature = "cuda"))]
fn replica_devices(base_index: usize, replicas: usize) -> Result<Vec<ServerDevice>> {
    let _ = base_index;
    Ok((0..replicas).map(|_| boostr::CpuDevice::new()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_replica_gets_one_device_with_a_label() {
        let devices = replica_devices(0, 3);
        if cfg!(feature = "cuda") {
            // A CUDA host with fewer than three devices refuses the count.
            return;
        }
        assert_eq!(devices.map(|d| d.len()).ok(), Some(3));
        assert_eq!(device_label(0, 2), "cpu:2");
    }
}
