//! A single-process tensor-parallel group: rank 0 plus follower rank threads.
//!
//! The executor drives rank 0. Each other rank runs in its own thread and
//! owns its model shard and its own KV cache. Before each rank-0 forward,
//! [`TpGroup::forward_with_kv_cache`] sends the step to every follower.
//! The token ids reach the followers through an NCCL broadcast from rank 0,
//! so a decode step needs no host copy.
//!
//! A step runs in two phases. First every follower prepares its buffers and
//! answers. Only then does any rank issue a collective. A setup error so
//! returns at once, and no rank waits forever in NCCL for a missing peer.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use anyhow::{anyhow, Result};
use boostr::distributed::broadcast_tensor;
use boostr::inference::LayeredKvCache;
use boostr::model::{LoadedModel, ModelClient};
use boostr::runtime::Communicator;
use boostr::{CudaDevice, CudaRuntime, DType, Runtime, Tensor};

use super::tensor_parallel::KvCacheSpec;
use crate::config::DtypeRequest;
use crate::loader;

/// One forward step, as rank 0 tells it to a follower.
struct Step {
    input_shape: Vec<usize>,
    input_dtype: DType,
    position: usize,
    fresh_cache: Option<KvCacheSpec>,
}

enum Command {
    /// Prepare the step and answer on the ready channel.
    Prepare(Step),
    /// Run the prepared step and answer on the done channel.
    Run,
    Stop,
}

struct Follower {
    rank: usize,
    commands: Sender<Command>,
    ready: Receiver<Result<(), String>>,
    done: Receiver<Result<(), String>>,
    thread: Option<JoinHandle<()>>,
}

/// Rank threads 1..world_size of one tensor-parallel group.
pub struct TpGroup {
    rank0_comm: Arc<dyn Communicator>,
    /// A `Mutex` makes the channel ends `Sync`. The generation lock already
    /// serializes steps, so it never contends.
    followers: Mutex<Vec<Follower>>,
    /// Held for a whole generation. Two sequences never interleave their
    /// collectives, and the followers keep one KV cache each.
    generation: Arc<tokio::sync::Mutex<()>>,
}

impl TpGroup {
    /// Start one thread per follower rank. Each thread loads its shard on
    /// device `base_device + rank` and waits for steps.
    ///
    /// `comms[r]` is rank r's communicator. `comms[0]` stays with the caller,
    /// which loads rank 0 itself. Call [`TpGroup::wait_loaded`] after that.
    pub fn spawn(
        comms: &[Arc<dyn Communicator>],
        base_device: usize,
        model_path: &Path,
        dtype: Option<DtypeRequest>,
    ) -> Result<Self> {
        let rank0_comm = comms
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("a tensor-parallel group needs at least one rank"))?;
        let mut followers = Vec::with_capacity(comms.len().saturating_sub(1));
        for (rank, comm) in comms.iter().enumerate().skip(1) {
            followers.push(spawn_follower(
                rank,
                Arc::clone(comm),
                base_device + rank,
                model_path.to_path_buf(),
                dtype,
            )?);
        }
        Ok(Self {
            rank0_comm,
            followers: Mutex::new(followers),
            generation: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    /// Block until every follower has loaded its shard.
    pub fn wait_loaded(&self) -> Result<()> {
        for follower in self.lock_followers()?.iter() {
            recv_answer(&follower.ready, follower.rank, "load")?;
        }
        Ok(())
    }

    /// The lock a caller holds for one whole generation.
    pub fn generation_lock(&self) -> Arc<tokio::sync::Mutex<()>> {
        Arc::clone(&self.generation)
    }

    /// Run one forward step on every rank. Rank 0 runs on the caller's
    /// thread and returns the full-vocab logits.
    ///
    /// `fresh_cache` is set when rank 0 has just built `kv_cache`. Each
    /// follower then builds a matching cache for its own head shard.
    pub fn forward_with_kv_cache<R>(
        &self,
        model: &LoadedModel<R>,
        input: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
        fresh_cache: Option<KvCacheSpec>,
    ) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ModelClient<R> + boostr::IndexingOps<R>,
    {
        let input = input.contiguous()?;
        let followers = self.lock_followers()?;
        for follower in followers.iter() {
            send(
                follower,
                Command::Prepare(Step {
                    input_shape: input.shape().to_vec(),
                    input_dtype: input.dtype(),
                    position,
                    fresh_cache,
                }),
            )?;
        }
        for follower in followers.iter() {
            recv_answer(&follower.ready, follower.rank, "prepare")?;
        }
        for follower in followers.iter() {
            send(follower, Command::Run)?;
        }
        broadcast_tensor(self.rank0_comm.as_ref(), &input, 0)
            .map_err(|e| anyhow!("tensor-parallel rank 0: input broadcast failed: {e}"))?;
        let logits = model
            .forward_with_kv_cache(&input, kv_cache, position)
            .map_err(|e| anyhow!("tensor-parallel rank 0: forward failed: {e}"));
        for follower in followers.iter() {
            recv_answer(&follower.done, follower.rank, "forward")?;
        }
        logits
    }

    fn lock_followers(&self) -> Result<MutexGuard<'_, Vec<Follower>>> {
        self.followers.lock().map_err(|_| {
            anyhow!("a tensor-parallel step panicked and left the rank group unusable. Reload the model")
        })
    }
}

impl Drop for TpGroup {
    fn drop(&mut self) {
        let followers = match self.followers.get_mut() {
            Ok(followers) => followers,
            Err(poisoned) => poisoned.into_inner(),
        };
        for follower in followers.iter() {
            let _ = follower.commands.send(Command::Stop);
        }
        for follower in followers.iter_mut() {
            if let Some(thread) = follower.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

fn send(follower: &Follower, command: Command) -> Result<()> {
    follower.commands.send(command).map_err(|_| {
        anyhow!(
            "tensor-parallel rank {} has stopped. Reload the model",
            follower.rank
        )
    })
}

fn recv_answer(channel: &Receiver<Result<(), String>>, rank: usize, what: &str) -> Result<()> {
    match channel.recv() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(anyhow!("tensor-parallel rank {rank}: {what} failed: {e}")),
        Err(_) => Err(anyhow!(
            "tensor-parallel rank {rank} stopped during {what}. Reload the model"
        )),
    }
}

fn spawn_follower(
    rank: usize,
    comm: Arc<dyn Communicator>,
    device_index: usize,
    model_path: PathBuf,
    dtype: Option<DtypeRequest>,
) -> Result<Follower> {
    let (commands, command_rx) = mpsc::channel();
    let (ready_tx, ready) = mpsc::channel();
    let (done_tx, done) = mpsc::channel();
    let thread = std::thread::Builder::new()
        .name(format!("blazr-tp-rank{rank}"))
        .spawn(move || {
            let device = CudaDevice::new(device_index);
            let model = match loader::load_model_tp::<CudaRuntime, _>(
                &model_path,
                &device,
                comm,
                dtype.as_ref(),
            ) {
                Ok((model, _config)) => model,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            if ready_tx.send(Ok(())).is_err() {
                return;
            }
            follower_loop(&model, &device, &command_rx, &ready_tx, &done_tx);
        })
        .map_err(|e| anyhow!("could not start the thread for tensor-parallel rank {rank}: {e}"))?;
    Ok(Follower {
        rank,
        commands,
        ready,
        done,
        thread: Some(thread),
    })
}

fn follower_loop(
    model: &LoadedModel<CudaRuntime>,
    device: &CudaDevice,
    commands: &Receiver<Command>,
    ready: &Sender<Result<(), String>>,
    done: &Sender<Result<(), String>>,
) {
    let comm = match model.tp_communicator() {
        Some(comm) => comm,
        None => return,
    };
    let mut kv_cache: Option<LayeredKvCache<CudaRuntime>> = None;
    let mut prepared: Option<(Tensor<CudaRuntime>, usize)> = None;
    while let Ok(command) = commands.recv() {
        match command {
            Command::Prepare(step) => {
                let answer = prepare(model, device, &step, &mut kv_cache).map(|input| {
                    prepared = Some((input, step.position));
                });
                if ready.send(answer.map_err(|e| format!("{e:#}"))).is_err() {
                    return;
                }
            }
            Command::Run => {
                let answer = match (prepared.take(), kv_cache.as_mut()) {
                    (Some((input, position)), Some(cache)) => {
                        run(model, comm, &input, cache, position)
                    }
                    _ => Err(anyhow!("a step ran with no prepared input or KV cache")),
                };
                if done.send(answer.map_err(|e| format!("{e:#}"))).is_err() {
                    return;
                }
            }
            Command::Stop => return,
        }
    }
}

fn prepare(
    model: &LoadedModel<CudaRuntime>,
    device: &CudaDevice,
    step: &Step,
    kv_cache: &mut Option<LayeredKvCache<CudaRuntime>>,
) -> Result<Tensor<CudaRuntime>> {
    if let Some(spec) = step.fresh_cache {
        let num_kv_heads = model
            .num_kv_heads()
            .ok_or_else(|| anyhow!("the model shard reports no key/value head count"))?;
        let head_dim = model
            .head_dim()
            .ok_or_else(|| anyhow!("the model shard reports no head dimension"))?;
        *kv_cache = Some(LayeredKvCache::new_positional(
            model.num_layers(),
            1,
            num_kv_heads,
            spec.initial_capacity,
            spec.max_seq_len,
            head_dim,
            spec.dtype,
            device,
        )?);
    }
    if kv_cache.is_none() {
        return Err(anyhow!(
            "a step at position {} arrived before any KV cache was built",
            step.position
        ));
    }
    Ok(Tensor::<CudaRuntime>::zeros(
        &step.input_shape,
        step.input_dtype,
        device,
    )?)
}

fn run(
    model: &LoadedModel<CudaRuntime>,
    comm: &dyn Communicator,
    input: &Tensor<CudaRuntime>,
    kv_cache: &mut LayeredKvCache<CudaRuntime>,
    position: usize,
) -> Result<()> {
    broadcast_tensor(comm, input, 0).map_err(|e| anyhow!("input broadcast failed: {e}"))?;
    model.forward_with_kv_cache(input, kv_cache, position)?;
    Ok(())
}
