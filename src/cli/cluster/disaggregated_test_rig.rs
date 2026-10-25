//! An in-process disaggregated cluster for the router tests: mock forward
//! closures over a 16-id vocab, and the real closures over a seeded model.
//!
//! The mock model's logits peak at `(token + 1) % VOCAB` for the last fed
//! token. Its "KV cache" is the fed ids as `u32` LE, and each step checks
//! that `position` equals the cache length.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor};
use nexar::NexarClient;

use super::disaggregated_forward::{build_decode_open_fn, build_prefill_fn};
use crate::distributed::disaggregated::{
    DecodeOpenFn, DecodeWorker, DisaggConfig, DisaggRouter, PrefillFn, PrefillOutput, PrefillWorker,
};
use crate::distributed::transport;
use crate::engine::Executor;

/// Ids of the mock model.
pub(super) const VOCAB: usize = 16;

/// The mock model's next id after `token`.
pub(super) fn next_of(token: u32) -> u32 {
    (token + 1) % VOCAB as u32
}

/// `[1, seq_len, VOCAB]` logits whose last row peaks at `peak`, or is flat.
fn logits(seq_len: usize, peak: Option<u32>) -> Result<Tensor<CpuRuntime>> {
    let mut data = vec![0.0f32; seq_len * VOCAB];
    if let Some(p) = peak {
        data[(seq_len - 1) * VOCAB + p as usize] = 10.0;
    }
    Ok(Tensor::<CpuRuntime>::from_slice(
        &data,
        &[1, seq_len, VOCAB],
        &CpuDevice::new(),
    )?)
}

/// Mock closure config, shared across clones.
#[derive(Default, Clone)]
pub(super) struct Mock {
    /// Flat logits, so a sampled rule draws from the whole vocab.
    pub flat: bool,
    /// Every prefill call fails with this text.
    pub prefill_fail: Option<&'static str>,
    /// The 1-based decode step of each request that fails.
    pub fail_at_step: Option<u32>,
    /// A sleep before each decode step.
    pub delay: Option<Duration>,
    /// Decode steps run across all requests.
    pub steps: Arc<AtomicU32>,
    /// Every prompt prefill received, in order.
    pub prompts: Arc<Mutex<Vec<Vec<u32>>>>,
}

impl Mock {
    pub(super) fn steps(&self) -> u32 {
        self.steps.load(Ordering::SeqCst)
    }

    pub(super) fn prompts(&self) -> Vec<Vec<u32>> {
        self.prompts.lock().map(|p| p.clone()).unwrap_or_default()
    }

    fn prefill_fn(&self) -> PrefillFn<CpuRuntime> {
        let opts = self.clone();
        Box::new(move |prompt: &[u32]| {
            if let Ok(mut seen) = opts.prompts.lock() {
                seen.push(prompt.to_vec());
            }
            if let Some(msg) = opts.prefill_fail {
                bail!("{msg}");
            }
            let last = *prompt.last().ok_or_else(|| anyhow!("empty prompt"))?;
            Ok(PrefillOutput {
                logits: logits(prompt.len(), (!opts.flat).then(|| next_of(last)))?,
                kv_cache: prompt.iter().flat_map(|t| t.to_le_bytes()).collect(),
            })
        })
    }

    fn open_fn(&self) -> DecodeOpenFn<CpuRuntime> {
        let opts = self.clone();
        Box::new(move |kv: &[u8]| {
            if !kv.len().is_multiple_of(4) {
                bail!("mock cache of {} bytes is not whole u32 ids", kv.len());
            }
            let mut cache: Vec<u32> = kv
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| u32::from_le_bytes(c))
                .collect();
            let opts = opts.clone();
            let mut step_no = 0u32;
            Ok(Box::new(move |token: u32, position: u32| {
                step_no += 1;
                opts.steps.fetch_add(1, Ordering::SeqCst);
                if let Some(d) = opts.delay {
                    std::thread::sleep(d);
                }
                if opts.fail_at_step == Some(step_no) {
                    bail!("mock step failure at step {step_no}");
                }
                if position as usize != cache.len() {
                    bail!("position {position} but the cache holds {}", cache.len());
                }
                cache.push(token);
                logits(1, (!opts.flat).then(|| next_of(token)))
            }) as _)
        })
    }
}

/// A router on rank 0, one prefill worker on rank 1, one decode worker on rank 2.
pub(super) struct Cluster {
    pub router: Arc<DisaggRouter>,
    workers: Vec<tokio::task::JoinHandle<Result<()>>>,
}

impl Cluster {
    /// The cluster over the mock closures `mock` configures.
    pub(super) async fn mock(mock: &Mock) -> Result<Self> {
        Self::start(mock.prefill_fn(), mock.open_fn()).await
    }

    /// The cluster over the real closures: prefill on `prefill`, decode on `decode`.
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    pub(super) async fn real(
        prefill: Executor<CpuRuntime>,
        decode: Executor<CpuRuntime>,
    ) -> Result<Self> {
        Self::start(
            build_prefill_fn(Arc::new(prefill)),
            build_decode_open_fn(Arc::new(decode)),
        )
        .await
    }

    async fn start(prefill: PrefillFn<CpuRuntime>, open: DecodeOpenFn<CpuRuntime>) -> Result<Self> {
        crate::tls::install_crypto_provider();
        let mut clients = NexarClient::bootstrap_local(3, transport::cpu_adapter())
            .await
            .map_err(|e| anyhow!("local nexar bootstrap failed: {e}"))?;
        clients.sort_by_key(|c| c.rank());
        let mut clients = clients.into_iter().map(Arc::new);
        let mut next = || clients.next().ok_or_else(|| anyhow!("too few ranks"));
        let (router, prefill_c, decode_c) = (next()?, next()?, next()?);
        let config = DisaggConfig {
            prefill_workers: vec![1],
            decode_workers: vec![2],
            router_rank: 0,
            max_kv_transfer_bytes: 1 << 26,
        };
        let sampler = CpuClient::new(CpuDevice::new());
        let p = PrefillWorker::new(prefill_c, 1, 0, 1 << 26, sampler.clone(), prefill);
        let d = DecodeWorker::new(decode_c, 2, 0, vec![1], 1 << 26, sampler, open);
        let workers = vec![
            tokio::spawn(async move { p.run_loop().await }),
            tokio::spawn(async move { d.run_loop().await }),
        ];
        Ok(Self {
            router: Arc::new(DisaggRouter::new(router, config)?),
            workers,
        })
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        for w in &self.workers {
            w.abort();
        }
    }
}
