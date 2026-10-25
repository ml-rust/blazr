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
        let config = DisaggConfig::standard(1, 1, 1 << 26)?;
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

/// A server over `serving`, as `blazr serve` builds its state.
pub(super) fn app(
    scheduler: Arc<crate::engine::Scheduler<crate::runtime::ServerRuntime>>,
    serving: crate::server::handlers::Serving,
) -> axum::Router {
    let metrics = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    let state = crate::server::handlers::AppState::new(scheduler, metrics).with_serving(serving);
    crate::server::api_routes()
        .route(
            "/health",
            axum::routing::get(crate::server::handlers::health),
        )
        .with_state(Arc::new(state))
}

/// The status and body of a JSON `POST` of `body` to `path`.
pub(super) async fn post(
    app: &axum::Router,
    path: &str,
    body: &serde_json::Value,
) -> Result<(u16, String)> {
    use tower::ServiceExt;
    let request = axum::http::Request::post(path)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))?;
    let response = app.clone().oneshot(request).await?;
    let status = response.status().as_u16();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
    Ok((status, String::from_utf8(bytes.to_vec())?))
}

/// The JSON answer of a `POST` that must succeed.
pub(super) async fn ok(
    app: &axum::Router,
    path: &str,
    body: &serde_json::Value,
) -> Result<serde_json::Value> {
    let (status, raw) = post(app, path, body).await?;
    if status != 200 {
        bail!("{path} answered {status}: {raw}");
    }
    Ok(serde_json::from_str(&raw)?)
}

/// The nodes of a cluster formed the production way, on the seeded model:
/// the router binds a seed, and each worker joins with its requested rank.
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub(super) mod nodes {
    use std::net::SocketAddr;
    use std::sync::Arc;

    use anyhow::Result;
    use boostr::{CpuClient, CpuDevice};
    use nexar::{ClusterConfig, SeedNode};
    use serde_json::{json, Value};

    use super::super::disaggregated_forward::{build_decode_open_fn, build_prefill_fn};
    use super::super::disaggregated_layout::DisaggLayout;
    use super::super::disaggregated_mesh::{
        bind_seed, router_client, serve_decode, serve_prefill, start_router, stop_router,
        worker_client,
    };
    use super::super::disaggregated_router::router_serving;
    use super::super::mesh::{parse_addr, MeshSettings};
    use super::{app, ok};
    use crate::cli::{DisaggRoleArg, MeshArgs};
    use crate::engine::test_support::seeded_full_vocab_executor;
    use crate::engine::{Executor, Scheduler};
    use crate::runtime::ServerRuntime;

    pub(crate) const SEED: u64 = 0x0d15;
    pub(crate) const MODEL: &str = "m";
    pub(crate) const TOKEN: &str = "disagg-mesh-test-token";

    pub(crate) fn runtime() -> Result<tokio::runtime::Runtime> {
        Ok(tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?)
    }

    pub(crate) fn layout(role: DisaggRoleArg) -> Result<DisaggLayout> {
        let index = (role != DisaggRoleArg::Router).then_some(0);
        DisaggLayout::resolve(role, index, 1, 1)
    }

    pub(crate) fn cfg(layout: &DisaggLayout) -> Result<ClusterConfig> {
        let mesh = MeshArgs {
            advertise_addr: None,
            join_timeout: 60,
        };
        let settings = MeshSettings::parse("127.0.0.1:0", &mesh, Some(TOKEN.into()))?;
        Ok(settings.cluster_config(layout.world_size(), Some(layout.rank)))
    }

    pub(crate) fn body() -> Value {
        json!({"model": MODEL, "prompt": "Once upon a time", "max_tokens": 10, "temperature": 0})
    }

    /// The completion of `body()` on one process.
    pub(crate) async fn plain_completion() -> Result<Value> {
        let dir = model_dir()?;
        let state = crate::server::handlers::test_state(dir.path());
        state
            .scheduler
            .load_base(MODEL, |_| seeded_full_vocab_executor(SEED))
            .await?;
        let plain = crate::server::api_routes().with_state(Arc::new(state));
        ok(&plain, "/v1/completions", &body()).await
    }

    /// A model directory that names `MODEL`, as the router and the plain server find it.
    fn model_dir() -> Result<tempfile::TempDir> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join(MODEL))?;
        std::fs::write(dir.path().join(MODEL).join("config.json"), "{}")?;
        Ok(dir)
    }

    /// A router seed on a free loopback port.
    pub(crate) fn seed() -> Result<SeedNode> {
        bind_seed(
            parse_addr("seed", "127.0.0.1:0")?,
            &cfg(&layout(DisaggRoleArg::Router)?)?,
        )
    }

    /// The router node: forms the cluster on `seed`, answers `body()`
    /// through the workers, then stops them and shuts down.
    pub(crate) async fn router_node(seed: SeedNode) -> Result<Value> {
        let layout = layout(DisaggRoleArg::Router)?;
        let client = router_client(seed, &cfg(&layout)?).await?;
        let (router, client) = start_router(client, &layout)?;
        let dir = model_dir()?;
        let model = seeded_full_vocab_executor(SEED)?;
        let executor = Executor::tokenizer_only(
            model.config().clone(),
            crate::tokenizer::from_pretrained("llama3")?,
            CpuDevice::new(),
            model.config().max_seq_len(),
        )?;
        let models = Scheduler::<ServerRuntime>::new(dir.path().to_path_buf(), CpuDevice::new());
        let path = dir.path().join(MODEL);
        let (scheduler, serving) =
            router_serving(Arc::clone(&router), MODEL, path, executor, models).await;
        let routed = app(scheduler, serving);
        let got = ok(&routed, "/v1/completions", &body()).await;
        drop(routed);
        stop_router(router, client).await?;
        got
    }

    /// A worker node: joins at `seed`, serves until the router stops it, and shuts down.
    pub(crate) async fn worker_node(role: DisaggRoleArg, seed: SocketAddr) -> Result<()> {
        let layout = layout(role)?;
        let client = worker_client(&layout, seed, &cfg(&layout)?).await?;
        let executor = Arc::new(seeded_full_vocab_executor(SEED)?);
        let sampler = CpuClient::new(CpuDevice::new());
        if role == DisaggRoleArg::Prefill {
            serve_prefill(client, &layout, sampler, build_prefill_fn(executor)).await
        } else {
            serve_decode(client, &layout, sampler, build_decode_open_fn(executor)).await
        }
    }

    pub(crate) fn assert_same(got: &Value, want: &Value) {
        let choice = |v: &Value, f: &str| v["choices"][0][f].clone();
        assert_eq!(choice(got, "text"), choice(want, "text"));
        assert_eq!(choice(got, "finish_reason"), choice(want, "finish_reason"));
        for field in ["prompt_tokens", "completion_tokens", "total_tokens"] {
            assert_eq!(got["usage"][field], want["usage"][field], "{field}");
        }
    }
}
