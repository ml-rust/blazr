//! Cluster formation of the disaggregated roles: the router's seed, the
//! workers' join, and each role's run until its nexar shutdown.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use nexar::{ClusterConfig, NexarClient, SeedNode};

use super::disaggregated_layout::{DisaggLayout, MAX_KV_TRANSFER_BYTES};
use super::mesh::{after_shutdown, formation_error, until_signal, RELEASE_WITHIN};
use crate::distributed::disaggregated::{
    DecodeOpenFn, DecodeWorker, DisaggRole, DisaggRouter, PrefillFn, PrefillWorker,
};
use crate::distributed::transport;
use crate::runtime::ServerRuntime;

/// How a disagg error names the cluster token flag.
pub(super) const TOKEN_FLAG: &str = "--cluster-token <secret> (or NEXAR_CLUSTER_TOKEN)";

/// The sampler client of a worker.
type Sampler = <ServerRuntime as boostr::Runtime>::Client;

/// Binds the router's seed on `router_addr`.
///
/// # Errors
///
/// A missing token for a non-loopback address, or a seed that does not bind.
pub(super) fn bind_seed(router_addr: SocketAddr, cfg: &ClusterConfig) -> Result<SeedNode> {
    SeedNode::from_config(router_addr, cfg).map_err(|e| {
        formation_error(
            &format!("binding the router seed on {router_addr}"),
            e,
            TOKEN_FLAG,
        )
    })
}

/// Forms the cluster as the router, rank 0, on the bound `seed`.
///
/// # Errors
///
/// Any nexar formation error, with its fix.
pub(super) async fn router_client(seed: SeedNode, cfg: &ClusterConfig) -> Result<NexarClient> {
    NexarClient::bootstrap_seed_with(seed, cfg, transport::cpu_adapter())
        .await
        .map_err(|e| formation_error("router cluster formation", e, TOKEN_FLAG))
}

/// Joins the router's cluster at `router_addr` as `layout`'s worker.
///
/// # Errors
///
/// Any nexar formation error, with its fix.
pub(super) async fn worker_client(
    layout: &DisaggLayout,
    router_addr: SocketAddr,
    cfg: &ClusterConfig,
) -> Result<NexarClient> {
    let role = role_name(layout.role);
    let client = NexarClient::bootstrap_join(router_addr, cfg, transport::cpu_adapter())
        .await
        .map_err(|e| formation_error(&format!("{role} worker join"), e, TOKEN_FLAG))?;
    tracing::info!(
        rank = client.rank(),
        world_size = client.world_size(),
        "{role} worker joined"
    );
    Ok(client)
}

/// The log name of `role`.
pub(super) fn role_name(role: DisaggRole) -> &'static str {
    match role {
        DisaggRole::Router => "router",
        DisaggRole::Prefill => "prefill",
        DisaggRole::Decode => "decode",
    }
}

/// Serves prefill requests on `client` until the router's shutdown notice
/// or a signal, then shuts the client down.
pub(super) async fn serve_prefill(
    client: NexarClient,
    layout: &DisaggLayout,
    sampler: Sampler,
    prefill: PrefillFn<ServerRuntime>,
) -> Result<()> {
    let router = layout.config.router_rank;
    let worker = PrefillWorker::new(
        Arc::new(client),
        layout.rank,
        router,
        MAX_KV_TRANSFER_BYTES,
        sampler,
        prefill,
    );
    tracing::info!(rank = layout.rank, "Prefill worker entering run loop");
    let run = until_signal("prefill worker", worker.run_loop()).await;
    let shutdown = worker.shutdown(RELEASE_WITHIN).await;
    after_shutdown("prefill worker", run, shutdown)
}

/// Serves KV handoffs on `client` until the router's shutdown notice or a
/// signal, then shuts the client down.
pub(super) async fn serve_decode(
    client: NexarClient,
    layout: &DisaggLayout,
    sampler: Sampler,
    open: DecodeOpenFn<ServerRuntime>,
) -> Result<()> {
    let worker = DecodeWorker::new(
        Arc::new(client),
        layout.rank,
        layout.config.router_rank,
        layout.config.prefill_workers.clone(),
        MAX_KV_TRANSFER_BYTES,
        sampler,
        open,
    );
    tracing::info!(rank = layout.rank, "Decode worker entering run loop");
    let run = until_signal("decode worker", worker.run_loop()).await;
    let shutdown = worker.shutdown(RELEASE_WITHIN).await;
    after_shutdown("decode worker", run, shutdown)
}

/// The router over the rank-0 `client`, and the client handle the final
/// shutdown takes.
///
/// # Errors
///
/// A client of another rank or world size than `layout` names.
pub(super) fn start_router(
    client: NexarClient,
    layout: &DisaggLayout,
) -> Result<(Arc<DisaggRouter>, Arc<NexarClient>)> {
    let client = Arc::new(client);
    let router = DisaggRouter::new(Arc::clone(&client), layout.config.clone())?;
    Ok((Arc::new(router), client))
}

/// Stops every worker, drops `router`, and shuts the router's client down.
///
/// Every other holder of `router` must drop it first, or the shutdown is
/// skipped with a warning.
pub(super) async fn stop_router(router: Arc<DisaggRouter>, client: Arc<NexarClient>) -> Result<()> {
    router.stop_workers().await;
    drop(router);
    transport::shutdown_shared(client, RELEASE_WITHIN).await
}

/// The cluster formed the production way: the router binds a seed and
/// forms the cluster, and each worker joins with its requested rank, each
/// on its own tokio runtime. One test runs the nodes on threads, one in
/// separate processes.
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use std::future::Future;
    use std::io::{BufRead, BufReader};
    use std::process::{Child, Command, Stdio};
    use std::sync::mpsc;
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use anyhow::{anyhow, bail};
    use serde_json::Value;

    use super::super::disaggregated_test_rig::nodes::{
        assert_same, plain_completion, router_node, runtime, seed, worker_node,
    };
    use super::super::mesh::parse_addr;
    use super::*;
    use crate::cli::DisaggRoleArg;

    const ROLE_ENV: &str = "BLAZR_DISAGG_XPROC_ROLE";
    const SEED_ENV: &str = "BLAZR_DISAGG_XPROC_SEED";
    const SEED_MARKER: &str = "BLAZR_DISAGG_XPROC_SEED_ADDR=";
    const RESULT_MARKER: &str = "BLAZR_DISAGG_XPROC_RESULT=";
    const CHILD_TIMEOUT: Duration = Duration::from_secs(180);

    /// Runs `f` on its own thread with its own tokio runtime, as its own process would.
    fn on_thread<F, Fut, T>(f: F) -> JoinHandle<Result<T>>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<T>>,
        T: Send + 'static,
    {
        std::thread::spawn(move || runtime()?.block_on(f()))
    }

    fn join<T>(node: JoinHandle<Result<T>>) -> Result<T> {
        node.join().map_err(|_| anyhow!("a node thread panicked"))?
    }

    #[test]
    fn the_production_formation_serves_as_one_process_serves() -> Result<()> {
        crate::tls::install_crypto_provider();
        let want = runtime()?.block_on(plain_completion())?;
        let (tx, rx) = mpsc::channel();
        let router = on_thread(move || async move {
            let seed = seed()?;
            tx.send(seed.local_addr())?;
            router_node(seed).await
        });
        let Ok(addr) = rx.recv() else {
            return Err(join(router)
                .err()
                .unwrap_or_else(|| anyhow!("no seed address")));
        };
        let prefill = on_thread(move || worker_node(DisaggRoleArg::Prefill, addr));
        let decode = on_thread(move || worker_node(DisaggRoleArg::Decode, addr));
        let got = join(router)?;
        join(prefill)?;
        join(decode)?;
        assert_same(&got, &want);
        Ok(())
    }

    /// A child process: one node of the cross-process test.
    fn run_child(role: &str) -> Result<()> {
        crate::tls::install_crypto_provider();
        runtime()?.block_on(async {
            let seed_addr = || parse_addr(SEED_ENV, &std::env::var(SEED_ENV)?);
            match role {
                "router" => {
                    let seed = seed()?;
                    println!("{SEED_MARKER}{}", seed.local_addr());
                    println!("{RESULT_MARKER}{}", router_node(seed).await?);
                    Ok(())
                }
                "prefill" => worker_node(DisaggRoleArg::Prefill, seed_addr()?).await,
                "decode" => worker_node(DisaggRoleArg::Decode, seed_addr()?).await,
                other => bail!("unknown {ROLE_ENV}={other}"),
            }
        })
    }

    fn spawn_child(role: &str, seed: Option<SocketAddr>) -> Result<Child> {
        let module = module_path!();
        let path = module.split_once("::").map_or(module, |(_, rest)| rest);
        let mut cmd = Command::new(std::env::current_exe()?);
        cmd.args([
            &format!("{path}::the_disagg_cluster_runs_across_processes"),
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ROLE_ENV, role)
        .stdout(if seed.is_none() {
            Stdio::piped()
        } else {
            Stdio::null()
        });
        if let Some(addr) = seed {
            cmd.env(SEED_ENV, addr.to_string());
        }
        Ok(cmd.spawn()?)
    }

    /// Kills and reaps every child on drop, so an early return leaves no process.
    struct Reap(Vec<(&'static str, Child)>);

    impl Drop for Reap {
        fn drop(&mut self) {
            for (_, child) in &mut self.0 {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    /// Waits for `child` until `deadline`, and kills it after.
    fn wait_one(role: &str, child: &mut Child, deadline: Instant) -> Result<()> {
        loop {
            if let Some(status) = child.try_wait()? {
                if !status.success() {
                    bail!("the {role} process failed: {status}");
                }
                return Ok(());
            }
            if Instant::now() >= deadline {
                child.kill()?;
                bail!("the {role} process ran past {CHILD_TIMEOUT:?}");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Waits for every child until `deadline`. On the first error, kills the rest.
    fn wait_all(children: &mut [(&str, Child)], deadline: Instant) -> Result<()> {
        let mut first = None;
        for (role, child) in children.iter_mut() {
            if first.is_some() {
                let _ = child.kill();
                let _ = child.wait();
            } else if let Err(e) = wait_one(role, child, deadline) {
                first = Some(e);
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Router, prefill and decode in three processes of this test binary.
    #[test]
    fn the_disagg_cluster_runs_across_processes() -> Result<()> {
        if let Ok(role) = std::env::var(ROLE_ENV) {
            return run_child(&role);
        }
        crate::tls::install_crypto_provider();
        let want = runtime()?.block_on(plain_completion())?;
        let deadline = Instant::now() + CHILD_TIMEOUT;
        let mut router = spawn_child("router", None)?;
        let stdout = router
            .stdout
            .take()
            .ok_or_else(|| anyhow!("no router stdout"))?;
        let (tx, rx) = mpsc::channel();
        // libtest can print its own text before a marker on the same line.
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
                for marker in [SEED_MARKER, RESULT_MARKER] {
                    if let Some(at) = line.find(marker) {
                        let _ = tx.send((marker, line[at + marker.len()..].trim().to_string()));
                    }
                }
            }
        });
        let mut children = Reap(vec![("router", router)]);
        let (_, addr) = rx.recv_timeout(CHILD_TIMEOUT)?;
        let addr = parse_addr("seed", &addr)?;
        children
            .0
            .push(("prefill", spawn_child("prefill", Some(addr))?));
        children
            .0
            .push(("decode", spawn_child("decode", Some(addr))?));
        let got = rx.recv_timeout(CHILD_TIMEOUT).map(|(_, json)| json);
        wait_all(&mut children.0, deadline)?;
        let got: Value = serde_json::from_str(&got?)?;
        assert_same(&got, &want);
        Ok(())
    }
}
