//! Disaggregated prefill/decode CLI entry points.
//!
//! Three roles launch independently:
//! - `router`: serves the HTTP API with the model's tokenizer and chat
//!   template, and routes every generation to the workers
//! - `prefill`: loads the model, runs each prompt, samples the first token
//!   and hands the KV cache to a decode worker
//! - `decode`: loads the model, resumes from the KV cache and streams the
//!   remaining tokens
//!
//! boostr's `distributed::inference::disaggregated` defines the protocol.

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use colored::Colorize;
use nexar::{NexarClient, Rank};

use super::disaggregated_forward::{build_decode_open_fn, build_prefill_fn, check_disagg_model};
use super::disaggregated_router::{router_executor, router_serving};
use crate::config::ClusterSettings;
use crate::distributed::disaggregated::{DecodeWorker, DisaggConfig, DisaggRouter, PrefillWorker};
use crate::distributed::transport;
use crate::engine::{Executor, Scheduler};
use crate::runtime::ServerRuntime;
use crate::server::{start_with_batch, LiveSettings};

/// The largest serialized KV cache one request transfers.
const MAX_KV_TRANSFER_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// The router's nexar rank.
const ROUTER_RANK: Rank = 0;

fn banner(role: &str, lines: &[(&str, String)]) {
    eprintln!();
    eprintln!(
        "  {} v{}",
        format!("blazr disagg {role}").bold().cyan(),
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  Role: {}", role.green().bold());
    for (label, value) in lines {
        eprintln!("  {label}: {}", value.bold());
    }
    eprintln!();
}

/// Run a disaggregated router.
///
/// The router loads the config, tokenizer and chat template of the model
/// `settings` names, and no weights. It serves the whole HTTP API that
/// `blazr serve` serves, and each generation runs on a prefill and a decode
/// worker. `settings.server` holds the HTTP host and port.
///
/// # Errors
///
/// No `--model`, no prefill or decode address, a model whose tokenizer does
/// not load, a failed nexar bootstrap, or a server that does not start.
pub async fn run_disagg_router(
    listen_addr: String,
    prefill_addrs: Vec<String>,
    decode_addrs: Vec<String>,
    settings: &ClusterSettings,
) -> Result<()> {
    let model = settings.router_model().map_err(anyhow::Error::msg)?;
    if prefill_addrs.is_empty() {
        bail!("the router needs at least one --prefill address");
    }
    if decode_addrs.is_empty() {
        bail!("the router needs at least one --decode address");
    }
    banner(
        "router",
        &[
            ("Model", model.clone()),
            ("Listen", listen_addr),
            ("Prefill workers", format!("{prefill_addrs:?}")),
            ("Decode workers", format!("{decode_addrs:?}")),
            ("HTTP API", settings.server.addr()),
        ],
    );

    let device = crate::runtime::open_device(settings.gpu_index)?;
    let models = Scheduler::<ServerRuntime>::new(settings.model_dir.clone(), device.clone());
    let path = models.model_path(&model).ok_or_else(|| {
        anyhow!(
            "model '{model}' not found as a path or under {}: the router reads its \
                 tokenizer and chat template",
            settings.model_dir.display()
        )
    })?;
    let executor = router_executor(&path, device)?;

    let num_prefill = prefill_addrs.len() as u32;
    let num_decode = decode_addrs.len() as u32;
    let world_size = 1 + num_prefill + num_decode;
    let clients = NexarClient::bootstrap_local(world_size, transport::cpu_adapter())
        .await
        .map_err(|e| anyhow!("router nexar bootstrap failed: {e}"))?;
    let client = Arc::new(
        clients
            .into_iter()
            .find(|c| c.rank() == ROUTER_RANK)
            .ok_or_else(|| anyhow!("the nexar bootstrap returned no rank {ROUTER_RANK}"))?,
    );
    // Ranks 1..=num_prefill prefill. The decode workers follow.
    let config = DisaggConfig {
        prefill_workers: (1..=num_prefill).collect(),
        decode_workers: (num_prefill + 1..world_size).collect(),
        router_rank: ROUTER_RANK,
        max_kv_transfer_bytes: MAX_KV_TRANSFER_BYTES,
    };
    let router = Arc::new(DisaggRouter::new(client, config)?);
    tracing::info!(
        prefill_count = num_prefill,
        decode_count = num_decode,
        "Disaggregated router ready"
    );

    let (scheduler, serving) = router_serving(router, &model, path, executor, models).await;
    let server = settings.server.clone();
    let live = LiveSettings::fixed(&server);
    start_with_batch(
        scheduler,
        server,
        live,
        settings.api_key.clone().into_iter().collect(),
        serving,
        Vec::new(),
        #[cfg(feature = "audio")]
        Vec::new(),
        #[cfg(feature = "audio")]
        Vec::new(),
        None,
    )
    .await
}

/// A worker's nexar client: connect to the router, then join the mesh.
async fn worker_client(role: &str, router_addr: &str) -> Result<Arc<NexarClient>> {
    let seed_addr: std::net::SocketAddr = router_addr
        .parse()
        .map_err(|e| anyhow!("invalid router address '{router_addr}': {e}"))?;
    let node = nexar::WorkerNode::connect(seed_addr)
        .await
        .map_err(|e| anyhow!("{role} worker failed to connect to the router: {e}"))?;
    let (rank, world_size) = (node.rank, node.world_size);
    tracing::info!(rank, world_size, "{role} worker connected");
    let clients = NexarClient::bootstrap_local(world_size, transport::cpu_adapter())
        .await
        .map_err(|e| anyhow!("{role} worker mesh formation failed: {e}"))?;
    let client = clients
        .into_iter()
        .find(|c| c.rank() == rank)
        .ok_or_else(|| anyhow!("{role} worker found no rank {rank} in the mesh"))?;
    Ok(Arc::new(client))
}

/// The started executor of the model `settings` names, checked for the
/// disaggregated `role`.
async fn worker_executor(
    role: &str,
    settings: &ClusterSettings,
) -> Result<Arc<Executor<ServerRuntime>>> {
    let model = settings
        .require_model(&format!("disagg {role}"))
        .map_err(anyhow::Error::msg)?;
    let device = crate::runtime::open_device(settings.gpu_index)?;
    let loader = Scheduler::<ServerRuntime>::new(settings.model_dir.clone(), device);
    let executor = loader.get_executor(&model).await?;
    check_disagg_model(&executor, role)?;
    tracing::info!(
        model = %model,
        vocab_size = executor.vocab_size(),
        "Model loaded on {role} worker"
    );
    Ok(executor)
}

/// Run a disaggregated prefill worker.
///
/// It loads the model `settings` names, joins the router's mesh and serves
/// prefill requests until the router disconnects.
///
/// # Errors
///
/// No model, a model the KV handoff cannot carry, or a failed connection.
pub async fn run_disagg_prefill_worker(
    router_addr: String,
    listen_addr: String,
    settings: &ClusterSettings,
) -> Result<()> {
    banner(
        "prefill",
        &[
            ("Model", settings.model.clone().unwrap_or_default()),
            ("Router", router_addr.clone()),
            ("Listen", listen_addr),
        ],
    );
    let executor = worker_executor("prefill", settings).await?;
    let client = worker_client("prefill", &router_addr).await?;
    let rank = client.rank();
    let sampler = <ServerRuntime as boostr::Runtime>::default_client(executor.device());
    let worker = PrefillWorker::new(
        client,
        rank,
        ROUTER_RANK,
        MAX_KV_TRANSFER_BYTES,
        sampler,
        build_prefill_fn(executor),
    );
    tracing::info!(rank, "Prefill worker entering run loop");
    worker.run_loop().await?;
    eprintln!("\n  Prefill worker shutting down.");
    Ok(())
}

/// Run a disaggregated decode worker.
///
/// It loads the model `settings` names, joins the router's mesh and serves
/// KV handoffs until every prefill worker disconnects.
///
/// # Errors
///
/// No model, a model the KV handoff cannot carry, or a failed connection.
pub async fn run_disagg_decode_worker(
    router_addr: String,
    listen_addr: String,
    settings: &ClusterSettings,
) -> Result<()> {
    banner(
        "decode",
        &[
            ("Model", settings.model.clone().unwrap_or_default()),
            ("Router", router_addr.clone()),
            ("Listen", listen_addr),
        ],
    );
    let executor = worker_executor("decode", settings).await?;
    let client = worker_client("decode", &router_addr).await?;
    let rank = client.rank();
    // Each rank between the router and this one gets a handoff listener.
    // A decode rank among them sends no handoff, so its listener stays idle.
    let prefill_workers: Vec<Rank> = (1..rank).collect();
    let sampler = <ServerRuntime as boostr::Runtime>::default_client(executor.device());
    let worker = DecodeWorker::new(
        client,
        rank,
        ROUTER_RANK,
        prefill_workers,
        MAX_KV_TRANSFER_BYTES,
        sampler,
        build_decode_open_fn(executor),
    );
    tracing::info!(rank, "Decode worker entering run loop");
    worker.run_loop().await?;
    eprintln!("\n  Decode worker shutting down.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::cli::Cli;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("blazr").chain(args.iter().copied()))
    }

    #[test]
    fn the_router_role_requires_model() {
        let err = parse(&[
            "disagg",
            "--role",
            "router",
            "--prefill",
            "a:1",
            "--decode",
            "b:2",
        ])
        .err()
        .expect("a router without --model is refused");
        assert!(err.to_string().contains("--model"), "{err}");
        assert!(parse(&["disagg", "--role", "router", "--model", "m"]).is_ok());
    }

    #[test]
    fn a_worker_role_takes_its_model_from_the_config() {
        assert!(parse(&["disagg", "--role", "prefill", "--router-addr", "a:1"]).is_ok());
        assert!(parse(&["disagg", "--role", "decode", "--router-addr", "a:1"]).is_ok());
    }
}
