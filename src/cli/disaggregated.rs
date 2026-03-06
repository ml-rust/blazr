//! Disaggregated prefill/decode CLI entry points.
//!
//! Provides three roles that can be launched independently:
//! - `router`  — accepts HTTP requests, dispatches to prefill workers, streams tokens
//! - `prefill` — loads a model, runs prompt forward passes, transfers KV cache
//! - `decode`  — receives KV cache, runs autoregressive generation
//!
//! Wire protocol is defined in `crate::distributed::disaggregated`.

use anyhow::Result;

use colored::Colorize;
use nexar::NexarClient;
use std::sync::Arc;

use crate::distributed::disaggregated::{DecodeWorker, DisaggConfig, DisaggRouter, PrefillWorker};
use crate::distributed::transport;

use super::disaggregated_forward::{
    build_decode_step_fn, build_prefill_fn, resolve_model_path, run_router_http_server,
};

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry points
// ─────────────────────────────────────────────────────────────────────────────

/// Run a disaggregated router.
///
/// The router accepts HTTP inference requests and dispatches them to prefill
/// workers, then coordinates the decode workers and streams tokens back over
/// the HTTP response.
pub async fn run_disagg_router(
    listen_addr: String,
    prefill_addrs: Vec<String>,
    decode_addrs: Vec<String>,
    port: u16,
) -> Result<()> {
    eprintln!();
    eprintln!(
        "  {} v{}",
        "blazr disagg router".bold().cyan(),
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  Role: {}", "router".green().bold());
    eprintln!("  Listen: {}", listen_addr.bold());
    eprintln!("  Prefill workers: {:?}", prefill_addrs);
    eprintln!("  Decode workers: {:?}", decode_addrs);
    eprintln!("  HTTP API: 0.0.0.0:{}", port);
    eprintln!();

    let num_prefill = prefill_addrs.len();
    let num_decode = decode_addrs.len();

    if num_prefill == 0 {
        anyhow::bail!("Router requires at least one --prefill address");
    }
    if num_decode == 0 {
        anyhow::bail!("Router requires at least one --decode address");
    }

    // Total world size: 1 router + N prefill + M decode workers.
    let world_size = (1 + num_prefill + num_decode) as u32;

    let adapter = transport::cpu_adapter();
    let clients = NexarClient::bootstrap_local(world_size, adapter)
        .await
        .map_err(|e| anyhow::anyhow!("Router nexar bootstrap failed: {}", e))?;

    // Router is always rank 0.
    let router_rank: nexar::Rank = 0;
    let client = Arc::new(
        clients
            .into_iter()
            .find(|c| c.rank() == router_rank)
            .ok_or_else(|| anyhow::anyhow!("Could not find rank 0 in bootstrap"))?,
    );

    // Assign ranks: prefill workers get ranks 1..=num_prefill,
    // decode workers get ranks num_prefill+1..=num_prefill+num_decode.
    let prefill_ranks: Vec<nexar::Rank> = (1..=num_prefill as u32).collect();
    let decode_ranks: Vec<nexar::Rank> =
        ((num_prefill as u32 + 1)..=(num_prefill + num_decode) as u32).collect();

    let config = DisaggConfig {
        prefill_workers: prefill_ranks,
        decode_workers: decode_ranks,
        router_rank,
        max_kv_transfer_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB default
    };

    let router = Arc::new(DisaggRouter::new(Arc::clone(&client), config));

    tracing::info!(
        prefill_count = num_prefill,
        decode_count = num_decode,
        "Disaggregated router ready"
    );

    // Start HTTP server that uses the router to handle requests.
    let server_config = crate::config::ServerConfig {
        port,
        host: "0.0.0.0".to_string(),
        ..Default::default()
    };

    // Build a scheduler backed by the router's token generation.
    // We use a simple no-model scheduler since the router delegates compute
    // to remote workers.
    run_router_http_server(router, server_config).await
}

/// Run a disaggregated prefill worker.
///
/// The worker:
/// 1. Loads the model from `model`.
/// 2. Connects to the router via nexar.
/// 3. Builds a prefill forward function over the loaded model.
/// 4. Runs `PrefillWorker::run_loop()`.
pub async fn run_disagg_prefill_worker(
    model: String,
    router_addr: String,
    listen_addr: String,
) -> Result<()> {
    eprintln!();
    eprintln!(
        "  {} v{}",
        "blazr disagg prefill".bold().cyan(),
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  Role: {}", "prefill".yellow().bold());
    eprintln!("  Model: {}", model.bold());
    eprintln!("  Router: {}", router_addr.bold());
    eprintln!("  Listen: {}", listen_addr.bold());
    eprintln!();

    // Parse listen address to get the port and derive world_size from config.
    // For a minimal bootstrap we need the rank assigned by the router.
    // We connect as a worker node and obtain our rank from nexar.
    let seed_addr: std::net::SocketAddr = router_addr
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid router address '{}': {}", router_addr, e))?;

    let worker_node = nexar::WorkerNode::connect(seed_addr)
        .await
        .map_err(|e| anyhow::anyhow!("Prefill worker failed to connect to router: {}", e))?;

    let our_rank = worker_node.rank;
    let world_size = worker_node.world_size;

    tracing::info!(rank = our_rank, world_size, "Prefill worker connected");

    let adapter = transport::cpu_adapter();
    let clients = NexarClient::bootstrap_local(world_size, adapter)
        .await
        .map_err(|e| anyhow::anyhow!("Prefill worker mesh formation failed: {}", e))?;

    let client = Arc::new(
        clients
            .into_iter()
            .find(|c| c.rank() == our_rank)
            .ok_or_else(|| anyhow::anyhow!("Prefill worker could not find its rank in mesh"))?,
    );

    // Load model.
    #[cfg(feature = "cuda")]
    let device = boostr::CudaDevice::new(0);
    #[cfg(not(feature = "cuda"))]
    let device = boostr::CpuDevice::new();

    let model_path = resolve_model_path(&model);
    let (loaded_model, _blazr_config) =
        crate::loader::load_model::<ServerRuntime, _>(&model_path, &device)?;
    let model = Arc::new(loaded_model);

    tracing::info!(
        rank = our_rank,
        layers = model.num_layers(),
        vocab_size = model.vocab_size(),
        "Model loaded on prefill worker"
    );

    // Build the prefill forward function.
    let forward_fn = build_prefill_fn(Arc::clone(&model), device);

    let router_rank: nexar::Rank = 0;
    let worker = PrefillWorker::new(
        client,
        our_rank,
        router_rank,
        2 * 1024 * 1024 * 1024, // 2 GiB
        forward_fn,
    );

    tracing::info!(rank = our_rank, "Prefill worker entering run loop");
    worker.run_loop().await?;

    eprintln!("\n  Prefill worker shutting down.");
    Ok(())
}

/// Run a disaggregated decode worker.
///
/// The worker:
/// 1. Loads the model from `model`.
/// 2. Connects to the router via nexar.
/// 3. Builds a decode step function over the loaded model.
/// 4. Runs `DecodeWorker::run_loop()`.
pub async fn run_disagg_decode_worker(
    model: String,
    router_addr: String,
    listen_addr: String,
) -> Result<()> {
    eprintln!();
    eprintln!(
        "  {} v{}",
        "blazr disagg decode".bold().cyan(),
        env!("CARGO_PKG_VERSION")
    );
    eprintln!("  Role: {}", "decode".yellow().bold());
    eprintln!("  Model: {}", model.bold());
    eprintln!("  Router: {}", router_addr.bold());
    eprintln!("  Listen: {}", listen_addr.bold());
    eprintln!();

    let seed_addr: std::net::SocketAddr = router_addr
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid router address '{}': {}", router_addr, e))?;

    let worker_node = nexar::WorkerNode::connect(seed_addr)
        .await
        .map_err(|e| anyhow::anyhow!("Decode worker failed to connect to router: {}", e))?;

    let our_rank = worker_node.rank;
    let world_size = worker_node.world_size;

    tracing::info!(rank = our_rank, world_size, "Decode worker connected");

    let adapter = transport::cpu_adapter();
    let clients = NexarClient::bootstrap_local(world_size, adapter)
        .await
        .map_err(|e| anyhow::anyhow!("Decode worker mesh formation failed: {}", e))?;

    let client = Arc::new(
        clients
            .into_iter()
            .find(|c| c.rank() == our_rank)
            .ok_or_else(|| anyhow::anyhow!("Decode worker could not find its rank in mesh"))?,
    );

    // Load model.
    #[cfg(feature = "cuda")]
    let device = boostr::CudaDevice::new(0);
    #[cfg(not(feature = "cuda"))]
    let device = boostr::CpuDevice::new();

    let model_path = resolve_model_path(&model);
    let (loaded_model, _blazr_config) =
        crate::loader::load_model::<ServerRuntime, _>(&model_path, &device)?;
    let model = Arc::new(loaded_model);

    tracing::info!(
        rank = our_rank,
        layers = model.num_layers(),
        vocab_size = model.vocab_size(),
        "Model loaded on decode worker"
    );

    // Build the decode step function.
    let step_fn = build_decode_step_fn(Arc::clone(&model), device);

    let router_rank: nexar::Rank = 0;
    // By convention: rank 0 = router, ranks 1..our_rank = prefill workers.
    let prefill_workers: Vec<nexar::Rank> = (1..our_rank).collect();
    let worker = DecodeWorker::new(
        client,
        our_rank,
        router_rank,
        prefill_workers,
        2 * 1024 * 1024 * 1024, // 2 GiB
        step_fn,
    );

    tracing::info!(rank = our_rank, "Decode worker entering run loop");
    worker.run_loop().await?;

    eprintln!("\n  Decode worker shutting down.");
    Ok(())
}
