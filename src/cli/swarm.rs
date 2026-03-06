//! Swarm mode CLI command
//!
//! Joins or starts a distributed inference swarm for multi-node model serving.

use anyhow::Result;
use colored::Colorize;
use nexar::NexarClient;
use std::sync::Arc;

use crate::distributed::pipeline::PipelineSchedule;
use crate::distributed::topology::{SwarmConfig, SwarmManager, SwarmNode, SwarmRole};
use crate::distributed::transport;
use crate::distributed::worker::SwarmWorker;

use super::swarm_forward::build_forward_fn;

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

/// Start swarm mode (leader or worker)
pub async fn swarm(
    role: String,
    token: String,
    model: Option<String>,
    leader: Option<String>,
    swarm_port: u16,
    port: u16,
    mdns: bool,
) -> Result<()> {
    let swarm_role = match role.as_str() {
        "leader" => SwarmRole::Leader,
        "worker" => SwarmRole::Worker,
        _ => anyhow::bail!("Invalid role '{}'. Must be 'leader' or 'worker'", role),
    };

    if swarm_role == SwarmRole::Worker && leader.is_none() && !mdns {
        anyhow::bail!("Worker nodes must specify --leader <addr> or --mdns for auto-discovery");
    }

    if swarm_role == SwarmRole::Leader && model.is_none() {
        anyhow::bail!("Leader must specify --model <name>");
    }

    let config = SwarmConfig {
        role: swarm_role,
        token: token.clone(),
        listen_addr: format!("0.0.0.0:{}", swarm_port),
        leader_addr: leader.clone(),
        mdns_discovery: mdns,
    };

    eprintln!();
    eprintln!(
        "  {} v{}",
        "blazr swarm".bold().cyan(),
        env!("CARGO_PKG_VERSION")
    );
    match swarm_role {
        SwarmRole::Leader => {
            eprintln!("  Role: {}", "leader".green().bold());
            eprintln!("  Model: {}", model.as_deref().unwrap_or("none").bold());
            eprintln!("  Swarm transport: 0.0.0.0:{}", swarm_port);
            eprintln!("  HTTP API: 0.0.0.0:{}", port);
        }
        SwarmRole::Worker => {
            eprintln!("  Role: {}", "worker".yellow().bold());
            if let Some(ref addr) = leader {
                eprintln!("  Leader: {}", addr.bold());
            }
            if mdns {
                eprintln!("  Discovery: {}", "mDNS".yellow());
            }
        }
    }
    if mdns {
        eprintln!("  mDNS: {}", "enabled".yellow());
    }
    eprintln!();

    let mut manager = SwarmManager::new(config);

    tracing::info!(
        role = ?swarm_role,
        listen = %format!("0.0.0.0:{}", swarm_port),
        "Swarm node starting"
    );

    if swarm_role == SwarmRole::Leader {
        run_leader(&mut manager, model.as_deref().unwrap(), port).await
    } else {
        run_worker(&manager, leader.as_deref()).await
    }
}

/// Leader: form cluster, assign layers, broadcast model path, start HTTP server.
async fn run_leader(manager: &mut SwarmManager, model_name: &str, http_port: u16) -> Result<()> {
    tracing::info!("Leader awaiting worker connections...");

    // Register self as the leader node
    manager.register_node(SwarmNode {
        node_id: "leader-0".into(),
        address: manager.config().listen_addr.clone(),
        role: SwarmRole::Leader,
        gpu_count: 1,
        vram_per_gpu: vec![0], // Will be updated with actual info
        assigned_layers: None,
    });

    // Load model on leader
    let model_dir = std::env::var("BLAZR_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("./models"));

    #[cfg(feature = "cuda")]
    let device = boostr::CudaDevice::new(0);
    #[cfg(not(feature = "cuda"))]
    let device = boostr::CpuDevice::new();

    let scheduler = std::sync::Arc::new(crate::engine::Scheduler::<ServerRuntime>::new(
        model_dir.clone(),
        device,
    ));

    // Pre-load model
    let executor = scheduler.get_executor(model_name).await?;
    let total_layers = executor.model().num_layers();
    tracing::info!(
        model = model_name,
        layers = total_layers,
        "Model loaded on leader"
    );

    // Compute layer assignments
    let assignments = manager.compute_layer_assignment(total_layers);
    let schedule = PipelineSchedule::new(&assignments, total_layers);

    tracing::info!(
        nodes = manager.node_count(),
        stages = schedule.num_stages(),
        "Pipeline schedule computed"
    );
    for stage in schedule.stages() {
        tracing::info!(
            node = %stage.node_id,
            layers = format!("{}..{}", stage.start_layer, stage.end_layer),
            embedding = stage.has_embedding,
            lm_head = stage.has_lm_head,
            "Pipeline stage"
        );
    }

    // Bootstrap a local nexar mesh for the leader so it can send model path and
    // layer assignments to workers. The world_size covers leader (rank 0) + workers.
    let world_size = manager.node_count() as u32;
    if world_size > 1 {
        let adapter = transport::cpu_adapter();
        let clients = NexarClient::bootstrap_local(world_size, adapter)
            .await
            .map_err(|e| anyhow::anyhow!("Leader nexar bootstrap failed: {}", e))?;

        // Leader takes rank 0
        let leader_client = Arc::new(
            clients
                .into_iter()
                .find(|c| c.rank() == 0)
                .ok_or_else(|| anyhow::anyhow!("Leader could not find rank 0 in bootstrap"))?,
        );

        // Resolve the model path to an absolute string so workers can locate it
        let model_path = model_dir.join(model_name);
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Model path contains non-UTF-8 characters"))?
            .to_owned();
        let path_bytes = model_path_str.as_bytes();

        // Send model path to each worker rank (1..world_size).
        // Protocol: first send a 4-byte little-endian path length, then send the path bytes.
        // Both messages use the MODEL_PATH tag.
        let path_len_bytes = (path_bytes.len() as u32).to_le_bytes();
        for worker_rank in 1..world_size {
            transport::send_bytes(
                &leader_client,
                &path_len_bytes,
                worker_rank,
                transport::tags::MODEL_PATH,
            )
            .await?;
            transport::send_bytes(
                &leader_client,
                path_bytes,
                worker_rank,
                transport::tags::MODEL_PATH,
            )
            .await?;
            tracing::info!(
                rank = worker_rank,
                path = %model_path_str,
                "Sent model path to worker"
            );
        }

        // Wait for all workers to report readiness
        for worker_rank in 1..world_size {
            let mut ack = [0u8; 1];
            transport::recv_bytes(
                &leader_client,
                &mut ack,
                worker_rank,
                transport::tags::WORKER_READY,
            )
            .await?;
            tracing::info!(rank = worker_rank, "Worker ready (acknowledged)");
        }

        tracing::info!("All workers ready, starting HTTP server");
    }

    // Start HTTP server (handles requests, coordinates pipeline)
    let server_config = crate::config::ServerConfig {
        port: http_port,
        host: "0.0.0.0".to_string(),
        ..Default::default()
    };
    crate::server::start(scheduler, server_config, vec![]).await?;

    Ok(())
}

/// Worker: connect to leader, receive assignment and model path, load model, enter compute loop.
async fn run_worker(_manager: &SwarmManager, leader_addr: Option<&str>) -> Result<()> {
    let leader_addr =
        leader_addr.ok_or_else(|| anyhow::anyhow!("Worker requires --leader address"))?;

    tracing::info!(leader = leader_addr, "Worker connecting to leader...");

    // Connect to leader via nexar
    let seed_addr: std::net::SocketAddr = leader_addr
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid leader address '{}': {}", leader_addr, e))?;

    let worker = nexar::WorkerNode::connect(seed_addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to connect to leader: {}", e))?;

    tracing::info!(
        rank = worker.rank,
        world_size = worker.world_size,
        "Connected to swarm"
    );

    // Build mesh with leader
    let adapter = transport::cpu_adapter();
    let clients = NexarClient::bootstrap_local(worker.world_size, adapter)
        .await
        .map_err(|e| anyhow::anyhow!("Mesh formation failed: {}", e))?;

    let our_rank = worker.rank;
    let client = Arc::new(
        clients
            .into_iter()
            .find(|c| c.rank() == our_rank)
            .ok_or_else(|| anyhow::anyhow!("Could not find our client in mesh"))?,
    );

    // Receive layer assignment from leader
    let leader_rank = 0; // Leader is always rank 0
    let assignment = SwarmWorker::receive_assignment(&client, leader_rank).await?;

    tracing::info!(
        rank = our_rank,
        layers = format!("{}..{}", assignment.start_layer, assignment.end_layer),
        embedding = assignment.has_embedding,
        lm_head = assignment.has_lm_head,
        "Received layer assignment"
    );

    // Receive model path from leader.
    // Wire format: 4-byte little-endian length prefix + UTF-8 path bytes.
    let mut len_buf = [0u8; 4];
    transport::recv_bytes(
        &client,
        &mut len_buf,
        leader_rank,
        transport::tags::MODEL_PATH,
    )
    .await?;
    let path_len = u32::from_le_bytes(len_buf) as usize;
    let mut path_buf = vec![0u8; path_len];
    transport::recv_bytes(
        &client,
        &mut path_buf,
        leader_rank,
        transport::tags::MODEL_PATH,
    )
    .await?;
    let model_path = std::str::from_utf8(&path_buf)
        .map_err(|e| anyhow::anyhow!("Model path is not valid UTF-8: {}", e))?
        .to_owned();

    tracing::info!(
        rank = our_rank,
        path = %model_path,
        "Received model path from leader"
    );

    // Load model on worker
    #[cfg(feature = "cuda")]
    let device = boostr::CudaDevice::new(0);
    #[cfg(not(feature = "cuda"))]
    let device = boostr::CpuDevice::new();

    let model_path_ref = std::path::Path::new(&model_path);
    let (loaded_model, _blazr_config) =
        crate::loader::load_model::<ServerRuntime, _>(model_path_ref, &device)?;
    let model = Arc::new(loaded_model);

    tracing::info!(
        rank = our_rank,
        layers = model.num_layers(),
        vocab_size = model.vocab_size(),
        "Model loaded on worker"
    );

    // Determine prev/next ranks in the pipeline
    let world_size = client.world_size();
    let prev_rank = if our_rank > 0 {
        Some(our_rank - 1)
    } else {
        None
    };
    let next_rank = if (our_rank + 1) < world_size {
        Some(our_rank + 1)
    } else {
        None
    };

    // Signal readiness to leader
    SwarmWorker::send_ready(&client, leader_rank).await?;
    tracing::info!(rank = our_rank, "Worker ready, entering compute loop");

    // Build the real forward function for this worker's assigned layers.
    let has_prev_rank = prev_rank.is_some();
    let has_next_rank = next_rank.is_some();
    let forward_fn = build_forward_fn(
        Arc::clone(&model),
        device,
        assignment,
        has_prev_rank,
        has_next_rank,
    );

    // Enter compute loop
    let worker = SwarmWorker::new(
        client,
        our_rank,
        assignment,
        prev_rank,
        next_rank,
        leader_rank,
        forward_fn,
    );
    worker.run_compute_loop().await?;

    eprintln!("\n  Worker shutting down.");
    Ok(())
}
