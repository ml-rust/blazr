//! Swarm mode CLI command
//!
//! Starts the leader or joins a worker of a pipeline swarm, one process per
//! node. The leader binds nexar's seed and holds rank 0. Workers take ranks
//! in arrival order: the leader's schedule runs stage `r` on rank `r`, so a
//! worker's previous and next stage follow from its rank.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use colored::Colorize;
use nexar::NexarClient;

use super::mesh::{
    after_shutdown, formation_error, parse_addr, until_signal, MeshSettings, RELEASE_WITHIN,
};
use super::swarm_forward::build_forward_fn;
use crate::cli::{SwarmArgs, SwarmRoleArg};
use crate::config::ClusterSettings;
use crate::distributed::leader::SwarmLeader;
use crate::distributed::pipeline::PipelineSchedule;
use crate::distributed::transport;
use crate::distributed::worker::SwarmWorker;
use crate::runtime::ServerRuntime;

/// How a swarm error names the cluster token flag.
const TOKEN_FLAG: &str = "--token <secret> (or BLAZR_SWARM_TOKEN)";

/// The leader's rank.
const LEADER_RANK: u32 = 0;

/// Start swarm mode (leader or worker).
///
/// `settings` holds the model, model directory, device and the leader's
/// HTTP `server` section, resolved from flags, environment and the file.
///
/// # Errors
///
/// A bad flag or address, a model that does not load, a failed nexar
/// formation, or a server that does not start.
pub async fn swarm(args: SwarmArgs, settings: ClusterSettings) -> Result<()> {
    let mesh = MeshSettings::parse(&args.listen_addr, &args.mesh, Some(args.token.clone()))?;
    let world_size = args
        .workers
        .checked_add(1)
        .ok_or_else(|| anyhow!("--workers {} is too large", args.workers))?;

    eprintln!();
    eprintln!(
        "  {} v{}",
        "blazr swarm".bold().cyan(),
        env!("CARGO_PKG_VERSION")
    );
    match args.role {
        SwarmRoleArg::Leader => {
            let model = settings
                .require_model("swarm leader")
                .map_err(anyhow::Error::msg)?;
            eprintln!("  Role: {}", "leader".green().bold());
            eprintln!("  Model: {}", model.bold());
            // The seed binds the mesh listener's IP, on its own UDP port.
            let seed = SocketAddr::new(mesh.listen.ip(), args.swarm_port);
            eprintln!("  Seed: {seed} (UDP)");
            eprintln!("  Mesh: {}", mesh.listen);
            eprintln!("  Workers: {}", args.workers);
            eprintln!("  HTTP API: {}", settings.server.addr());
            eprintln!();
            run_leader(&model, seed, &mesh, world_size, settings).await
        }
        SwarmRoleArg::Worker => {
            if args.workers == 0 {
                bail!("a worker needs --workers 1 or more: the same value the leader passes");
            }
            let leader = args
                .leader
                .as_deref()
                .ok_or_else(|| anyhow!("a worker needs --leader <leader-host>:<swarm-port>"))?;
            let leader = parse_addr("--leader", leader)?;
            eprintln!("  Role: {}", "worker".yellow().bold());
            eprintln!("  Leader: {}", leader.to_string().bold());
            eprintln!("  Mesh: {}", mesh.listen);
            eprintln!();
            run_worker(leader, &mesh, world_size, settings.gpu_index).await
        }
    }
}

/// Leader: form the cluster, assign layers, send the model path, serve HTTP,
/// then stop every worker and shut down.
async fn run_leader(
    model_name: &str,
    seed: SocketAddr,
    mesh: &MeshSettings,
    world_size: u32,
    settings: ClusterSettings,
) -> Result<()> {
    let ClusterSettings {
        model_dir,
        gpu_index,
        server: server_config,
        api_key,
        ..
    } = settings;
    let device = crate::runtime::open_device(gpu_index)?;
    let scheduler = Arc::new(crate::engine::Scheduler::<ServerRuntime>::new(
        model_dir.clone(),
        device,
    ));
    let executor = scheduler.get_executor(model_name).await?;
    let total_layers = executor.model()?.num_layers();
    let vocab_size = executor.vocab_size();
    tracing::info!(
        model = model_name,
        layers = total_layers,
        "Model loaded on leader"
    );
    let serve = || crate::server::start(scheduler, server_config, api_key.into_iter().collect());
    if world_size == 1 {
        tracing::info!("No workers: the leader serves alone");
        return serve().await;
    }
    // Refuse a swarm wider than the model before any worker joins.
    PipelineSchedule::by_rank(world_size, total_layers)?;
    mesh.check_seed_reachable(seed, "the seed", world_size)?;

    let cfg = mesh.cluster_config(world_size, Some(LEADER_RANK));
    tracing::info!(%seed, world_size, "Leader waiting for workers");
    let client = SwarmLeader::form_cluster(seed, &cfg)
        .await
        .map_err(|e| formation_error("swarm leader formation", e, TOKEN_FLAG))?;
    let leader = SwarmLeader::by_rank(Arc::new(client), total_layers, vocab_size)?;
    for stage in leader.schedule().stages() {
        tracing::info!(
            node = %stage.node_id,
            layers = format!("{}..{}", stage.start_layer, stage.end_layer),
            "Pipeline stage"
        );
    }
    let model_path = model_dir.join(model_name);
    let run = async {
        let path = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path {} is not UTF-8", model_path.display()))?;
        leader.assign_layers().await?;
        leader.send_model_path(path).await?;
        leader.wait_for_workers().await?;
        tracing::info!("All workers ready, starting HTTP server");
        serve().await
    }
    .await;
    let shutdown = leader.shutdown(RELEASE_WITHIN).await;
    after_shutdown("swarm leader", run, shutdown)
}

/// Worker: join the leader's cluster, run its stage until the leader stops
/// it or a signal arrives, then shut down.
async fn run_worker(
    leader: SocketAddr,
    mesh: &MeshSettings,
    world_size: u32,
    gpu_index: usize,
) -> Result<()> {
    tracing::info!(%leader, "Worker joining the swarm");
    let cfg = mesh.cluster_config(world_size, None);
    let client = NexarClient::bootstrap_join(leader, &cfg, transport::cpu_adapter())
        .await
        .map_err(|e| formation_error("swarm worker join", e, TOKEN_FLAG))?;
    let client = Arc::new(client);
    tracing::info!(rank = client.rank(), world_size, "Joined the swarm");
    let run = run_stage(Arc::clone(&client), gpu_index).await;
    let shutdown = transport::shutdown_shared(client, RELEASE_WITHIN).await;
    eprintln!("\n  Worker shut down.");
    after_shutdown("swarm worker", run, shutdown)
}

/// Receive this worker's stage and model, load it, and run the compute loop.
async fn run_stage(client: Arc<NexarClient>, gpu_index: usize) -> Result<()> {
    let rank = client.rank();
    let assignment = SwarmWorker::receive_assignment(&client, LEADER_RANK).await?;
    tracing::info!(
        rank,
        layers = format!("{}..{}", assignment.start_layer, assignment.end_layer),
        embedding = assignment.has_embedding,
        lm_head = assignment.has_lm_head,
        "Received layer assignment"
    );
    let model_path = SwarmWorker::receive_model_path(&client, LEADER_RANK).await?;
    tracing::info!(rank, path = %model_path, "Received model path from leader");

    let device = crate::runtime::open_device(gpu_index)?;
    let (loaded_model, _blazr_config) =
        crate::loader::load_model::<ServerRuntime, _>(std::path::Path::new(&model_path), &device)?;
    let model = Arc::new(loaded_model);
    tracing::info!(
        rank,
        layers = model.num_layers(),
        vocab_size = model.vocab_size(),
        "Model loaded on worker"
    );

    // Stage r runs on rank r: the leader is stage 0, so every worker has a
    // previous stage, and the last rank has no next one.
    let prev_rank = Some(rank - 1);
    let next_rank = (rank + 1 < client.world_size()).then_some(rank + 1);
    let forward_fn = build_forward_fn(
        model,
        device,
        assignment,
        prev_rank.is_some(),
        next_rank.is_some(),
    )?;
    SwarmWorker::send_ready(&client, LEADER_RANK).await?;
    tracing::info!(rank, "Worker ready, entering compute loop");
    let worker = SwarmWorker::new(
        client,
        rank,
        assignment,
        prev_rank,
        next_rank,
        LEADER_RANK,
        forward_fn,
    );
    until_signal("swarm worker", worker.run_compute_loop()).await
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::cli::{Cli, Commands, SwarmRoleArg};

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("blazr").chain(args.iter().copied()))
    }

    #[test]
    fn a_worker_needs_the_leader_address() {
        let err = parse(&["swarm", "--role", "worker", "--token", "t"])
            .err()
            .expect("a worker without --leader is refused");
        assert!(err.to_string().contains("--leader"), "{err}");
        assert!(parse(&["swarm", "--token", "t", "--leader", "10.0.0.1:9090"]).is_ok());
    }

    #[test]
    fn the_leader_flags_parse() {
        let cli = parse(&[
            "swarm",
            "--role",
            "leader",
            "--token",
            "t",
            "--workers",
            "3",
            "--listen-addr",
            "10.0.0.1:9091",
            "--advertise-addr",
            "192.0.2.1:9091",
            "--join-timeout",
            "30",
        ])
        .unwrap();
        let Commands::Swarm(args) = cli.command else {
            panic!("not the swarm command");
        };
        assert_eq!(args.role, SwarmRoleArg::Leader);
        assert_eq!((args.workers, args.swarm_port), (3, 9090));
        assert_eq!(args.listen_addr, "10.0.0.1:9091");
        assert_eq!(args.mesh.advertise_addr.as_deref(), Some("192.0.2.1:9091"));
        assert_eq!(args.mesh.join_timeout, 30);
        assert!(parse(&["swarm", "--role", "leader", "--token", "t", "--mdns"]).is_err());
    }
}
