//! Swarm mode CLI command
//!
//! Starts the leader or joins a worker of a pipeline swarm, one process per
//! node. The leader binds nexar's seed and holds rank 0. Workers take ranks
//! in arrival order: the leader's schedule runs stage `r` on rank `r`, so a
//! worker's previous and next stage follow from its rank.
//!
//! The leader runs stage 0 and serves the HTTP API. Every generation of the
//! start model runs through the stages, and the leader samples each token.
//! Each node reads only its own layers' tensors and runs only them. The
//! leader's HTTP front end holds the config and tokenizer and no weights.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use colored::Colorize;
use nexar::NexarClient;

use super::disaggregated_router::router_executor;
use super::mesh::{
    after_shutdown, formation_error, parse_addr, until_signal, MeshSettings, RELEASE_WITHIN,
};
use super::swarm_scheduler::{swarm_context, SwarmScheduler};
use super::swarm_serving::swarm_serving;
use super::swarm_stage::{
    build_stage, leader_assignment, load_stage_model, stage_config, LocalStage,
};
use crate::cli::{SwarmArgs, SwarmRoleArg};
use crate::config::{ClusterSettings, ServerConfig};
use crate::distributed::leader::SwarmLeader;
use crate::distributed::stage::Stage;
use crate::distributed::transport::{self, LayerAssignment};
use crate::distributed::worker::SwarmWorker;
use crate::engine::{Executor, Scheduler};
use crate::runtime::ServerRuntime;
use crate::server::{start_with_batch, LiveSettings};

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
            let max_requests = args.max_requests;
            run_leader(&model, seed, &mesh, world_size, max_requests, settings).await
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
            run_worker(
                leader,
                &mesh,
                world_size,
                args.max_requests,
                settings.gpu_index,
            )
            .await
        }
    }
}

/// Leader: load the model's config and its own stage's layers, form the
/// cluster, assign layers, send the model path, serve HTTP through the
/// pipeline, then stop every worker and shut down.
async fn run_leader(
    model_name: &str,
    seed: SocketAddr,
    mesh: &MeshSettings,
    world_size: u32,
    max_requests: usize,
    settings: ClusterSettings,
) -> Result<()> {
    let ClusterSettings {
        model_dir,
        gpu_index,
        server: server_config,
        api_key,
        ..
    } = settings;
    let api_keys: Vec<String> = api_key.into_iter().collect();
    let device = crate::runtime::open_device(gpu_index)?;
    let models = Scheduler::<ServerRuntime>::new(model_dir.clone(), device.clone())
        .with_operator_model(Some(model_name.to_string()));
    let model_path = models.model_path(model_name).ok_or_else(|| {
        anyhow!(
            "model '{model_name}' has no path under {}: the workers load it from there",
            model_dir.display()
        )
    })?;
    if world_size == 1 {
        tracing::info!("No workers: the leader serves alone");
        models.get_executor(model_name).await?;
        return crate::server::start(Arc::new(models), server_config, api_keys).await;
    }
    // The front end: config, tokenizer and chat template. The stages hold the weights.
    let executor = router_executor(&model_path, device.clone())?;
    let total_layers = executor.config().num_layers();
    let vocab_size = executor.vocab_size();
    // Refuse a model or a swarm the pipeline cannot run before any worker joins.
    let assignment = leader_assignment(world_size, total_layers)?;
    let stage_cfg = stage_config(
        max_requests,
        swarm_context(&executor),
        executor.config().kv_cache_dtype()?,
    );
    let (model, _) = load_stage_model(&model_path, &device, &assignment)?;
    let (start, end) = (assignment.start_layer, assignment.end_layer);
    tracing::info!(
        model = model_name,
        layers = format!("{start}..{end} of {total_layers}"),
        "Stage loaded on leader"
    );
    let stage = build_stage(model, device, assignment, stage_cfg)?;
    let models = models.pinned();
    let executor = models
        .register_base(model_name, model_path.clone(), executor)
        .await;
    mesh.check_seed_reachable(seed, "the seed", world_size)?;

    let cfg = mesh.cluster_config(world_size, Some(LEADER_RANK));
    tracing::info!(%seed, world_size, "Leader waiting for workers");
    let client = SwarmLeader::form_cluster(seed, &cfg)
        .await
        .map_err(|e| formation_error("swarm leader formation", e, TOKEN_FLAG))?;
    let leader = Arc::new(SwarmLeader::by_rank(
        Arc::new(client),
        total_layers,
        vocab_size,
    )?);
    for stage in leader.schedule().stages() {
        tracing::info!(
            node = %stage.node_id,
            layers = format!("{}..{}", stage.start_layer, stage.end_layer),
            "Pipeline stage"
        );
    }
    let swarm = SwarmServe {
        leader: Arc::clone(&leader),
        stage,
        assignment,
        executor,
        max_requests,
        model_path,
    };
    let run = swarm.run(models, server_config, api_keys).await;
    let shutdown = match Arc::try_unwrap(leader) {
        Ok(leader) => leader.shutdown(RELEASE_WITHIN).await,
        Err(shared) => {
            shared.shutdown_workers().await?;
            Err(anyhow!(
                "the swarm leader was still shared after its requests ended, so its nexar \
                 client did not shut down"
            ))
        }
    };
    after_shutdown("swarm leader", run, shutdown)
}

/// What the leader serves once the cluster formed.
struct SwarmServe {
    leader: Arc<SwarmLeader>,
    stage: Box<dyn Stage>,
    assignment: LayerAssignment,
    executor: Arc<Executor<ServerRuntime>>,
    max_requests: usize,
    model_path: PathBuf,
}

impl SwarmServe {
    /// Set up the workers, serve HTTP until the server stops, then end
    /// every request. On return no task holds the leader. `models` is
    /// pinned to the executor the swarm serves.
    async fn run(
        self,
        models: Scheduler<ServerRuntime>,
        server: ServerConfig,
        api_keys: Vec<String>,
    ) -> Result<()> {
        let Self {
            leader,
            stage,
            assignment,
            executor,
            max_requests,
            model_path,
        } = self;
        if leader.local_assignment() != Some(assignment) {
            bail!("the swarm leader's schedule does not give it stage 0");
        }
        let path = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path {} is not UTF-8", model_path.display()))?;
        leader.assign_layers().await?;
        leader.send_model_path(path).await?;
        leader.wait_for_workers().await?;
        tracing::info!("All workers ready, starting HTTP server");
        let local = LocalStage::spawn(stage)?;
        let swarm = SwarmScheduler::new(leader, Some(local), Arc::clone(&executor), max_requests)?;
        let serving = swarm_serving(executor, Arc::clone(&swarm));
        let live = LiveSettings::fixed(&server);
        let served = start_with_batch(
            Arc::new(models),
            server,
            live,
            api_keys,
            serving,
            Vec::new(),
            #[cfg(feature = "audio")]
            Vec::new(),
            #[cfg(feature = "audio")]
            Vec::new(),
            None,
        )
        .await;
        swarm.close(RELEASE_WITHIN).await;
        served
    }
}

/// Worker: join the leader's cluster, run its stage until the leader stops
/// it or a signal arrives, then shut down.
async fn run_worker(
    leader: SocketAddr,
    mesh: &MeshSettings,
    world_size: u32,
    max_requests: usize,
    gpu_index: usize,
) -> Result<()> {
    tracing::info!(%leader, "Worker joining the swarm");
    let cfg = mesh.cluster_config(world_size, None);
    let client = NexarClient::bootstrap_join(leader, &cfg, transport::cpu_adapter())
        .await
        .map_err(|e| formation_error("swarm worker join", e, TOKEN_FLAG))?;
    let client = Arc::new(client);
    tracing::info!(rank = client.rank(), world_size, "Joined the swarm");
    let run = run_stage(Arc::clone(&client), max_requests, gpu_index).await;
    let shutdown = transport::shutdown_shared(client, RELEASE_WITHIN).await;
    eprintln!("\n  Worker shut down.");
    after_shutdown("swarm worker", run, shutdown)
}

/// Receive this worker's stage and model, load it, and run the compute loop.
///
/// The worker reads only its layer range's tensors and runs only them.
async fn run_stage(client: Arc<NexarClient>, max_requests: usize, gpu_index: usize) -> Result<()> {
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
    let (model, blazr_config) =
        load_stage_model(std::path::Path::new(&model_path), &device, &assignment)?;
    let (start, end, total) = (
        assignment.start_layer,
        assignment.end_layer,
        model.num_layers(),
    );
    tracing::info!(
        rank,
        layers = format!("{start}..{end} of {total}"),
        vocab_size = model.vocab_size(),
        "Stage loaded on worker"
    );
    // The model's own maximum: the leader's context never passes it.
    let config = stage_config(
        max_requests,
        model.max_seq_len(),
        blazr_config.kv_cache_dtype()?,
    );
    let stage = build_stage(model, device, assignment, config)?;

    // Stage r runs on rank r: the leader is stage 0, so every worker has a
    // previous stage, and the last rank has no next one.
    let prev_rank = Some(rank - 1);
    let next_rank = (rank + 1 < client.world_size()).then_some(rank + 1);
    SwarmWorker::send_ready(&client, LEADER_RANK).await?;
    tracing::info!(rank, "Worker ready, entering compute loop");
    let mut worker = SwarmWorker::new(
        client,
        rank,
        assignment,
        prev_rank,
        next_rank,
        LEADER_RANK,
        stage,
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
            "--max-requests",
            "8",
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
        assert_eq!(args.max_requests, 8);
        assert!(parse(&["swarm", "--role", "leader", "--token", "t", "--mdns"]).is_err());
    }
}
