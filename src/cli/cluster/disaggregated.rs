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

use anyhow::{anyhow, Result};
use colored::Colorize;

use super::disaggregated_forward::{build_decode_open_fn, build_prefill_fn, check_disagg_model};
use super::disaggregated_layout::DisaggLayout;
use super::disaggregated_mesh::{
    bind_seed, role_name, router_client, serve_decode, serve_prefill, start_router, stop_router,
    worker_client,
};
use super::disaggregated_router::{router_executor, router_serving};
use super::mesh::{after_shutdown, parse_addr, MeshSettings};
use crate::cli::DisaggArgs;
use crate::config::ClusterSettings;
use crate::distributed::disaggregated::DisaggRole;
use crate::engine::{Executor, Scheduler};
use crate::runtime::ServerRuntime;
use crate::server::{start_with_batch, LiveSettings};

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

/// Run one `disagg` node: the router, a prefill worker or a decode worker.
///
/// # Errors
///
/// See [`run_disagg_router`] and [`run_disagg_worker`].
pub async fn run_disagg(args: DisaggArgs, settings: &ClusterSettings) -> Result<()> {
    let layout =
        DisaggLayout::resolve(args.role, args.index, args.prefill_count, args.decode_count)?;
    match layout.role {
        DisaggRole::Router => run_disagg_router(&args, &layout, settings).await,
        DisaggRole::Prefill | DisaggRole::Decode => {
            run_disagg_worker(&args, &layout, settings).await
        }
    }
}

/// Run a disaggregated router.
///
/// The router loads the config, tokenizer and chat template of the model
/// `settings` names, and no weights. It binds the seed on `--router-addr`,
/// waits until every worker joins, and serves the whole HTTP API that
/// `blazr serve` serves. Each generation runs on a prefill and a decode
/// worker. `settings.server` holds the HTTP host and port. When the server
/// stops, the router stops every worker and shuts down its nexar client.
///
/// # Errors
///
/// No `--model`, a model whose tokenizer does not load, a bad address, a
/// failed nexar formation, or a server that does not start.
async fn run_disagg_router(
    args: &DisaggArgs,
    layout: &DisaggLayout,
    settings: &ClusterSettings,
) -> Result<()> {
    let model = settings.router_model().map_err(anyhow::Error::msg)?;
    let router_addr = parse_addr("--router-addr", &args.router_addr)?;
    let mesh = MeshSettings::parse(&args.listen_addr, &args.mesh, args.cluster_token.clone())?;
    mesh.check_seed_reachable(router_addr, "--router-addr", layout.world_size())?;
    banner(
        "router",
        &[
            ("Model", model.clone()),
            ("Seed", router_addr.to_string()),
            ("Mesh", mesh.listen.to_string()),
            ("Prefill workers", args.prefill_count.to_string()),
            ("Decode workers", args.decode_count.to_string()),
            ("HTTP API", settings.server.addr()),
        ],
    );

    let device = crate::runtime::open_device(settings.gpu_index)?;
    let models = Scheduler::<ServerRuntime>::new(settings.model_dir.clone(), device.clone())
        .with_operator_model(Some(model.clone()));
    let path = models.model_path(&model).ok_or_else(|| {
        anyhow!(
            "model '{model}' not found as a path or under {}: the router reads its \
                 tokenizer and chat template",
            settings.model_dir.display()
        )
    })?;
    let executor = router_executor(&path, device)?;

    let cfg = mesh.cluster_config(layout.world_size(), Some(layout.rank));
    let seed = bind_seed(router_addr, &cfg)?;
    tracing::info!(seed = %seed.local_addr(), world_size = layout.world_size(), "Router waiting for workers");
    let (router, client) = start_router(router_client(seed, &cfg).await?, layout)?;
    tracing::info!(
        prefill_count = args.prefill_count,
        decode_count = args.decode_count,
        "Disaggregated router ready"
    );

    let (scheduler, serving) = router_serving(
        Arc::clone(&router),
        client.failure_watch(),
        &model,
        path,
        executor,
        models,
    )
    .await;
    let server = settings.server.clone();
    let live = LiveSettings::fixed(&server);
    let run = start_with_batch(
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
    .await;
    let shutdown = stop_router(router, client).await;
    after_shutdown("router", run, shutdown)
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
    let loader = Scheduler::<ServerRuntime>::new(settings.model_dir.clone(), device)
        .with_operator_model(Some(model.clone()));
    let executor = loader.get_executor(&model).await?;
    check_disagg_model(&executor, role)?;
    tracing::info!(
        model = %model,
        vocab_size = executor.vocab_size(),
        "Model loaded on {role} worker"
    );
    Ok(executor)
}

/// Run a disaggregated prefill or decode worker.
///
/// It loads the model `settings` names, joins the router's cluster with
/// the rank of its role and `--index`, and serves until the router stops it
/// or a signal arrives. Then it shuts down its nexar client.
///
/// # Errors
///
/// No model, a model the KV handoff cannot carry, a bad address, or a failed
/// nexar formation.
async fn run_disagg_worker(
    args: &DisaggArgs,
    layout: &DisaggLayout,
    settings: &ClusterSettings,
) -> Result<()> {
    let role = role_name(layout.role);
    let router_addr = parse_addr("--router-addr", &args.router_addr)?;
    let mesh = MeshSettings::parse(&args.listen_addr, &args.mesh, args.cluster_token.clone())?;
    banner(
        role,
        &[
            ("Model", settings.model.clone().unwrap_or_default()),
            ("Router", router_addr.to_string()),
            ("Mesh", mesh.listen.to_string()),
            ("Rank", layout.rank.to_string()),
        ],
    );
    let executor = worker_executor(role, settings).await?;
    let cfg = mesh.cluster_config(layout.world_size(), Some(layout.rank));
    let client = worker_client(layout, router_addr, &cfg).await?;
    let sampler = <ServerRuntime as boostr::Runtime>::default_client(executor.device());
    let result = if layout.role == DisaggRole::Prefill {
        serve_prefill(client, layout, sampler, build_prefill_fn(executor)).await
    } else {
        serve_decode(client, layout, sampler, build_decode_open_fn(executor)).await
    };
    eprintln!("\n  {role} worker shut down.");
    result
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
        let err = parse(&["disagg", "--role", "router", "--router-addr", "a:1"])
            .err()
            .expect("a router without --model is refused");
        assert!(err.to_string().contains("--model"), "{err}");
        let ok = &[
            "disagg",
            "--role",
            "router",
            "--model",
            "m",
            "--router-addr",
            "a:1",
        ];
        assert!(parse(ok).is_ok());
    }

    #[test]
    fn a_worker_role_takes_its_model_from_the_config() {
        assert!(parse(&["disagg", "--role", "prefill", "--router-addr", "a:1"]).is_ok());
        assert!(parse(&["disagg", "--role", "decode", "--router-addr", "a:1"]).is_ok());
    }

    #[test]
    fn every_role_needs_the_router_address_and_an_unknown_role_is_refused() {
        let err = parse(&["disagg", "--role", "prefill"])
            .err()
            .expect("no --router-addr");
        assert!(err.to_string().contains("--router-addr"), "{err}");
        assert!(parse(&["disagg", "--role", "bogus", "--router-addr", "a:1"]).is_err());
    }

    #[test]
    fn the_counts_index_and_token_parse() {
        let cli = parse(&[
            "disagg",
            "--role",
            "decode",
            "--router-addr",
            "10.0.0.1:9190",
            "--prefill-count",
            "2",
            "--decode-count",
            "3",
            "--index",
            "1",
            "--cluster-token",
            "s3cret",
            "--advertise-addr",
            "192.0.2.4:9191",
        ])
        .unwrap();
        let crate::cli::Commands::Disagg(args) = cli.command else {
            panic!("not the disagg command");
        };
        assert_eq!(
            (args.prefill_count, args.decode_count, args.index),
            (2, 3, Some(1))
        );
        assert_eq!(args.cluster_token.as_deref(), Some("s3cret"));
        assert_eq!(args.mesh.advertise_addr.as_deref(), Some("192.0.2.4:9191"));
        assert_eq!(args.mesh.join_timeout, 600);
        assert_eq!(args.listen_addr, "0.0.0.0:9191");
    }
}
