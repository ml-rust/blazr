use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use blazr::cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "blazr=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model,
            prompt,
            verbose,
            sampling,
            runtime,
        } => {
            let mut gen_config = sampling.into_gen_config();
            gen_config.verbose_prompt = verbose;
            blazr::cli::run(
                model,
                prompt,
                gen_config,
                runtime.gpu_layers,
                runtime.cpu,
                runtime.num_ctx,
                runtime.paged_attention,
                runtime.graphs,
            )
            .await?;
        }
        Commands::Serve {
            model,
            port,
            host,
            api_key,
            api_key_file,
            tls_cert,
            tls_key,
        } => {
            blazr::cli::serve(model, port, host, api_key, api_key_file, tls_cert, tls_key).await?;
        }
        Commands::List { verbose } => {
            blazr::cli::list(verbose).await?;
        }
        Commands::Info { model } => {
            blazr::cli::info(model).await?;
        }
        Commands::Pull { repo, file, output } => {
            blazr::cli::pull(repo, file, output).await?;
        }
        Commands::Generate {
            model,
            prompt,
            sampling,
            runtime,
            verbose_prompt,
        } => {
            let mut gen_config = sampling.into_gen_config();
            gen_config.verbose_prompt = verbose_prompt;
            blazr::cli::run(
                model,
                Some(prompt),
                gen_config,
                runtime.gpu_layers,
                runtime.cpu,
                runtime.num_ctx,
                runtime.paged_attention,
                runtime.graphs,
            )
            .await?;
        }
        Commands::Chat {
            model,
            system,
            max_tokens,
            temperature,
            top_p,
            num_ctx,
            verbose,
        } => {
            blazr::cli::chat(
                model,
                system,
                max_tokens,
                temperature,
                top_p,
                num_ctx,
                verbose,
            )
            .await?;
        }
        Commands::Bench {
            model,
            num_ctx,
            decode_tokens,
            runs,
            json,
            concurrency,
        } => {
            blazr::cli::bench(model, num_ctx, decode_tokens, runs, json, concurrency).await?;
        }
        Commands::Ps { server } => {
            blazr::cli::ps(server).await?;
        }
        Commands::Convert {
            input,
            output,
            format,
            quantization,
            verbose,
        } => {
            blazr::cli::convert(input, output, format, quantization, verbose)?;
        }
        #[cfg(feature = "distributed")]
        Commands::Swarm {
            role,
            token,
            model,
            leader,
            swarm_port,
            port,
            mdns,
        } => {
            blazr::cli::swarm(role, token, model, leader, swarm_port, port, mdns).await?;
        }
        #[cfg(feature = "distributed")]
        Commands::Disagg {
            role,
            model,
            listen_addr,
            router_addr,
            prefill,
            decode,
            port,
        } => match role.as_str() {
            "router" => {
                let prefill_addrs = prefill.unwrap_or_default();
                let decode_addrs = decode.unwrap_or_default();
                blazr::cli::run_disagg_router(listen_addr, prefill_addrs, decode_addrs, port)
                    .await?;
            }
            "prefill" => {
                let model_name = model.ok_or_else(|| {
                    anyhow::anyhow!("--model is required for disagg prefill role")
                })?;
                let raddr = router_addr.ok_or_else(|| {
                    anyhow::anyhow!("--router-addr is required for disagg prefill role")
                })?;
                blazr::cli::run_disagg_prefill_worker(model_name, raddr, listen_addr).await?;
            }
            "decode" => {
                let model_name = model
                    .ok_or_else(|| anyhow::anyhow!("--model is required for disagg decode role"))?;
                let raddr = router_addr.ok_or_else(|| {
                    anyhow::anyhow!("--router-addr is required for disagg decode role")
                })?;
                blazr::cli::run_disagg_decode_worker(model_name, raddr, listen_addr).await?;
            }
            _ => {
                anyhow::bail!(
                    "Invalid disagg role '{}'. Must be 'router', 'prefill', or 'decode'",
                    role
                );
            }
        },
        Commands::Completions { shell } => {
            clap_complete::generate(
                shell,
                &mut <Cli as clap::CommandFactory>::command(),
                "blazr",
                &mut std::io::stdout(),
            );
        }
        Commands::Decode { model: _, input: _ } => {
            eprintln!("Decode command not yet implemented");
        }
    }

    Ok(())
}
