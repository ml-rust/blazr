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
            max_tokens,
            temperature,
            top_p,
            gpu_layers,
            cpu,
            num_ctx,
        } => {
            blazr::cli::run(
                model,
                prompt,
                max_tokens,
                temperature,
                top_p,
                gpu_layers,
                cpu,
                num_ctx,
            )
            .await?;
        }
        Commands::Serve { model, port, host } => {
            blazr::cli::serve(model, port, host).await?;
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
            max_tokens,
            temperature,
            gpu_layers,
            cpu,
            num_ctx,
        } => {
            blazr::cli::run(
                model,
                Some(prompt),
                max_tokens,
                temperature,
                1.0,
                gpu_layers,
                cpu,
                num_ctx,
            )
            .await?;
        }
        Commands::Decode { model: _, input: _ } => {
            eprintln!("Decode command not yet implemented");
        }
    }

    Ok(())
}
