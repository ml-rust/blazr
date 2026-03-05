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
            top_k,
            min_p,
            repeat_penalty,
            repeat_last_n,
            gpu_layers,
            cpu,
            num_ctx,
            paged_attention,
            graphs,
        } => {
            let gen_config = blazr::config::GenerationConfig {
                max_tokens,
                temperature,
                top_p,
                top_k,
                min_p,
                repeat_penalty,
                repeat_last_n,
                ..Default::default()
            };
            blazr::cli::run(
                model,
                prompt,
                gen_config,
                gpu_layers,
                cpu,
                num_ctx,
                paged_attention,
                graphs,
            )
            .await?;
        }
        Commands::Serve {
            model,
            port,
            host,
            api_key,
        } => {
            blazr::cli::serve(model, port, host, api_key).await?;
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
            top_p,
            top_k,
            min_p,
            repeat_penalty,
            repeat_last_n,
            gpu_layers,
            cpu,
            num_ctx,
            paged_attention,
            graphs,
            verbose_prompt,
        } => {
            let gen_config = blazr::config::GenerationConfig {
                max_tokens,
                temperature,
                top_p,
                top_k,
                min_p,
                repeat_penalty,
                repeat_last_n,
                verbose_prompt,
                ..Default::default()
            };
            blazr::cli::run(
                model,
                Some(prompt),
                gen_config,
                gpu_layers,
                cpu,
                num_ctx,
                paged_attention,
                graphs,
            )
            .await?;
        }
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
