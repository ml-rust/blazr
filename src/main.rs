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
        } => {
            blazr::cli::bench(model, num_ctx, decode_tokens, runs).await?;
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
