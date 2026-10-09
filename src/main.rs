use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use blazr::cli::{Cli, Commands};

/// Whether `command` logs to standard error: the narration commands print
/// their report on standard output.
fn logs_to_stderr(command: &Commands) -> bool {
    #[cfg(feature = "audio")]
    if matches!(
        command,
        Commands::Speak(_) | Commands::Transcribe(_) | Commands::SoundReview(_)
    ) {
        return true;
    }
    let _ = command;
    false
}

/// The arguments after the narration command `command`, as given. clap
/// drops a leading `--`, which the skill reads as an unknown option, so
/// the command reads the process's own arguments.
#[cfg(feature = "audio")]
fn narration_argv(command: &str) -> Vec<String> {
    let args: Vec<String> = std::env::args_os()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    args.iter()
        .skip(1)
        .position(|a| a == command)
        .map(|i| args[i + 2..].to_vec())
        .unwrap_or_default()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging. The narration commands keep standard output for
    // their report, so their logs go to standard error.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "blazr=info,tower_http=debug".into());
    if logs_to_stderr(&cli.command) {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    match cli.command {
        Commands::Run {
            model,
            prompt,
            chat_template,
            verbose,
            sampling,
            runtime,
        } => {
            let mut gen_config = sampling.into_gen_config();
            gen_config.verbose_prompt = verbose;
            blazr::cli::run(
                model,
                prompt,
                chat_template,
                gen_config,
                runtime,
                Vec::new(),
            )
            .await?;
        }
        Commands::Serve {
            model,
            chat_template,
            port,
            host,
            api_key,
            api_key_file,
            tls_cert,
            tls_key,
            vision_models,
            #[cfg(feature = "audio")]
            asr_models,
            #[cfg(feature = "audio")]
            tts_models,
            #[cfg(feature = "audio")]
            tts_max_batch,
            #[cfg(feature = "audio")]
            tts_batch_window_ms,
            #[cfg(feature = "audio")]
            tts_no_continuous_batching,
            #[cfg(feature = "audio")]
            tts_continuous_capacity,
            voice_dir,
            reasoning_format,
            vision,
        } => {
            blazr::cli::serve(
                model,
                chat_template,
                port,
                host,
                api_key,
                api_key_file,
                tls_cert,
                tls_key,
                vision_models,
                #[cfg(feature = "audio")]
                asr_models,
                #[cfg(feature = "audio")]
                tts_models,
                #[cfg(feature = "audio")]
                tts_max_batch,
                #[cfg(feature = "audio")]
                tts_batch_window_ms,
                #[cfg(feature = "audio")]
                !tts_no_continuous_batching,
                #[cfg(feature = "audio")]
                tts_continuous_capacity,
                voice_dir,
                reasoning_format,
                vision.into_overrides(),
            )
            .await?;
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
            images,
        } => {
            let mut gen_config = sampling.into_gen_config();
            gen_config.verbose_prompt = verbose_prompt;
            blazr::cli::run(model, Some(prompt), None, gen_config, runtime, images).await?;
        }
        Commands::Chat {
            model,
            system,
            chat_template,
            max_tokens,
            temperature,
            top_p,
            num_ctx,
            verbose,
            vision,
        } => {
            blazr::cli::chat(
                model,
                system,
                chat_template,
                max_tokens,
                temperature,
                top_p,
                num_ctx,
                verbose,
                vision.into_overrides(),
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
        #[cfg(feature = "audio")]
        Commands::Speak(_) => std::process::exit(blazr::cli::speak(&narration_argv("speak"))),
        #[cfg(feature = "audio")]
        Commands::Transcribe(_) => {
            std::process::exit(blazr::cli::transcribe(&narration_argv("transcribe")))
        }
        #[cfg(feature = "audio")]
        Commands::SoundReview(_) => {
            std::process::exit(blazr::cli::sound_review(&narration_argv("sound-review")))
        }
        Commands::Decode { model: _, input: _ } => {
            eprintln!("Decode command not yet implemented");
        }
    }

    Ok(())
}
