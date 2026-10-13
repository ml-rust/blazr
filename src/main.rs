use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use blazr::cli::{Cli, Commands, LoadSettings};
use blazr::config::{resolve, resolve_serve, EnvVars, FileConfig, GenerationFlags, ServeFlags};
#[cfg(feature = "distributed")]
use blazr::config::{resolve_cluster, ClusterFlags, ClusterSettings};

/// `swarm` and `disagg` settings: flags over the environment and the file.
#[cfg(feature = "distributed")]
fn cluster_settings(flags: ClusterFlags) -> Result<ClusterSettings> {
    let file = FileConfig::load()?;
    let env = EnvVars::from_process()?;
    resolve_cluster(flags, &env, &file).map_err(anyhow::Error::msg)
}

/// The model directory: `flag`, `BLAZR_MODEL_DIR`, `model_dir`, `./models`.
fn model_dir(flag: Option<std::path::PathBuf>) -> Result<std::path::PathBuf> {
    let file = FileConfig::load()?;
    let env = EnvVars::from_process()?;
    Ok(resolve::resolve_model_dir(flag, &env, &file))
}

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
    blazr::tls::install_crypto_provider();
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
            let file = FileConfig::load()?;
            let model = resolve::require_model(model, &file).map_err(anyhow::Error::msg)?;
            let gen_config =
                resolve::resolve_generation(&sampling.into_flags(verbose), &file.generation);
            blazr::cli::run(
                model,
                prompt,
                chat_template,
                gen_config,
                runtime.resolve(&file),
                Vec::new(),
            )
            .await?;
        }
        Commands::Serve {
            model,
            chat_template,
            port,
            host,
            data_parallel,
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
            dtype,
            decision,
            lora,
        } => {
            let file_path = FileConfig::path();
            let file = FileConfig::load()?;
            let env = EnvVars::from_process()?;
            let flags = ServeFlags {
                model,
                port,
                host,
                tls_cert,
                tls_key,
                api_key,
                reasoning_format,
                voice_dir,
                data_parallel,
                dtype: dtype.dtype,
                decision: decision.into_flags(),
            };
            let settings = resolve_serve(flags, &env, &file).map_err(anyhow::Error::msg)?;
            blazr::cli::serve(
                settings,
                file,
                file_path,
                chat_template,
                api_key_file,
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
                vision.into_overrides(),
                lora.lora,
            )
            .await?;
        }
        Commands::List { verbose } => {
            blazr::cli::list(verbose, model_dir(None)?).await?;
        }
        Commands::Info { model } => {
            blazr::cli::info(model, model_dir(None)?).await?;
        }
        Commands::Pull { repo, file, output } => {
            blazr::cli::pull(repo, file, model_dir(output)?).await?;
        }
        Commands::Generate {
            model,
            prompt,
            sampling,
            runtime,
            verbose_prompt,
            images,
        } => {
            let file = FileConfig::load()?;
            let model = resolve::require_model(model, &file).map_err(anyhow::Error::msg)?;
            let gen_config =
                resolve::resolve_generation(&sampling.into_flags(verbose_prompt), &file.generation);
            let runtime = runtime.resolve(&file);
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
            dtype,
        } => {
            let file = FileConfig::load()?;
            let model = resolve::require_model(model, &file).map_err(anyhow::Error::msg)?;
            let flags = GenerationFlags {
                max_tokens,
                temperature,
                top_p,
                verbose_prompt: verbose,
                ..Default::default()
            };
            let gen_config = resolve::resolve_generation(&flags, &file.generation);
            let gpu_index = blazr::cli::build_gpu_index(&file)?;
            let load = LoadSettings::resolve(num_ctx, vision.into_overrides(), dtype.dtype, &file);
            blazr::cli::chat(model, system, chat_template, gen_config, gpu_index, load).await?;
        }
        Commands::Bench {
            model,
            num_ctx,
            decode_tokens,
            runs,
            json,
            concurrency,
            dtype,
        } => {
            let file = FileConfig::load()?;
            let model = resolve::require_model(model, &file).map_err(anyhow::Error::msg)?;
            let gpu_index = blazr::cli::build_gpu_index(&file)?;
            let load = LoadSettings::resolve(num_ctx, Default::default(), dtype.dtype, &file);
            blazr::cli::bench(
                model,
                load,
                gpu_index,
                decode_tokens,
                runs,
                json,
                concurrency,
            )
            .await?;
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
            host,
            api_key,
            mdns,
        } => {
            let settings = cluster_settings(ClusterFlags {
                model,
                port,
                host,
                api_key,
            })?;
            blazr::cli::swarm(role, token, leader, swarm_port, mdns, settings).await?;
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
            host,
        } => {
            let settings = cluster_settings(ClusterFlags {
                model,
                port,
                host,
                api_key: None,
            })?;
            match role.as_str() {
                "router" => {
                    let prefill_addrs = prefill.unwrap_or_default();
                    let decode_addrs = decode.unwrap_or_default();
                    blazr::cli::run_disagg_router(
                        listen_addr,
                        prefill_addrs,
                        decode_addrs,
                        settings.server,
                    )
                    .await?;
                }
                "prefill" => {
                    let raddr = router_addr.ok_or_else(|| {
                        anyhow::anyhow!("--router-addr is required for disagg prefill role")
                    })?;
                    blazr::cli::run_disagg_prefill_worker(raddr, listen_addr, &settings).await?;
                }
                "decode" => {
                    let raddr = router_addr.ok_or_else(|| {
                        anyhow::anyhow!("--router-addr is required for disagg decode role")
                    })?;
                    blazr::cli::run_disagg_decode_worker(raddr, listen_addr, &settings).await?;
                }
                _ => {
                    anyhow::bail!(
                        "Invalid disagg role '{}'. Must be 'router', 'prefill', or 'decode'",
                        role
                    );
                }
            }
        }
        Commands::Decision(command) => blazr::cli::decision(command).await?,
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
    }

    Ok(())
}
