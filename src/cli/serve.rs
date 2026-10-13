//! HTTP server command

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use colored::Colorize;

use crate::config::{FileConfig, LiveConfig, ServeSettings, VisionOverrides};
use crate::server::handlers::Serving;
use crate::server::{self, LiveSettings};

use super::serve_replicas::{load_serving, SchedulerSettings};
use crate::runtime::ServerRuntime;

/// Start the inference server.
///
/// `settings` holds every value resolved from flags, environment and
/// `file`. `file_path` is the file the server hot-reloads.
#[allow(clippy::too_many_arguments)]
pub async fn serve(
    settings: ServeSettings,
    file: FileConfig,
    file_path: Option<PathBuf>,
    chat_template: Option<String>,
    api_key_file: Option<PathBuf>,
    vision_models: Vec<String>,
    #[cfg(feature = "audio")] asr_models: Vec<String>,
    #[cfg(feature = "audio")] tts_models: Vec<String>,
    #[cfg(feature = "audio")] tts_max_batch: usize,
    #[cfg(feature = "audio")] tts_batch_window_ms: u64,
    #[cfg(feature = "audio")] tts_continuous_batching: bool,
    #[cfg(feature = "audio")] tts_continuous_capacity: usize,
    vision: VisionOverrides,
    lora: Vec<String>,
) -> Result<()> {
    // `--lora`, else `inference.lora`: variants of the start model.
    let adapters = crate::engine::lora::configured_adapters(&lora, file.inference.value("lora"))?;
    let ServeSettings {
        model,
        model_dir,
        gpu_index,
        num_ctx,
        tensor_parallel_size,
        data_parallel,
        server: server_config,
        reasoning_format_flag,
        api_key,
        voice_dir,
        dtype,
        decision,
    } = settings;

    // Adapter variants need a start model, and are not replicated.
    if !adapters.is_empty() {
        let Some(start) = model.as_deref() else {
            anyhow::bail!(
                "--lora and inference.lora make variants of the start model. Pass --model or \
                 set default_model"
            );
        };
        if data_parallel > 1 {
            return Err(crate::engine::LoraError::data_parallel(start, data_parallel).into());
        }
    }

    // Check the tensor parallel degree before any model loads.
    if let Some(tp) = tensor_parallel_size {
        crate::engine::tensor_parallel::validate_tp_config(tp, None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    if file.inference.graphs() == Some(true) {
        tracing::warn!(
            "inference.graphs applies to `blazr run` and `generate`. blazr serve decodes without \
             graph capture"
        );
    }

    // The build's runtime fixes the device type. A wgpu adapter opens here.
    let device = crate::runtime::open_device(gpu_index)?;

    // The model scheduler. Data-parallel replicas build theirs the same way.
    let scheduler_settings = SchedulerSettings {
        model_dir,
        num_ctx,
        chat_template,
        inference: file.inference.clone(),
        tensor_parallel_size,
        vision,
        dtype,
        decision,
    };
    let scheduler = Arc::new(scheduler_settings.build(device.clone()));
    // The SLO checks and the batch policy start before any batching engine.
    let slo_policy = server_config.slo.clone().and_then(server::slo::install);
    let live = LiveSettings {
        config: Arc::new(LiveConfig::new(file)),
        path: file_path,
        reasoning_format_flag,
    };

    // Pre-load and warm up model if specified
    if let Some(ref model_name) = model {
        let spinner = super::util::spinner(format!("Loading model '{}'...", model_name));

        // Snapshot VRAM before loading (CUDA only)
        #[cfg(feature = "cuda")]
        let vram_before = device.memory_info().ok().map(|(free, _)| free);

        let load_start = std::time::Instant::now();
        let executor = scheduler.get_executor(model_name).await?;
        let load_time = load_start.elapsed();
        if let Err(e) = executor.warmup() {
            tracing::warn!("Model warmup failed (first request may be slower): {}", e);
        }

        spinner.finish_and_clear();

        // Calculate VRAM usage
        #[cfg(feature = "cuda")]
        let vram_str = {
            let vram_info = device.memory_info().ok();
            match (vram_before, vram_info) {
                (Some(free_before), Some((free_after, total))) => {
                    let used = free_before.saturating_sub(free_after);
                    format!(
                        ", VRAM: {:.1}/{:.1} GB",
                        used as f64 / (1024.0 * 1024.0 * 1024.0),
                        total as f64 / (1024.0 * 1024.0 * 1024.0),
                    )
                }
                _ => String::new(),
            }
        };
        #[cfg(not(feature = "cuda"))]
        let vram_str = String::new();

        eprintln!(
            "  {} {} in {:.1}s (vocab={}, ctx={}, template={}{})",
            "Loaded".green(),
            model_name.bold(),
            load_time.as_secs_f64(),
            executor.vocab_size(),
            executor.config().max_seq_len(),
            executor.chat_template().describe(),
            vram_str,
        );
        if let Some(tower) = executor.vision_tower() {
            eprintln!(
                "  {} vision tower (image tokens {}..={})",
                "Loaded".green(),
                tower.config().image_min_tokens,
                tower.config().image_max_tokens,
            );
        }
        // Each adapter becomes the variant `<model>:<name>` over the loaded
        // weights, with its own executor and KV cache.
        for spec in adapters {
            let info = scheduler.register_variant(model_name, spec).await?;
            eprintln!(
                "  {} LoRA variant {} (rank {}, {} modules, {})",
                "Loaded".green(),
                info.id.bold(),
                info.rank,
                info.adapted_modules,
                info.path.display(),
            );
        }
        // A decision model starts its worker now, so a contract mismatch
        // shows at start and the first request pays no renderer load.
        match scheduler.decision_service(model_name).await {
            Ok(service) => eprintln!(
                "  {} {} decision model on POST /v1/systemone ({})",
                "Loaded".green(),
                service.family().name(),
                match service.calibration() {
                    Some(c) => format!("temperatures from {}", c.source),
                    None => "uncalibrated".to_string(),
                },
            ),
            Err(crate::engine::decision::DecisionFailure::NotDecisionModel(_)) => {}
            Err(e) => tracing::warn!("POST /v1/systemone cannot serve '{model_name}': {e}"),
        }
    }

    // Collect API keys from --api-key and --api-key-file
    let mut api_keys: Vec<String> = Vec::new();
    if let Some(key) = api_key {
        api_keys.push(key);
    }
    if let Some(ref path) = api_key_file {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read API key file {:?}: {}", path, e))?;
        for line in content.lines() {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with('#') {
                api_keys.push(line.to_string());
            }
        }
        tracing::info!("Loaded {} API key(s) from {:?}", api_keys.len(), path);
    }

    // Startup banner
    let addr = server_config.addr();
    eprintln!();
    eprintln!("  {} v{}", "blazr".bold().cyan(), env!("CARGO_PKG_VERSION"));
    let scheme = if server_config.tls_enabled() {
        "https"
    } else {
        "http"
    };
    eprintln!(
        "  Listening on {}",
        format!("{}://{}", scheme, addr).underline()
    );
    if let Some(ref m) = model {
        eprintln!("  Model: {}", m.bold());
    }
    match crate::runtime::GpuBackend::of_build() {
        Some(crate::runtime::GpuBackend::Cuda) => eprintln!("  Backend: {}", "CUDA".green()),
        Some(crate::runtime::GpuBackend::Wgpu) => eprintln!("  Backend: {}", "wgpu".green()),
        None => eprintln!("  Backend: {}", "CPU".yellow()),
    }
    if !api_keys.is_empty() {
        eprintln!("  Auth: {} API key(s)", api_keys.len());
    }
    if server_config.cors_enabled {
        eprintln!("  CORS: {}", "enabled".yellow());
    }
    eprintln!();

    // The start model's copies, each with the batching scheduler its decode
    // path calls for and that scheduler's driver task running.
    let serving = match model {
        Some(ref model_name) => {
            load_serving(
                &scheduler,
                &scheduler_settings,
                model_name,
                gpu_index,
                data_parallel,
                slo_policy,
            )
            .await?
        }
        None => Serving::OnDemand,
    };

    // Load any standalone vision embedders before serving
    let mut loaded_vision: Vec<(
        String,
        Arc<crate::server::handlers::VisionEmbedder>,
        PathBuf,
    )> = Vec::with_capacity(vision_models.len());
    for arg in &vision_models {
        let (name, path) = crate::loader::parse_vision_model_arg(arg)?;
        let spinner = super::util::spinner(format!(
            "Loading vision embedder '{}' from {}...",
            name,
            path.display()
        ));
        let embedder =
            crate::loader::load_vision_embedder_from_dir::<ServerRuntime, _>(&path, &device)?;
        spinner.finish_and_clear();
        eprintln!(
            "  {} vision embedder {} (hidden={}, image={})",
            "Loaded".green(),
            name.bold(),
            embedder.hidden_size(),
            path.display(),
        );
        loaded_vision.push((name, Arc::new(embedder), path));
    }

    // ASR and TTS bundles exist only in an `audio` build. Both lists are
    // threaded into `startup` under the same gate, so without the feature
    // neither the flags nor the loaders are compiled at all.
    #[cfg(feature = "audio")]
    let (loaded_asr, loaded_tts) = {
        // Load any standalone Whisper ASR bundles before serving.
        let mut loaded_asr: Vec<(String, Arc<crate::server::handlers::AsrBundle>, PathBuf)> =
            Vec::with_capacity(asr_models.len());
        for arg in &asr_models {
            let (name, path) = crate::loader::parse_asr_model_arg(arg)?;
            let spinner = super::util::spinner(format!(
                "Loading ASR model '{}' from {}...",
                name,
                path.display()
            ));
            let bundle = crate::loader::load_whisper_from_dir::<ServerRuntime, _>(&path, &device)?;
            spinner.finish_and_clear();
            eprintln!(
                "  {} ASR model {} (variant={:?}, vocab={}, mel_bins={})",
                "Loaded".green(),
                name.bold(),
                bundle.variant,
                bundle.config.vocab_size,
                bundle.num_mel_bins,
            );
            loaded_asr.push((name, Arc::new(bundle), path));
        }

        // Load any TTS bundles on the server's device: Kokoro, VITS or
        // VoxCPM2 by directory layout. The batching options apply to VoxCPM2
        // only. Kokoro and VITS have no batching of their own.
        let tts_synth_options = voxr::voxcpm::VoxCpm2SynthOptions {
            max_batch: tts_max_batch,
            batch_window: std::time::Duration::from_millis(tts_batch_window_ms),
            continuous_batching: tts_continuous_batching,
            continuous_capacity: tts_continuous_capacity,
            ..Default::default()
        };
        let mut loaded_tts: Vec<(String, Arc<crate::server::handlers::TtsBundle>, PathBuf)> =
            Vec::with_capacity(tts_models.len());
        for arg in &tts_models {
            let (name, path) = crate::loader::parse_tts_model_arg(arg)?;
            let spinner = super::util::spinner(format!(
                "Loading TTS model '{}' from {}...",
                name,
                path.display()
            ));
            let bundle =
                crate::loader::load_tts_from_dir(&path, tts_synth_options.clone(), &device)?;
            spinner.finish_and_clear();
            let status = if bundle.has_engine() {
                "".normal()
            } else {
                "[no engine: synthesis returns 503]".yellow()
            };
            eprintln!(
                "  {} TTS model {} (voices={}, sr={} Hz) {}",
                "Loaded".green(),
                name.bold(),
                bundle.voices().len(),
                bundle.sample_rate,
                status,
            );
            loaded_tts.push((name, Arc::new(bundle), path));
        }
        (loaded_asr, loaded_tts)
    };

    // The voice directory: the CLI flag, else the env var. The model
    // directory's `voices/` is searched after it.
    if let Some(dir) = &voice_dir {
        eprintln!(
            "  {} Voice directory: {}",
            "✓".green(),
            dir.display().to_string().bold()
        );
    }

    // Start server
    server::start_with_batch(
        scheduler,
        server_config,
        live,
        api_keys,
        serving,
        loaded_vision,
        #[cfg(feature = "audio")]
        loaded_asr,
        #[cfg(feature = "audio")]
        loaded_tts,
        voice_dir,
    )
    .await?;

    Ok(())
}
