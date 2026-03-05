//! HTTP server command

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use colored::Colorize;

use crate::config::ServerConfig;
use crate::engine::Scheduler;
use crate::server;

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

/// Start the inference server
pub async fn serve(
    model: Option<String>,
    port: u16,
    host: String,
    api_key: Option<String>,
    api_key_file: Option<PathBuf>,
) -> Result<()> {
    // Get model directory
    let model_dir = std::env::var("BLAZR_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./models"));

    // Initialize device
    #[cfg(feature = "cuda")]
    let device = boostr::CudaDevice::new(0);
    #[cfg(not(feature = "cuda"))]
    let device = boostr::CpuDevice::new();

    // Create scheduler
    let scheduler = Arc::new(Scheduler::<ServerRuntime>::new(model_dir, device));

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
            "  {} {} in {:.1}s (vocab={}, ctx={}{})",
            "Loaded".green(),
            model_name.bold(),
            load_time.as_secs_f64(),
            executor.vocab_size(),
            executor.config().max_seq_len(),
            vram_str,
        );
    }

    // Server config
    let server_config = ServerConfig {
        port,
        host,
        ..Default::default()
    };

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
    eprintln!("  Listening on {}", format!("http://{}", addr).underline());
    if let Some(ref m) = model {
        eprintln!("  Model: {}", m.bold());
    }
    #[cfg(feature = "cuda")]
    eprintln!("  Backend: {}", "CUDA".green());
    #[cfg(not(feature = "cuda"))]
    eprintln!("  Backend: {}", "CPU".yellow());
    if !api_keys.is_empty() {
        eprintln!("  Auth: {} API key(s)", api_keys.len());
    }
    if server_config.cors_enabled {
        eprintln!("  CORS: {}", "enabled".yellow());
    }
    eprintln!();

    // Start server
    server::start(scheduler, server_config, api_keys).await?;

    Ok(())
}
