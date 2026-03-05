//! HTTP server command

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

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
        tracing::info!("Pre-loading model: {}", model_name);
        let load_start = std::time::Instant::now();
        let executor = scheduler.get_executor(model_name).await?;
        let load_time = load_start.elapsed();
        if let Err(e) = executor.warmup() {
            tracing::warn!("Model warmup failed (first request may be slower): {}", e);
        }
        tracing::info!(
            "Model '{}' loaded in {:.1}s (vocab={}, ctx={})",
            model_name,
            load_time.as_secs_f64(),
            executor.vocab_size(),
            executor.config().max_seq_len(),
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

    let addr = server_config.addr();
    eprintln!();
    eprintln!("  blazr v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("  Listening on http://{}", addr);
    if let Some(ref m) = model {
        eprintln!("  Model: {}", m);
    }
    #[cfg(feature = "cuda")]
    eprintln!("  Backend: CUDA");
    #[cfg(not(feature = "cuda"))]
    eprintln!("  Backend: CPU");
    if !api_keys.is_empty() {
        eprintln!("  Auth: {} API key(s)", api_keys.len());
    }
    if server_config.cors_enabled {
        eprintln!("  CORS: enabled");
    }
    eprintln!();

    // Start server
    server::start(scheduler, server_config, api_keys).await?;

    Ok(())
}
