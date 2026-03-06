//! HTTP server command

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use colored::Colorize;

use crate::config::ServerConfig;
use crate::engine::{BatchEngine, RequestScheduler, Scheduler};
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
    tls_cert: Option<PathBuf>,
    tls_key: Option<PathBuf>,
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
        tls_cert,
        tls_key,
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

    // Create RequestScheduler + BatchEngine if continuous batching is configured
    let request_scheduler = if let Some(ref model_name) = model {
        let executor = scheduler.get_executor(model_name).await?;
        let paged = executor.config().inference.paged_attention;
        let max_batch = executor.config().inference.max_batch_size;
        let block_size = executor.config().inference.block_size;
        let kv_pool = executor.config().inference.kv_pool_blocks;
        let max_ctx = executor.config().inference.max_context_len.unwrap_or(4096);

        if paged && max_batch > 1 {
            let total_blocks = if kv_pool > 0 {
                kv_pool
            } else {
                (max_ctx * max_batch / block_size).max(256)
            };

            let allocator =
                boostr::inference::memory::CpuBlockAllocator::new(total_blocks, block_size);
            let sched_config = boostr::inference::scheduler::SchedulerConfig {
                max_batch_size: max_batch,
                max_batch_tokens: max_ctx,
                block_size,
                ..Default::default()
            };

            // Create prefix cache if enabled
            let prefix_cache_enabled = executor.config().inference.prefix_cache;
            let rs = if prefix_cache_enabled {
                let cache_config = boostr::inference::prefix_cache::PrefixCacheConfig {
                    enabled: true,
                    block_size,
                    max_cached_blocks: executor
                        .config()
                        .inference
                        .max_cached_blocks
                        .max(total_blocks / 4),
                    ..Default::default()
                };
                let cache = boostr::inference::prefix_cache::PrefixCache::new(
                    allocator.clone(),
                    cache_config,
                );
                RequestScheduler::with_prefix_cache(allocator, sched_config, cache)
            } else {
                RequestScheduler::new(allocator, sched_config)
            };

            // Spawn BatchEngine in background
            let batch_engine = BatchEngine::new(executor, Arc::clone(&rs))?;
            tokio::spawn(async move {
                batch_engine.run().await;
            });

            eprintln!(
                "  Batching: {} (max_batch={}, blocks={})",
                "continuous".green(),
                max_batch,
                total_blocks,
            );

            Some(rs)
        } else {
            None
        }
    } else {
        None
    };

    // Start server
    server::start_with_batch(scheduler, server_config, api_keys, request_scheduler).await?;

    Ok(())
}
