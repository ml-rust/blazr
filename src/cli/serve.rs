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
pub async fn serve(model: Option<String>, port: u16, host: String) -> Result<()> {
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

    // Pre-load model if specified
    if let Some(ref model_name) = model {
        tracing::info!("Pre-loading model: {}", model_name);
        scheduler.get_executor(model_name).await?;
        tracing::info!("Model loaded successfully");
    }

    // Server config
    let server_config = ServerConfig {
        port,
        host: host.clone(),
        ..Default::default()
    };

    let addr = server_config.addr();
    tracing::info!("Starting server at http://{}", addr);

    // Start server
    server::start(scheduler, server_config).await?;

    Ok(())
}
