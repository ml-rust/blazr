//! Show running models on a blazr server

use anyhow::Result;
use colored::Colorize;

/// Query a running blazr server for loaded models
pub async fn ps(server: String) -> Result<()> {
    let url = format!("{}/api/ps", server.trim_end_matches('/'));

    let client = reqwest::Client::new();
    let response = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "{} Could not connect to server at {}",
                "Error:".red().bold(),
                server.bold()
            );
            eprintln!("  {}", e);
            eprintln!("\nMake sure {} is running.", "blazr serve".bold());
            return Ok(());
        }
    };

    if !response.status().is_success() {
        eprintln!(
            "{} Server returned status {}",
            "Error:".red().bold(),
            response.status()
        );
        return Ok(());
    }

    let body: serde_json::Value = response.json().await?;

    // Parse response - /api/ps returns { models: [...] }
    let models = body.get("models").and_then(|v| v.as_array());

    match models {
        Some(models) if !models.is_empty() => {
            eprintln!("{}\n", "Running models:".bold());
            for model in models {
                let name = model
                    .get("name")
                    .or_else(|| model.get("model"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                eprint!("  {}", name.bold());

                if let Some(size) = model.get("size").and_then(|v| v.as_u64()) {
                    let gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
                    eprint!("  {}", format!("{:.1} GB", gb).dimmed());
                }

                if let Some(expires) = model.get("expires_at").and_then(|v| v.as_str()) {
                    eprint!("  expires: {}", expires.dimmed());
                }

                eprintln!();
            }
        }
        _ => {
            eprintln!("No models currently loaded.");
        }
    }

    Ok(())
}
