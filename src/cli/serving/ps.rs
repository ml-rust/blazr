//! Show running models on a blazr server

use anyhow::Result;
use colored::Colorize;

/// Query a running blazr server for loaded models. `api_key` goes in
/// `Authorization: Bearer <key>` when set.
pub async fn ps(server: String, api_key: Option<String>) -> Result<()> {
    let url = format!("{}/api/ps", server.trim_end_matches('/'));

    crate::tls::install_crypto_provider();
    let client = reqwest::Client::new();
    let mut request = client.get(&url);
    if let Some(key) = &api_key {
        request = request.bearer_auth(key);
    }
    let response = match request.send().await {
        Ok(r) => r,
        Err(e) => anyhow::bail!(
            "could not connect to the server at {server}: {e}. Start `blazr serve`, or pass \
             its address with --server"
        ),
    };

    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        let why = if api_key.is_some() {
            "rejected the API key"
        } else {
            "needs an API key"
        };
        anyhow::bail!(
            "the server at {url} answered 401: it {why}. Pass a key it accepts with --api-key, \
             --api-key-file or BLAZR_API_KEY"
        );
    }
    if !response.status().is_success() {
        anyhow::bail!("the server at {url} answered {}", response.status());
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
