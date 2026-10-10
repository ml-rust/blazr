//! `POST /api/pull` — download a model from HuggingFace with streaming
//! NDJSON progress.

use axum::{
    extract::Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use hf_hub::repository::RepoTypeModel;
use hf_hub::HFRepository;
use serde::Deserialize;
use std::path::Path;
use tokio::sync::mpsc;

use crate::loader::hub_client::{async_client, async_get, async_model};

/// POST /api/pull — Pull model from HuggingFace with streaming progress
pub async fn pull_model(Json(request): Json<PullRequest>) -> Response {
    use axum::body::Body;
    use futures::stream;

    let (tx, rx) = mpsc::channel::<String>(32);
    let repo = request.name.clone();

    // Spawn download in background
    tokio::spawn(async move {
        let client = match async_client(None) {
            Ok(client) => client,
            Err(e) => {
                let msg = serde_json::json!({ "error": format!("Failed to init HF API: {}", e) });
                let _ = tx.send(format!("{}\n", msg)).await;
                return;
            }
        };
        let repo_api = async_model(&client, &repo);

        let model_name = repo.split('/').next_back().unwrap_or(&repo);
        let model_dir = std::env::var("BLAZR_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("./models"));
        let dest = model_dir.join(model_name);
        let _ = tokio::fs::create_dir_all(&dest).await;

        let files_to_try = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
        ];

        for filename in files_to_try {
            pull_file(&tx, &repo_api, filename, &dest).await;
        }

        // Try sharded safetensors
        if let Ok(cached) = async_get(&repo_api, "model.safetensors.index.json", true).await {
            let index_dest = dest.join("model.safetensors.index.json");
            let _ = tokio::fs::copy(&cached, &index_dest).await;
            if let Ok(content) = tokio::fs::read_to_string(&index_dest).await {
                if let Ok(index) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                        let mut shards: Vec<String> = weight_map
                            .values()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                        shards.sort();
                        shards.dedup();
                        for shard in &shards {
                            pull_file(&tx, &repo_api, shard, &dest).await;
                        }
                    }
                }
            }
        }

        send_status(&tx, "success").await;
    });

    // Stream status messages as NDJSON
    let body_stream = stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|msg| (Ok::<_, std::convert::Infallible>(msg), rx))
    });

    let body = Body::from_stream(body_stream);
    Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Sends one NDJSON `{"status": ...}` line. A closed stream drops it.
async fn send_status(tx: &mpsc::Sender<String>, status: &str) {
    let msg = serde_json::json!({ "status": status });
    let _ = tx.send(format!("{}\n", msg)).await;
}

/// Downloads the repo file `name` and copies it into `dest`, with a
/// `pulling` line before and a `downloaded` line after a fetch that worked.
async fn pull_file(
    tx: &mpsc::Sender<String>,
    repo_api: &HFRepository<RepoTypeModel>,
    name: &str,
    dest: &Path,
) {
    send_status(tx, &format!("pulling {}", name)).await;
    if let Ok(cached) = async_get(repo_api, name, true).await {
        let target = dest.join(name);
        if let Some(parent) = target.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let _ = tokio::fs::copy(&cached, &target).await;
        send_status(tx, &format!("downloaded {}", name)).await;
    }
}

#[derive(Deserialize)]
pub struct PullRequest {
    pub name: String,
}
