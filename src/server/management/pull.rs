//! `POST /api/pull` — download a model from HuggingFace with streaming
//! NDJSON progress.

use axum::{
    extract::Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;

/// POST /api/pull — Pull model from HuggingFace with streaming progress
pub async fn pull_model(Json(request): Json<PullRequest>) -> Response {
    use axum::body::Body;
    use futures::stream;
    use tokio::sync::mpsc;

    let (tx, rx) = mpsc::channel::<String>(32);
    let repo = request.name.clone();

    // Spawn download in background
    tokio::spawn(async move {
        let send_status = |tx: &mpsc::Sender<String>, status: &str| {
            let msg = serde_json::json!({ "status": status });
            let _ = tx.blocking_send(format!("{}\n", msg));
        };

        let api = match hf_hub::api::sync::Api::new() {
            Ok(api) => api,
            Err(e) => {
                let msg = serde_json::json!({ "error": format!("Failed to init HF API: {}", e) });
                let _ = tx.blocking_send(format!("{}\n", msg));
                return;
            }
        };

        let repo_api = api.model(repo.clone());

        let model_name = repo.split('/').next_back().unwrap_or(&repo);
        let model_dir = std::env::var("BLAZR_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("./models"));
        let dest = model_dir.join(model_name);
        let _ = std::fs::create_dir_all(&dest);

        let files_to_try = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
        ];

        for filename in files_to_try {
            send_status(&tx, &format!("pulling {}", filename));
            if let Ok(cached) = repo_api.get(filename) {
                let _ = std::fs::copy(&cached, dest.join(filename));
                send_status(&tx, &format!("downloaded {}", filename));
            }
        }

        // Try sharded safetensors
        if let Ok(cached) = repo_api.get("model.safetensors.index.json") {
            let index_dest = dest.join("model.safetensors.index.json");
            let _ = std::fs::copy(&cached, &index_dest);
            if let Ok(content) = std::fs::read_to_string(&index_dest) {
                if let Ok(index) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
                        let mut shards: Vec<String> = weight_map
                            .values()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                        shards.sort();
                        shards.dedup();
                        for shard in &shards {
                            send_status(&tx, &format!("pulling {}", shard));
                            if let Ok(cached) = repo_api.get(shard) {
                                let _ = std::fs::copy(&cached, dest.join(shard));
                                send_status(&tx, &format!("downloaded {}", shard));
                            }
                        }
                    }
                }
            }
        }

        send_status(&tx, "success");
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

#[derive(Deserialize)]
pub struct PullRequest {
    pub name: String,
}
