//! `POST /api/pull` — download a model from HuggingFace with streaming
//! NDJSON progress.

use axum::{
    extract::{Json, State},
    response::Response,
};
use hf_hub::repository::RepoTypeModel;
use hf_hub::HFRepository;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::engine::model_dir::stays_in_dir;
use crate::loader::hub_client::{async_client, async_get, async_model};
use crate::server::handlers::AppState;
use crate::server::ndjson::ndjson_response;

/// One progress line: a `{"status": ...}` object, or the message of the
/// error that ends the pull.
type Progress = Result<Value, String>;

/// POST /api/pull — Pull model from HuggingFace with streaming progress.
/// The model lands in the scheduler's model directory.
pub async fn pull_model(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PullRequest>,
) -> Response {
    let (tx, rx) = mpsc::channel::<Progress>(32);
    let repo = request.name.clone();
    let model_dir = state.scheduler.model_dir().to_path_buf();

    // Spawn download in background
    tokio::spawn(async move {
        let client = match async_client(None) {
            Ok(client) => client,
            Err(e) => {
                let _ = tx.send(Err(format!("Failed to init HF API: {}", e))).await;
                return;
            }
        };
        let repo_api = async_model(&client, &repo);

        let model_name = repo.split('/').next_back().unwrap_or(&repo);
        if !stays_in_dir(model_name) {
            let message = format!(
                "repo name '{repo}' leaves the model directory. Send a Hugging Face repo id, \
                 such as 'org/name'"
            );
            let _ = tx.send(Err(message)).await;
            return;
        }
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

    progress_response(rx)
}

/// The NDJSON response of the progress lines `rx` receives.
fn progress_response(rx: mpsc::Receiver<Progress>) -> Response {
    ndjson_response(ReceiverStream::new(rx))
}

/// Sends one NDJSON `{"status": ...}` line. A closed stream drops it.
async fn send_status(tx: &mpsc::Sender<Progress>, status: &str) {
    let _ = tx.send(Ok(json!({ "status": status }))).await;
}

/// Downloads the repo file `name` and copies it into `dest`, with a
/// `pulling` line before and a `downloaded` line after a fetch that worked.
async fn pull_file(
    tx: &mpsc::Sender<Progress>,
    repo_api: &HFRepository<RepoTypeModel>,
    name: &str,
    dest: &Path,
) {
    if !stays_in_dir(name) {
        let message = format!(
            "file name '{name}' in the repo's index leaves the model directory. Pull a repo with \
             plain file names"
        );
        let _ = tx.send(Err(message)).await;
        return;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ndjson::ndjson_test::body;

    #[tokio::test]
    async fn progress_lines_keep_their_ndjson_form() -> anyhow::Result<()> {
        let (tx, rx) = mpsc::channel::<Progress>(8);
        send_status(&tx, "pulling config.json").await;
        send_status(&tx, "success").await;
        drop(tx);
        let response = progress_response(rx);
        assert_eq!(response.headers()["content-type"], "application/x-ndjson");
        assert_eq!(
            body(response).await?,
            "{\"status\":\"pulling config.json\"}\n{\"status\":\"success\"}\n"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_pull_ends_with_an_error_line() -> anyhow::Result<()> {
        let (tx, rx) = mpsc::channel::<Progress>(8);
        let _ = tx.send(Err("Failed to init HF API: offline".into())).await;
        drop(tx);
        assert_eq!(
            body(progress_response(rx)).await?,
            "{\"error\":\"Failed to init HF API: offline\"}\n"
        );
        Ok(())
    }
}
