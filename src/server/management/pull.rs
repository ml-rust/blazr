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
use crate::loader::hub_client::pull_plan::{plan, shards_of, SAFETENSORS_INDEX};
use crate::loader::hub_client::{async_client, async_files, async_get, async_model};
use crate::server::handlers::AppState;
use crate::server::ndjson::ndjson_response;

/// How a pull error tells the client to name one file: this route takes
/// no file name.
const PICK: &str = "Pull one file with `blazr pull <repo> --file <name>`";

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

    // Spawn download in background. The first error ends the pull with an
    // error line.
    tokio::spawn(async move {
        if let Err(message) = pull_repo(&tx, &repo, &model_dir).await {
            let _ = tx.send(Err(message)).await;
            return;
        }
        send_status(&tx, "success").await;
    });

    progress_response(rx)
}

/// Pulls the files of `repo` the repo's file list picks into its directory
/// under `model_dir`. The error is the message of the first failure.
async fn pull_repo(
    tx: &mpsc::Sender<Progress>,
    repo: &str,
    model_dir: &Path,
) -> Result<(), String> {
    let model_name = repo.split('/').next_back().unwrap_or(repo);
    if !stays_in_dir(model_name) {
        return Err(format!(
            "repo name '{repo}' leaves the model directory. Send a Hugging Face repo id, \
             such as 'org/name'"
        ));
    }
    let client = async_client(None).map_err(|e| format!("Failed to init HF API: {e}"))?;
    let repo_api = async_model(&client, repo);
    let files = async_files(&repo_api)
        .await
        .map_err(|e| format!("listing the files of {repo} failed: {e}"))?;
    let plan = plan(&files).map_err(|e| e.message(repo, PICK))?;
    let dest = model_dir.join(model_name);
    tokio::fs::create_dir_all(&dest)
        .await
        .map_err(|e| format!("creating {} failed: {e}", dest.display()))?;
    for name in &plan.files {
        pull_file(tx, &repo_api, repo, name, &dest).await?;
    }
    if plan.sharded {
        let index_path = dest.join(SAFETENSORS_INDEX);
        let index = tokio::fs::read_to_string(&index_path)
            .await
            .map_err(|e| format!("reading {} failed: {e}", index_path.display()))?;
        for shard in shards_of(&index).map_err(|e| format!("{repo}: {e}"))? {
            pull_file(tx, &repo_api, repo, &shard, &dest).await?;
        }
    }
    Ok(())
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
/// `pulling` line before and a `downloaded` line after. The plan checked
/// that `name` stays in `dest`.
async fn pull_file(
    tx: &mpsc::Sender<Progress>,
    repo_api: &HFRepository<RepoTypeModel>,
    repo: &str,
    name: &str,
    dest: &Path,
) -> Result<(), String> {
    send_status(tx, &format!("pulling {}", name)).await;
    let cached = async_get(repo_api, name, true)
        .await
        .map_err(|e| format!("downloading {name} from {repo} failed: {e}"))?;
    let target = dest.join(name);
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("creating {} failed: {e}", parent.display()))?;
    }
    tokio::fs::copy(&cached, &target)
        .await
        .map_err(|e| format!("copying {name} to {} failed: {e}", target.display()))?;
    send_status(tx, &format!("downloaded {}", name)).await;
    Ok(())
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
