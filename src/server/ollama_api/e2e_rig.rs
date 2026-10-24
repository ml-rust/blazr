//! A router over one loaded seeded model, for the Ollama end-to-end tests.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use axum::{body::Body, http::Request, Router};
use serde_json::Value;
use tower::ServiceExt;

use crate::engine::test_support::seeded_full_vocab_executor;
use crate::server::handlers::{test_state, AppState};
use crate::server::routes::api_routes;

/// The name the model loads under.
pub(in crate::server) const MODEL: &str = "m";
/// The name a client sends: Ollama clients add `:latest`.
pub(in crate::server) const REQUESTED: &str = "m:latest";
/// The seed of the model's weights.
const SEED: u64 = 0x011a;

/// The server state and the model directory it searches.
pub(in crate::server) struct Rig {
    pub state: Arc<AppState>,
    /// Holds the model directory open for the rig's life.
    _dir: tempfile::TempDir,
}

impl Rig {
    /// A server whose model [`MODEL`] is loaded.
    pub(in crate::server) async fn new() -> Result<Self> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join(MODEL))?;
        std::fs::write(dir.path().join(MODEL).join("config.json"), "{}")?;
        let state = Arc::new(test_state(dir.path()));
        state
            .scheduler
            .load_base(MODEL, |_| seeded_full_vocab_executor(SEED))
            .await?;
        Ok(Self { state, _dir: dir })
    }

    fn router(&self) -> Router {
        api_routes().with_state(Arc::clone(&self.state))
    }

    /// The status and raw body of `POST path` with `body`.
    pub(in crate::server) async fn post(&self, path: &str, body: &Value) -> Result<(u16, String)> {
        let request = Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))?;
        let response = self.router().oneshot(request).await?;
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
        Ok((status, String::from_utf8(bytes.to_vec())?))
    }

    /// The JSON body of `POST path`, which must answer 200.
    pub(in crate::server) async fn ok(&self, path: &str, body: &Value) -> Result<Value> {
        let (status, raw) = self.post(path, body).await?;
        if status != 200 {
            return Err(anyhow!("{path} answered {status}: {raw}"));
        }
        Ok(serde_json::from_str(&raw)?)
    }

    /// The NDJSON lines of `POST path`, which must answer 200.
    pub(in crate::server) async fn lines(&self, path: &str, body: &Value) -> Result<Vec<Value>> {
        let (status, raw) = self.post(path, body).await?;
        if status != 200 {
            return Err(anyhow!("{path} answered {status}: {raw}"));
        }
        if !raw.ends_with('\n') {
            return Err(anyhow!("{path}: the body does not end a line: {raw:?}"));
        }
        raw.lines()
            .map(|line| serde_json::from_str(line).map_err(|e| anyhow!("{line:?}: {e}")))
            .collect()
    }
}
