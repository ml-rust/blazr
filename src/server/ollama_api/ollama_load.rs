//! Load and unload requests: `/api/chat` with no messages and
//! `/api/generate` with no prompt.

use std::sync::Arc;
use std::time::Duration;

use axum::{http::StatusCode, response::IntoResponse, response::Response, Json};

use super::model_name::{is_known, not_found, resolve};
use super::ollama_error::{from_openai, ollama_error};
use super::ollama_render::{Kind, Render};
use super::ollama_types::OllamaKeepAlive;
use crate::engine::parse_keep_alive;
use crate::server::generation::apply_keep_alive;
use crate::server::handlers::{select_executor, AppState};

/// Load the model `requested` names, or unload it when `keep_alive` is
/// zero. The reply has empty output and `done_reason` `load` or `unload`.
pub(super) async fn load_reply(
    state: &Arc<AppState>,
    requested: &str,
    keep_alive: Option<&OllamaKeepAlive>,
    kind: Kind,
) -> Response {
    let name = resolve(state, requested).await;
    let keep_alive = keep_alive.map(OllamaKeepAlive::to_arg);
    let render = Render::new(kind);
    if keep_alive.as_deref().map(parse_keep_alive) == Some(Some(Duration::ZERO)) {
        if !is_known(state, &name).await {
            return ollama_error(StatusCode::NOT_FOUND, &not_found(requested));
        }
        state
            .scheduler
            .set_keep_alive(&name, Some(Duration::ZERO))
            .await;
        return (StatusCode::OK, Json(render.lifecycle(requested, "unload"))).into_response();
    }
    if let Err(response) = select_executor(state, &name, None).await {
        return from_openai(*response, requested).await;
    }
    apply_keep_alive(state, &name, &keep_alive).await;
    (StatusCode::OK, Json(render.lifecycle(requested, "load"))).into_response()
}

/// End-to-end runs through the router on a loaded seeded model.
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use anyhow::Result;
    use serde_json::json;

    use super::super::e2e_rig::{Rig, MODEL, REQUESTED};

    #[tokio::test]
    async fn an_empty_request_loads_and_keep_alive_zero_unloads() -> Result<()> {
        let rig = Rig::new().await?;
        let load = json!({"model": REQUESTED, "messages": []});
        let got = rig.ok("/api/chat", &load).await?;
        assert_eq!(got["done"], true, "{got}");
        assert_eq!(got["done_reason"], "load", "{got}");
        assert_eq!(got["model"], REQUESTED, "{got}");
        assert_eq!(got["message"]["content"], "", "{got}");
        let got = rig
            .ok("/api/generate", &json!({"model": REQUESTED}))
            .await?;
        assert_eq!(got["done_reason"], "load", "{got}");
        assert_eq!(got["response"], "", "{got}");
        assert!(rig.state.scheduler.is_loaded(MODEL).await);

        let unload = json!({"model": REQUESTED, "messages": [], "keep_alive": 0});
        let got = rig.ok("/api/chat", &unload).await?;
        assert_eq!(got["done"], true, "{got}");
        assert_eq!(got["done_reason"], "unload", "{got}");
        assert!(!rig.state.scheduler.is_loaded(MODEL).await);
        let loaded = rig.state.scheduler.list_loaded().await;
        assert!(loaded.iter().all(|m| m.name != MODEL));
        Ok(())
    }
}
