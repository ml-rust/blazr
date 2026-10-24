//! `POST /api/generate`, the Ollama prompt endpoint.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{rejection::JsonRejection, State},
    response::Response,
    Json,
};

use super::model_name::resolve;
use super::ollama_convert::generate_request;
use super::ollama_error::{bad_request, rejection};
use super::ollama_load::load_reply;
use super::ollama_render::Kind;
use super::ollama_reply::{reply, Reply};
use super::ollama_types::OllamaGenerateRequest;
use crate::server::handlers::AppState;

/// POST /api/generate. An empty `prompt` loads or unloads the model.
pub async fn ollama_generate(
    State(state): State<Arc<AppState>>,
    body: Result<Json<OllamaGenerateRequest>, JsonRejection>,
) -> Response {
    let start = Instant::now();
    let request = match body {
        Ok(Json(request)) => request,
        Err(e) => return rejection(e),
    };
    if request.prompt.as_deref().unwrap_or("").is_empty() {
        let keep_alive = request.keep_alive.as_ref();
        return load_reply(&state, &request.model, keep_alive, Kind::Generate).await;
    }
    let model = resolve(&state, &request.model).await;
    let chat = match generate_request(&request, &model) {
        Ok(chat) => chat,
        Err(message) => return bad_request(&message),
    };
    let how = Reply {
        requested: request.model.clone(),
        stream: request.stream.unwrap_or(true),
        kind: Kind::Generate,
        start,
        num_ctx: request.options.as_ref().and_then(|o| o.num_ctx),
    };
    reply(&state, &chat, how).await
}

/// End-to-end runs through the router on a loaded seeded model.
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use anyhow::{anyhow, Result};
    use serde_json::{json, Value};

    use super::super::e2e_rig::{Rig, MODEL, REQUESTED};

    /// Most tokens each greedy run generates.
    const NUM_PREDICT: usize = 6;
    const PROMPT: &str = "The sky is";

    /// A greedy `/api/generate` body with `stream` false.
    fn greedy(raw: bool) -> Value {
        json!({
            "model": REQUESTED,
            "prompt": PROMPT,
            "raw": raw,
            "stream": false,
            "options": {"temperature": 0, "num_predict": NUM_PREDICT},
        })
    }

    /// `got[key]` as a count.
    fn count(got: &Value, key: &str) -> Result<u64> {
        got[key].as_u64().ok_or_else(|| anyhow!("no {key}: {got}"))
    }

    #[tokio::test]
    async fn a_templated_prompt_returns_its_context_and_matches_the_chat_route() -> Result<()> {
        let rig = Rig::new().await?;
        let got = rig.ok("/api/generate", &greedy(false)).await?;
        assert_eq!(got["model"], REQUESTED, "{got}");
        assert_eq!(got["done"], true, "{got}");
        let response = got["response"].as_str().unwrap_or_default();
        let prompt = count(&got, "prompt_eval_count")? as usize;

        // The context is the prompt ids, then the response re-encoded.
        let context: Vec<u32> = serde_json::from_value(got["context"].clone())?;
        let executor = rig.state.scheduler.get_executor(MODEL).await?;
        let tail = executor
            .tokenizer()
            .encode_with(response, &splintr::SpecialMode::Ordinary)?;
        assert_eq!(context.len(), prompt + tail.len(), "{got}");
        assert_eq!(context[prompt..], tail[..], "{got}");

        let chat = json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "max_tokens": NUM_PREDICT,
        });
        let openai = rig.ok("/v1/chat/completions", &chat).await?;
        assert_eq!(openai["choices"][0]["message"]["content"], response);
        assert_eq!(count(&openai["usage"], "prompt_tokens")? as usize, prompt);
        assert_eq!(
            count(&openai["usage"], "completion_tokens")?,
            count(&got, "eval_count")?
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_raw_prompt_has_no_context_and_matches_the_completions_route() -> Result<()> {
        let rig = Rig::new().await?;
        let got = rig.ok("/api/generate", &greedy(true)).await?;
        assert_eq!(got["done"], true, "{got}");
        assert!(got.get("context").is_none(), "{got}");
        let reason = got["done_reason"].as_str().unwrap_or_default();
        assert!(matches!(reason, "stop" | "length"), "{got}");

        let completion = json!({
            "model": MODEL,
            "prompt": PROMPT,
            "temperature": 0,
            "max_tokens": NUM_PREDICT,
        });
        let openai = rig.ok("/v1/completions", &completion).await?;
        assert_eq!(openai["choices"][0]["text"], got["response"], "{got}");
        let usage = &openai["usage"];
        assert_eq!(
            count(usage, "prompt_tokens")?,
            count(&got, "prompt_eval_count")?
        );
        assert_eq!(
            count(usage, "completion_tokens")?,
            count(&got, "eval_count")?
        );
        Ok(())
    }
}
