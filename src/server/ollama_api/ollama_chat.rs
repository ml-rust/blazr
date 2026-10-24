//! `POST /api/chat`, the Ollama chat endpoint.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{rejection::JsonRejection, State},
    response::Response,
    Json,
};

use super::model_name::resolve;
use super::ollama_convert::chat_request;
use super::ollama_error::{bad_request, rejection};
use super::ollama_load::load_reply;
use super::ollama_render::Kind;
use super::ollama_reply::{reply, Reply};
use super::ollama_types::OllamaChatRequest;
use crate::server::handlers::AppState;

/// POST /api/chat. An empty `messages` loads or unloads the model.
pub async fn ollama_chat(
    State(state): State<Arc<AppState>>,
    body: Result<Json<OllamaChatRequest>, JsonRejection>,
) -> Response {
    let start = Instant::now();
    let request = match body {
        Ok(Json(request)) => request,
        Err(e) => return rejection(e),
    };
    if request.messages.is_empty() {
        return load_reply(
            &state,
            &request.model,
            request.keep_alive.as_ref(),
            Kind::Chat,
        )
        .await;
    }
    let model = resolve(&state, &request.model).await;
    let chat = match chat_request(&request, &model) {
        Ok(chat) => chat,
        Err(message) => return bad_request(&message),
    };
    let how = Reply {
        requested: request.model.clone(),
        stream: request.stream.unwrap_or(true),
        kind: Kind::Chat,
        start,
        num_ctx: request.options.as_ref().and_then(|o| o.num_ctx),
    };
    reply(&state, &chat, how).await
}

/// End-to-end runs through the router on a loaded seeded model.
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use anyhow::Result;
    use serde_json::{json, Value};

    use super::super::e2e_rig::{Rig, MODEL, REQUESTED};

    /// Most tokens each greedy run generates.
    const NUM_PREDICT: usize = 6;

    fn messages() -> Value {
        json!([{"role": "user", "content": "Name a colour."}])
    }

    /// An `/api/chat` body: greedy, [`NUM_PREDICT`] tokens.
    fn greedy(stream: Option<bool>) -> Value {
        let mut body = json!({
            "model": REQUESTED,
            "messages": messages(),
            "options": {"temperature": 0, "num_predict": NUM_PREDICT},
        });
        if let Some(stream) = stream {
            body["stream"] = stream.into();
        }
        body
    }

    /// The text, prompt tokens and completion tokens `/v1/chat/completions`
    /// answers the same request with at temperature 0.
    async fn openai_chat(rig: &Rig) -> Result<(String, u64, u64)> {
        let body = json!({
            "model": MODEL,
            "messages": messages(),
            "temperature": 0,
            "max_tokens": NUM_PREDICT,
        });
        let got = rig.ok("/v1/chat/completions", &body).await?;
        let text = got["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let usage = &got["usage"];
        let count = |key: &str| usage[key].as_u64().unwrap_or(u64::MAX);
        Ok((text, count("prompt_tokens"), count("completion_tokens")))
    }

    #[tokio::test]
    async fn a_default_request_streams_lines_that_match_the_openai_route() -> Result<()> {
        let rig = Rig::new().await?;
        let lines = rig.lines("/api/chat", &greedy(None)).await?;
        let (done, chunks) = lines.split_last().expect("at least the done object");
        for chunk in chunks {
            assert_eq!(chunk["done"], false, "{chunk}");
            assert_eq!(chunk["model"], REQUESTED, "{chunk}");
            assert_eq!(chunk["message"]["role"], "assistant", "{chunk}");
        }
        let text: String = chunks
            .iter()
            .filter_map(|c| c["message"]["content"].as_str())
            .collect();
        assert_eq!(done["done"], true, "{done}");
        assert_eq!(done["model"], REQUESTED, "{done}");
        assert_eq!(done["message"]["content"], "", "{done}");
        let reason = done["done_reason"].as_str().unwrap_or_default();
        assert!(matches!(reason, "stop" | "length"), "{done}");

        let (expected, prompt, completion) = openai_chat(&rig).await?;
        assert_eq!(text, expected);
        assert_eq!(done["prompt_eval_count"].as_u64(), Some(prompt), "{done}");
        assert_eq!(done["eval_count"].as_u64(), Some(completion), "{done}");
        assert!(prompt > 0 && completion > 0, "{done}");
        if reason == "length" {
            assert_eq!(completion, NUM_PREDICT as u64, "{done}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn stream_false_answers_one_object_with_the_whole_message() -> Result<()> {
        let rig = Rig::new().await?;
        let got = rig.ok("/api/chat", &greedy(Some(false))).await?;
        assert_eq!(got["done"], true, "{got}");
        assert_eq!(got["model"], REQUESTED, "{got}");
        assert_eq!(got["message"]["role"], "assistant", "{got}");
        let reason = got["done_reason"].as_str().unwrap_or_default();
        assert!(matches!(reason, "stop" | "length"), "{got}");

        let (expected, prompt, completion) = openai_chat(&rig).await?;
        assert_eq!(got["message"]["content"], expected.as_str(), "{got}");
        assert_eq!(got["prompt_eval_count"].as_u64(), Some(prompt), "{got}");
        assert_eq!(got["eval_count"].as_u64(), Some(completion), "{got}");
        for key in ["total_duration", "load_duration", "eval_duration"] {
            assert!(got[key].is_u64(), "{key}: {got}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn format_json_constrains_the_output_to_a_json_object() -> Result<()> {
        let rig = Rig::new().await?;
        let mut body = greedy(Some(false));
        body["format"] = "json".into();
        body["options"]["num_predict"] = 40.into();
        let got = rig.ok("/api/chat", &body).await?;
        // The grammar ends generation once the object closes.
        assert_eq!(got["done_reason"], "stop", "{got}");
        let text = got["message"]["content"].as_str().unwrap_or_default();
        let parsed: Value = serde_json::from_str(text)?;
        assert!(parsed.is_object(), "{got}");
        Ok(())
    }
}
