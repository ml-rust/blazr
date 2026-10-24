//! Route definitions

use std::sync::Arc;

use axum::{
    routing::{delete, get, post},
    Router,
};

use super::anthropic::{count_tokens, messages};
#[cfg(feature = "audio")]
use super::audio::{speech, transcriptions, translations, voices};
use super::chat::chat_completions;
use super::completions::completions;
use super::decision::systemone;
use super::embeddings::embeddings;
use super::handlers::{
    apply_template, create_slot, delete_slot, detokenize, get_model, list_models, list_slots,
    tokenize, AppState,
};
use super::infill::infill;
use super::lora::{list_loras, load_lora, unload_lora};
use super::management::{
    copy_model, delete_model, list_running, list_tags, pull_model, show_model,
};
use super::ollama_chat::ollama_chat;
use super::ollama_embed::{ollama_embed, ollama_embeddings};
use super::ollama_generate::ollama_generate;
use super::rerank::rerank;
use super::responses::responses;

/// Create the API router with OpenAI-compatible endpoints (auth-protected)
pub fn api_routes() -> Router<Arc<AppState>> {
    let router = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/models", get(list_models))
        .route("/v1/models/{model_id}", get(get_model))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/responses", post(responses))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/infill", post(infill))
        .route("/v1/systemone", post(systemone));

    // The audio endpoints exist only in a build that can serve them.
    // Registering them anyway would answer a client with a 500 where a 404 is
    // the truth: the endpoint is absent from this build, not broken in it.
    #[cfg(feature = "audio")]
    let router = router
        .route("/v1/audio/speech", post(speech))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/v1/audio/translations", post(translations))
        .route("/v1/audio/voices", get(voices));

    router
        .route("/rerank", post(rerank))
        .route("/v1/rerank", post(rerank))
        // Tokenization endpoints
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/apply-template", post(apply_template))
        // Slot management endpoints
        .route("/api/slots", get(list_slots).post(create_slot))
        .route("/api/slots/{slot_id}", delete(delete_slot))
        // Model management endpoints
        .route("/api/tags", get(list_tags))
        .route("/api/show", post(show_model))
        .route("/api/ps", get(list_running))
        .route("/api/delete", delete(delete_model))
        .route("/api/copy", post(copy_model))
        .route("/api/pull", post(pull_model))
        // Ollama generation and embedding endpoints
        .route("/api/chat", post(ollama_chat))
        .route("/api/generate", post(ollama_generate))
        .route("/api/embed", post(ollama_embed))
        .route("/api/embeddings", post(ollama_embeddings))
        // LoRA adapter management endpoints
        .route("/v1/lora", get(list_loras).post(load_lora))
        .route("/v1/lora/{name}", delete(unload_lora))
}

#[cfg(test)]
mod tests {
    /// axum panics on a route path it cannot parse, such as the `/:name`
    /// form it no longer accepts. Building the router parses every path.
    #[test]
    fn every_route_path_parses() {
        let _ = super::api_routes();
    }

    use std::sync::Arc;

    use axum::{body::Body, http::Request};
    use serde_json::{json, Value};
    use tower::ServiceExt;

    /// The status and JSON body of `POST path` with `body`, on a server
    /// whose model directory is empty.
    async fn post(path: &str, body: Value) -> (u16, Value) {
        let dir = tempfile::tempdir().expect("temp dir");
        let state = Arc::new(super::super::handlers::test_state(dir.path()));
        let app = super::api_routes().with_state(state);
        let request = Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("request builds");
        let response = app.oneshot(request).await.expect("router answers");
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body reads");
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    #[tokio::test]
    async fn ollama_paths_answer_an_unknown_model_with_a_404() {
        let missing = json!({"error": "model 'ghost:latest' not found"});
        let user = json!([{"role": "user", "content": "hi"}]);
        let cases = [
            (
                "/api/chat",
                json!({"model": "ghost:latest", "messages": user}),
            ),
            (
                "/api/chat",
                json!({"model": "ghost:latest", "messages": []}),
            ),
            (
                "/api/generate",
                json!({"model": "ghost:latest", "prompt": "hi"}),
            ),
            ("/api/generate", json!({"model": "ghost:latest"})),
            (
                "/api/generate",
                json!({"model": "ghost:latest", "keep_alive": 0}),
            ),
            (
                "/api/embed",
                json!({"model": "ghost:latest", "input": ["a", "b"]}),
            ),
            (
                "/api/embeddings",
                json!({"model": "ghost:latest", "prompt": "a"}),
            ),
            ("/api/show", json!({"model": "ghost:latest"})),
        ];
        for (path, body) in cases {
            assert_eq!(
                post(path, body.clone()).await,
                (404, missing.clone()),
                "{path} {body}"
            );
        }
    }

    #[tokio::test]
    async fn ollama_paths_answer_bad_input_with_an_error_object() {
        let (status, body) = post("/api/chat", json!({"messages": []})).await;
        assert!((400..500).contains(&status), "{status}");
        assert!(body["error"].is_string(), "{body}");
        let (status, body) = post(
            "/api/embed",
            json!({"model": "m", "input": "a", "dimensions": 0}),
        )
        .await;
        assert_eq!(status, 400);
        assert!(body["error"].as_str().unwrap_or("").contains("dimensions"));
        let go = json!({"model": "m", "prompt": "x", "template": "{{ .Prompt }}"});
        let (status, body) = post("/api/generate", go).await;
        assert_eq!(status, 400);
        assert!(body["error"]
            .as_str()
            .unwrap_or("")
            .contains("Go templates"));
    }
}
