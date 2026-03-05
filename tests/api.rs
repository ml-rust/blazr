//! API integration tests for blazr server endpoints.
//!
//! Tests that don't require a GPU or loaded model test config, types, and structure.
//!
//! Tests marked `#[ignore]` require a running blazr server with test models.
//! Start server: `cargo run --features cuda -- serve --model llama-3.2-1b --port 8090`
//! Run them with: `cargo test -- --ignored`

/// Verify GenerationConfig defaults match OpenAI-compatible expectations
#[test]
fn test_generation_config_defaults() {
    let config = blazr::GenerationConfig::default();
    assert_eq!(config.max_tokens, 2048);
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.top_p, 1.0);
    assert_eq!(config.top_k, 0);
    assert!((config.min_p - 0.05).abs() < f32::EPSILON);
    assert!((config.repeat_penalty - 1.1).abs() < f32::EPSILON);
    assert_eq!(config.frequency_penalty, 0.0);
    assert_eq!(config.presence_penalty, 0.0);
    assert!(config.stop_sequences.is_empty());
    assert!(config.seed.is_none());
}

#[test]
fn test_greedy_config() {
    let config = blazr::GenerationConfig::greedy();
    assert!(config.is_greedy());
    assert_eq!(config.temperature, 0.0);
}

#[test]
fn test_generation_config_penalties() {
    let config = blazr::GenerationConfig::default();
    assert!(config.has_penalties()); // repeat_penalty defaults to 1.1

    let no_penalty = blazr::GenerationConfig {
        repeat_penalty: 1.0,
        ..Default::default()
    };
    assert!(!no_penalty.has_penalties());
}

#[test]
fn test_finish_reason_strings() {
    use blazr::engine::FinishReason;
    assert_eq!(FinishReason::Eos.as_str(), "stop");
    assert_eq!(FinishReason::Length.as_str(), "length");
    assert_eq!(FinishReason::Stop.as_str(), "stop");
}

#[test]
fn test_server_config_defaults() {
    let config = blazr::ServerConfig::default();
    assert_eq!(config.port, 8080);
    assert_eq!(config.host, "0.0.0.0");
    assert_eq!(config.max_concurrent_requests, 16);
    assert_eq!(config.request_timeout_secs, 300);
    assert!(config.cors_enabled);
    assert!(config.cors_origins.is_empty());
    assert_eq!(config.max_body_size, 10 * 1024 * 1024);
    assert_eq!(config.addr(), "0.0.0.0:8080");
}

// ── Live server integration tests ──
// These require a running blazr server at localhost:8090.
// Start: cargo run --features cuda -- serve --model llama-3.2-1b --port 8090

const TEST_SERVER: &str = "http://localhost:8090";

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090"]
async fn test_health_endpoint() {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/health", TEST_SERVER))
        .send()
        .await
        .expect("health request failed");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert!(body["version"].is_string());
    assert!(body["loaded_models"].is_array());
}

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090"]
async fn test_list_models() {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/v1/models", TEST_SERVER))
        .send()
        .await
        .expect("list models request failed");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array());
}

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090"]
async fn test_invalid_model_returns_404() {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/completions", TEST_SERVER))
        .json(&serde_json::json!({
            "model": "nonexistent-model-xyz",
            "prompt": "Hello",
            "max_tokens": 5
        }))
        .send()
        .await
        .expect("completions request failed");
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("not found"));
}

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090"]
async fn test_invalid_temperature_returns_400() {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/completions", TEST_SERVER))
        .json(&serde_json::json!({
            "model": "llama-3.2-1b",
            "prompt": "Hello",
            "temperature": 5.0
        }))
        .send()
        .await
        .expect("completions request failed");
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090 with llama-3.2-1b"]
async fn test_completions_non_streaming() {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/completions", TEST_SERVER))
        .json(&serde_json::json!({
            "model": "llama-3.2-1b",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.0
        }))
        .send()
        .await
        .expect("completions request failed");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "text_completion");
    assert!(!body["choices"][0]["text"].as_str().unwrap().is_empty());
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(body["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    assert!(body["usage"]["total_tokens"].as_u64().unwrap() > 0);
    let reason = body["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(reason == "stop" || reason == "length");
}

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090 with llama-3.2-1b"]
async fn test_completions_streaming_done_marker() {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/completions", TEST_SERVER))
        .json(&serde_json::json!({
            "model": "llama-3.2-1b",
            "prompt": "Hello",
            "max_tokens": 5,
            "temperature": 0.0,
            "stream": true
        }))
        .send()
        .await
        .expect("streaming request failed");
    assert_eq!(resp.status(), 200);

    let text = resp.text().await.unwrap();
    assert!(
        text.contains("[DONE]"),
        "Stream must end with [DONE] marker"
    );
}

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090 with llama-3.2-1b"]
async fn test_chat_completions_non_streaming() {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", TEST_SERVER))
        .json(&serde_json::json!({
            "model": "llama-3.2-1b",
            "messages": [
                {"role": "user", "content": "Say hello in one word."}
            ],
            "max_tokens": 5,
            "temperature": 0.0
        }))
        .send()
        .await
        .expect("chat completions request failed");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(!body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .is_empty());
    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090"]
async fn test_empty_messages_returns_400() {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", TEST_SERVER))
        .json(&serde_json::json!({
            "model": "llama-3.2-1b",
            "messages": []
        }))
        .send()
        .await
        .expect("chat completions request failed");
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090 with llama-3.2-1b"]
async fn test_tokenize_endpoint() {
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/tokenize", TEST_SERVER))
        .json(&serde_json::json!({
            "model": "llama-3.2-1b",
            "content": "Hello world"
        }))
        .send()
        .await
        .expect("tokenize request failed");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["tokens"].as_array().unwrap().len() >= 2);
}

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090 with llama-3.2-1b"]
async fn test_detokenize_endpoint() {
    let client = reqwest::Client::new();
    // First tokenize
    let tok_resp = client
        .post(format!("{}/tokenize", TEST_SERVER))
        .json(&serde_json::json!({
            "model": "llama-3.2-1b",
            "content": "Hello world"
        }))
        .send()
        .await
        .unwrap();
    let tok_body: serde_json::Value = tok_resp.json().await.unwrap();
    let tokens = tok_body["tokens"].clone();

    // Then detokenize
    let resp = client
        .post(format!("{}/detokenize", TEST_SERVER))
        .json(&serde_json::json!({
            "model": "llama-3.2-1b",
            "tokens": tokens
        }))
        .send()
        .await
        .expect("detokenize request failed");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["content"], "Hello world");
}

#[tokio::test]
#[ignore = "requires running blazr server at localhost:8090 with llama-3.2-1b"]
async fn test_concurrent_requests() {
    let client = reqwest::Client::new();
    let mut handles = Vec::new();

    for _ in 0..3 {
        let c = client.clone();
        handles.push(tokio::spawn(async move {
            c.post(format!("{}/v1/completions", TEST_SERVER))
                .json(&serde_json::json!({
                    "model": "llama-3.2-1b",
                    "prompt": "Count: 1, 2, 3,",
                    "max_tokens": 3,
                    "temperature": 0.0
                }))
                .send()
                .await
        }));
    }

    for handle in handles {
        let resp = handle.await.unwrap().unwrap();
        assert_eq!(resp.status(), 200);
    }
}
