//! API integration tests for blazr server endpoints.
//!
//! ## Test tiers
//!
//! **Unit tests** — always run, no server or model needed.
//! Test config defaults, type invariants, chat templates, parameter validation.
//!
//! **Live server tests** — require env vars, test real API endpoints.
//! Set these environment variables to enable:
//!   - `BLAZR_TEST_SERVER` — server URL (e.g. `http://localhost:8080`)
//!   - `BLAZR_TEST_MODEL`  — model path as loaded on that server (e.g. `/home/user/models/llama-3.2-1b`)
//!
//! Run: `BLAZR_TEST_SERVER=http://localhost:8080 BLAZR_TEST_MODEL=/path/to/model cargo test -- --ignored`

// ── Helpers ──

/// Get the test server URL from env, or skip the test.
fn test_server() -> String {
    std::env::var("BLAZR_TEST_SERVER").expect("BLAZR_TEST_SERVER env var required for live tests")
}

/// Get the test model identifier from env, or skip the test.
fn test_model() -> String {
    std::env::var("BLAZR_TEST_MODEL").expect("BLAZR_TEST_MODEL env var required for model tests")
}

// ═══════════════════════════════════════════════════════════
// Unit tests — always run, no server needed
// ═══════════════════════════════════════════════════════════

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

#[test]
fn test_chat_template_from_name() {
    use blazr::ChatTemplate;
    assert_eq!(ChatTemplate::from_name("llama3"), ChatTemplate::Llama3);
    assert_eq!(ChatTemplate::from_name("chatml"), ChatTemplate::ChatML);
    assert_eq!(
        ChatTemplate::from_name("MISTRAL"),
        ChatTemplate::MistralInstruct
    );
    assert_eq!(ChatTemplate::from_name("phi3"), ChatTemplate::Phi3);
    assert_eq!(ChatTemplate::from_name("gemma"), ChatTemplate::Gemma);
    assert_eq!(ChatTemplate::from_name("deepseek"), ChatTemplate::DeepSeek);
    assert_eq!(ChatTemplate::from_name("unknown"), ChatTemplate::Generic);
}

#[test]
fn test_seed_does_not_force_greedy() {
    // Seed with temperature > 0 should use seeded stochastic sampling, not greedy
    let config = blazr::GenerationConfig {
        temperature: 0.8,
        seed: Some(42),
        ..Default::default()
    };
    assert!(
        !config.is_greedy(),
        "seed with temperature > 0 should not force greedy"
    );

    // Temperature 0 is still greedy regardless of seed
    let greedy_config = blazr::GenerationConfig {
        temperature: 0.0,
        seed: Some(42),
        ..Default::default()
    };
    assert!(greedy_config.is_greedy());

    let no_seed = blazr::GenerationConfig {
        temperature: 0.8,
        ..Default::default()
    };
    assert!(!no_seed.is_greedy());
}

#[test]
fn test_logit_bias_in_config() {
    use std::collections::HashMap;
    let mut bias = HashMap::new();
    bias.insert(100u32, 5.0f32);
    bias.insert(200, -10.0);
    let config = blazr::GenerationConfig {
        logit_bias: bias,
        ..Default::default()
    };
    assert_eq!(config.logit_bias.len(), 2);
    assert_eq!(config.logit_bias[&100], 5.0);
    assert_eq!(config.logit_bias[&200], -10.0);
}

#[test]
fn test_mirostat_config() {
    let config = blazr::GenerationConfig {
        mirostat_mode: 2,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        ..Default::default()
    };
    assert_eq!(config.mirostat_mode, 2);
    assert_eq!(config.mirostat_tau, 5.0);
    assert_eq!(config.mirostat_eta, 0.1);
}

#[test]
fn test_dynatemp_config() {
    let config = blazr::GenerationConfig {
        temperature: 0.8,
        dynatemp_range: 0.3,
        dynatemp_exponent: 1.5,
        ..Default::default()
    };
    assert_eq!(config.dynatemp_range, 0.3);
    assert_eq!(config.dynatemp_exponent, 1.5);
    assert!(!config.is_greedy());
}

#[test]
fn test_logprobs_config() {
    let config = blazr::GenerationConfig {
        logprobs: true,
        top_logprobs: 10,
        ..Default::default()
    };
    assert!(config.logprobs);
    assert_eq!(config.top_logprobs, 10);
}

#[test]
fn test_json_mode_config() {
    let config = blazr::GenerationConfig {
        json_mode: true,
        ..Default::default()
    };
    assert!(config.json_mode);
}

#[test]
fn test_dry_sampling_config() {
    let config = blazr::GenerationConfig {
        dry_multiplier: 1.0,
        dry_base: 3,
        dry_allowed_length: 128,
        ..Default::default()
    };
    assert_eq!(config.dry_multiplier, 1.0);
    assert_eq!(config.dry_base, 3);
    assert_eq!(config.dry_allowed_length, 128);
    assert!(config.dry_sequence_breakers.is_empty());
}

#[test]
fn test_typical_sampling_config() {
    let config = blazr::GenerationConfig {
        typical_p: 0.95,
        ..Default::default()
    };
    assert_eq!(config.typical_p, 0.95);
    // typical_p = 0 by default (disabled)
    let default_config = blazr::GenerationConfig::default();
    assert_eq!(default_config.typical_p, 0.0);
}

// ═══════════════════════════════════════════════════════════
// Live server tests — require BLAZR_TEST_SERVER env var
// ═══════════════════════════════════════════════════════════

#[tokio::test]
#[ignore = "requires BLAZR_TEST_SERVER"]
async fn test_health_endpoint() {
    let server = test_server();
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/health", server))
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
#[ignore = "requires BLAZR_TEST_SERVER"]
async fn test_list_models() {
    let server = test_server();
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/v1/models", server))
        .send()
        .await
        .expect("list models request failed");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array());
}

#[tokio::test]
#[ignore = "requires BLAZR_TEST_SERVER"]
async fn test_invalid_model_returns_404() {
    let server = test_server();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/completions", server))
        .json(&serde_json::json!({
            "model": "nonexistent-model-xyz-99999",
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
#[ignore = "requires BLAZR_TEST_SERVER"]
async fn test_invalid_temperature_returns_400() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/completions", server))
        .json(&serde_json::json!({
            "model": model,
            "prompt": "Hello",
            "temperature": 5.0
        }))
        .send()
        .await
        .expect("completions request failed");
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
#[ignore = "requires BLAZR_TEST_SERVER"]
async fn test_empty_messages_returns_400() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", server))
        .json(&serde_json::json!({
            "model": model,
            "messages": []
        }))
        .send()
        .await
        .expect("chat completions request failed");
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
#[ignore = "requires BLAZR_TEST_SERVER"]
async fn test_metrics_endpoint() {
    let server = test_server();
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/metrics", server))
        .send()
        .await
        .expect("metrics request failed");
    assert_eq!(resp.status(), 200);

    let text = resp.text().await.unwrap();
    assert!(
        text.contains("blazr_requests_total"),
        "Should contain request counter"
    );
    assert!(
        text.contains("blazr_models_loaded"),
        "Should contain models gauge"
    );
}

// ═══════════════════════════════════════════════════════════
// Model-specific tests — require BLAZR_TEST_SERVER + BLAZR_TEST_MODEL
// ═══════════════════════════════════════════════════════════

#[tokio::test]
#[ignore = "requires BLAZR_TEST_SERVER and BLAZR_TEST_MODEL"]
async fn test_completions_non_streaming() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/completions", server))
        .json(&serde_json::json!({
            "model": model,
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
#[ignore = "requires BLAZR_TEST_SERVER and BLAZR_TEST_MODEL"]
async fn test_completions_streaming_done_marker() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/completions", server))
        .json(&serde_json::json!({
            "model": model,
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
#[ignore = "requires BLAZR_TEST_SERVER and BLAZR_TEST_MODEL"]
async fn test_chat_completions_non_streaming() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", server))
        .json(&serde_json::json!({
            "model": model,
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
#[ignore = "requires BLAZR_TEST_SERVER and BLAZR_TEST_MODEL"]
async fn test_chat_completions_streaming() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/chat/completions", server))
        .json(&serde_json::json!({
            "model": model,
            "messages": [
                {"role": "user", "content": "Say hi."}
            ],
            "max_tokens": 5,
            "temperature": 0.0,
            "stream": true
        }))
        .send()
        .await
        .expect("chat streaming request failed");
    assert_eq!(resp.status(), 200);

    let text = resp.text().await.unwrap();
    assert!(
        text.contains("chat.completion.chunk"),
        "Should contain chunk objects"
    );
    assert!(
        text.contains("\"role\":\"assistant\""),
        "First chunk should set role"
    );
    assert!(text.contains("[DONE]"), "Stream must end with [DONE]");
}

#[tokio::test]
#[ignore = "requires BLAZR_TEST_SERVER and BLAZR_TEST_MODEL"]
async fn test_tokenize_endpoint() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/tokenize", server))
        .json(&serde_json::json!({
            "model": model,
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
#[ignore = "requires BLAZR_TEST_SERVER and BLAZR_TEST_MODEL"]
async fn test_detokenize_endpoint() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    // First tokenize
    let tok_resp = client
        .post(format!("{}/tokenize", server))
        .json(&serde_json::json!({
            "model": model,
            "content": "Hello world"
        }))
        .send()
        .await
        .unwrap();
    let tok_body: serde_json::Value = tok_resp.json().await.unwrap();
    let tokens = tok_body["tokens"].clone();

    // Then detokenize
    let resp = client
        .post(format!("{}/detokenize", server))
        .json(&serde_json::json!({
            "model": model,
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
#[ignore = "requires BLAZR_TEST_SERVER and BLAZR_TEST_MODEL"]
async fn test_stop_sequence() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/completions", server))
        .json(&serde_json::json!({
            "model": model,
            "prompt": "Count from 1 to 10: 1, 2, 3, 4, 5,",
            "max_tokens": 50,
            "temperature": 0.0,
            "stop": [" 8"]
        }))
        .send()
        .await
        .expect("stop sequence request failed");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    let text = body["choices"][0]["text"].as_str().unwrap();
    assert!(
        !text.contains(" 8"),
        "Output should not contain the stop sequence ' 8'"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
#[ignore = "requires BLAZR_TEST_SERVER and BLAZR_TEST_MODEL"]
async fn test_concurrent_requests() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    let mut handles = Vec::new();

    for _ in 0..3 {
        let c = client.clone();
        let s = server.clone();
        let m = model.clone();
        handles.push(tokio::spawn(async move {
            c.post(format!("{}/v1/completions", s))
                .json(&serde_json::json!({
                    "model": m,
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

#[tokio::test]
#[ignore = "requires BLAZR_TEST_SERVER and BLAZR_TEST_MODEL"]
async fn test_streaming_disconnect_no_crash() {
    let server = test_server();
    let model = test_model();
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/completions", server))
        .json(&serde_json::json!({
            "model": model,
            "prompt": "Tell me a very long story about dragons",
            "max_tokens": 100,
            "stream": true
        }))
        .send()
        .await
        .expect("streaming request failed");
    assert_eq!(resp.status(), 200);

    // Read just one byte and drop — simulates client disconnect
    drop(resp);

    // Give server a moment to handle the disconnect
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Verify server is still healthy
    let health = client
        .get(format!("{}/health", server))
        .send()
        .await
        .expect("health check after disconnect failed");
    assert_eq!(health.status(), 200);
}
