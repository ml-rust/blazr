//! Concurrent `/v1/chat/completions` over `qwen35` decode lanes.
//!
//! Starts the real HTTP surface with a [`LaneScheduler`] behind it and a
//! [`LaneEngine`] task driving the lanes, then checks two things end to end:
//!
//! - Three prompts of different lengths sent at once return exactly what each
//!   returns when it is the only request in flight.
//! - A request sent after that batch finishes still answers, so the lanes the
//!   batch used came back.
//!
//! # Fixture file
//!
//! `BLAZR_TEST_QWEN35_GGUF` is the path of a `qwen35` GGUF to run
//! against, whichever quantization it is. Unset,
//! `BOOSTR_BONSAI2_DIR` must hold `Ternary-Bonsai-2-27B-PQ2_0.gguf`, and a
//! CUDA device must be present. Either missing, the test prints one `skip:`
//! line and passes.
//!
//! ```bash
//! BLAZR_TEST_QWEN35_GGUF=/path/to/model.gguf cargo test --release -p blazr --features cuda \
//!   --test qwen35_concurrent_chat -- --nocapture
//! ```
//!
//! # Memory
//!
//! Every dense tensor in the file is decoded to F32. The lane
//! session also allocates a KV cache and GDN state at `LANES` lanes, so expect
//! peak device memory well above the file size.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use blazr::engine::{GenerationScheduler, LaneEngine, LaneScheduler, Scheduler};
use blazr::server::handlers::{ServeReplica, Serving};
use blazr::server::{api_routes, metrics, AppState};
use boostr::{CudaDevice, CudaRuntime};

/// A `qwen35` GGUF to run against, whichever quantization it is.
const ENV_GGUF: &str = "BLAZR_TEST_QWEN35_GGUF";
/// A directory holding [`PQ2_0_FILE`], used when [`ENV_GGUF`] is unset.
const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const PQ2_0_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
/// Three user messages of deliberately different token lengths.
const PROMPTS: [&str; 3] = [
    "Hi",
    "Name the capital of France.",
    "Write one short sentence about the sea and then stop talking.",
];
const MAX_TOKENS: usize = 12;
const LANES: usize = 3;
const NUM_CTX: usize = 256;
/// Polls allowed while waiting for the driver to hand every lane back.
const RELEASE_POLLS: usize = 200;

/// The PQ2_0 file, or `None` after one `skip:` line.
fn require_file() -> Option<PathBuf> {
    let path = match std::env::var(ENV_GGUF).ok().map(PathBuf::from) {
        Some(path) => path,
        None => {
            let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
                println!(
                    "skip: set {ENV_GGUF} to a qwen35 GGUF, or {ENV_DIR} to a directory holding {PQ2_0_FILE}"
                );
                return None;
            };
            dir.join(PQ2_0_FILE)
        }
    };
    if !path.is_file() {
        println!("skip: {} not found", path.display());
        return None;
    }
    Some(path)
}

/// A greedy, non-streaming chat request body for `prompt`.
fn chat_body(model: &str, prompt: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": false,
    })
}

/// Post one chat request and return `choices[0].message.content`.
async fn chat_content(client: &reqwest::Client, url: &str, model: &str, prompt: &str) -> String {
    let response = client
        .post(url)
        .json(&chat_body(model, prompt))
        .send()
        .await
        .unwrap_or_else(|e| panic!("POST {url} for prompt {prompt:?}: {e}"));
    let status = response.status();
    let body: serde_json::Value = response
        .json()
        .await
        .unwrap_or_else(|e| panic!("decoding the response for prompt {prompt:?}: {e}"));
    assert!(
        status.is_success(),
        "prompt {prompt:?} answered {status}: {body}"
    );
    // A short budget can be spent entirely inside the model's reasoning
    // block, so the answer a lane has to reproduce is both fields together.
    let message = &body["choices"][0]["message"];
    let reasoning = message["reasoning_content"].as_str().unwrap_or("");
    let content = message["content"]
        .as_str()
        .unwrap_or_else(|| panic!("no content in the response for prompt {prompt:?}: {body}"));
    format!("{reasoning}{content}")
}

#[tokio::test]
async fn concurrent_chat_requests_match_their_solo_answers() {
    let Some(path) = require_file() else {
        return;
    };
    if !boostr::runtime::cuda::is_cuda_available() {
        println!("skip: no CUDA device");
        return;
    }

    let model = path.display().to_string();
    let model_dir = path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let scheduler = Arc::new(
        Scheduler::<CudaRuntime>::new(model_dir, CudaDevice::new(0)).with_num_ctx(NUM_CTX),
    );
    let executor = scheduler
        .get_executor(&model)
        .await
        .unwrap_or_else(|e| panic!("loading {model}: {e}"));
    // Lanes serve the models that need GDN state, which is what `qwen35` is.
    assert_eq!(executor.config().model_type(), "qwen35");

    // The lane scheduler and the task that drives its lanes.
    let lane_scheduler = LaneScheduler::new(LANES).unwrap_or_else(|e| panic!("LaneScheduler: {e}"));
    let engine = LaneEngine::open(Arc::clone(&executor), Arc::clone(&lane_scheduler))
        .await
        .unwrap_or_else(|e| panic!("LaneEngine::open: {e}"));
    println!("lane decode mode: {:?}", engine.mode());
    tokio::spawn(engine.run());

    let handle = metrics::install_recorder().unwrap_or_else(|e| panic!("metrics recorder: {e}"));
    let scheduler_for_state: Arc<dyn GenerationScheduler> = lane_scheduler.clone();
    let serving = Serving::Single(ServeReplica {
        executor: Arc::clone(&executor),
        scheduler: Some(scheduler_for_state),
        device: "cuda:0".to_string(),
    });
    let state = Arc::new(AppState::new(Arc::clone(&scheduler), handle).with_serving(serving));
    assert!(
        state.route(Arc::clone(&executor), &[]).scheduler.is_some(),
        "the handlers must route through the lane scheduler, not the single-sequence path"
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap_or_else(|e| panic!("binding a test port: {e}"));
    let addr = listener
        .local_addr()
        .unwrap_or_else(|e| panic!("reading the bound port: {e}"));
    let app = api_routes().with_state(state);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let url = format!("http://{addr}/v1/chat/completions");
    let client = reqwest::Client::new();

    // One at a time: each prompt's own answer.
    let mut solo = Vec::with_capacity(PROMPTS.len());
    for prompt in PROMPTS {
        let content = chat_content(&client, &url, &model, prompt).await;
        assert!(!content.is_empty(), "prompt {prompt:?} answered nothing");
        println!("solo {prompt:?} -> {content:?}");
        solo.push(content);
    }

    // All at once: the same answers, produced in parallel lanes.
    let together = futures::future::join_all(PROMPTS.iter().map(|prompt| {
        let client = client.clone();
        let url = url.clone();
        let model = model.clone();
        async move { chat_content(&client, &url, &model, prompt).await }
    }))
    .await;
    for (i, content) in together.iter().enumerate() {
        println!("concurrent {:?} -> {content:?}", PROMPTS[i]);
        assert_eq!(
            *content, solo[i],
            "prompt {:?} answered differently when it shared the session",
            PROMPTS[i]
        );
    }

    // After the batch: the lanes it used serve the next request.
    let recycled = chat_content(&client, &url, &model, PROMPTS[1]).await;
    assert_eq!(
        recycled, solo[1],
        "the request after the batch answered differently, so a lane was not recycled cleanly"
    );

    // Every lane back, so nothing the batch admitted leaked one.
    let mut free = 0usize;
    for _ in 0..RELEASE_POLLS {
        free = executor
            .qwen35_lane_free()
            .await
            .unwrap_or_else(|e| panic!("qwen35_lane_free: {e}"));
        if free == LANES {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        free,
        LANES,
        "{} of {LANES} lanes are still held after every request finished",
        LANES - free
    );
}
