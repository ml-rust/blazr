//! End-to-end `qwen35` through blazr's loader and executor on CPU: a real
//! Ternary-Bonsai-2-27B `PQ2_0` GGUF, its embedded tokenizer, 4 greedy
//! decode steps from the prompt "Hello".
//!
//! # Fixture file
//!
//! `BOOSTR_BONSAI2_DIR` must hold `Ternary-Bonsai-2-27B-PQ2_0.gguf`. Unset or
//! missing, the test prints one `skip:` line and passes.
//!
//! ```bash
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo test --release -p blazr \
//!   --test qwen35_generate -- --nocapture
//! ```
//!
//! # Memory
//!
//! The file is 7.2 GB and every dense tensor is decoded to F32, so expect
//! peak RSS above the file size.

use std::path::PathBuf;

use futures::StreamExt;

use blazr::loader::load_gguf_with_tokenizer;
use blazr::{Executor, GenerationConfig};
use boostr::{CpuDevice, CpuRuntime};

const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const PQ2_0_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const PROMPT: &str = "Hello";
const NUM_TOKENS: usize = 4;

/// The PQ2_0 file, or `None` after one `skip:` line.
fn require_file() -> Option<PathBuf> {
    let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return None;
    };
    let path = dir.join(PQ2_0_FILE);
    if !path.is_file() {
        println!("skip: {} not found", path.display());
        return None;
    }
    Some(path)
}

#[tokio::test]
async fn generates_four_tokens_greedily_on_cpu() {
    let Some(path) = require_file() else {
        return;
    };
    let device = CpuDevice::new();

    let (model, config, tokenizer) = load_gguf_with_tokenizer::<CpuRuntime, _>(&path, &device)
        .unwrap_or_else(|e| panic!("load_gguf_with_tokenizer on {}: {e}", path.display()));
    assert_eq!(config.model_type(), "qwen35");
    assert!(model.needs_gdn_state());
    assert_eq!(model.num_attention_layers(), Some(16));
    assert_eq!(model.num_gdn_layers(), Some(48));

    let executor = Executor::new(model, config, tokenizer, device, 64)
        .unwrap_or_else(|e| panic!("Executor::new: {e}"));

    let prompt_tokens = executor.tokenizer().encode(PROMPT);
    assert!(!prompt_tokens.is_empty(), "prompt encoded to no tokens");

    let gen_config = GenerationConfig {
        max_tokens: NUM_TOKENS,
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..GenerationConfig::default()
    };
    assert!(gen_config.is_greedy());

    let mut stream = std::pin::pin!(executor.generate(&prompt_tokens, &gen_config));
    let mut ids = Vec::new();
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        let token = item.unwrap_or_else(|e| panic!("generate: {e}"));
        ids.push(token.token_id);
        text.push_str(&token.text);
    }

    println!("prompt {PROMPT:?} -> ids {ids:?} text {text:?}");
    assert_eq!(
        ids.len(),
        NUM_TOKENS,
        "expected {NUM_TOKENS} tokens, got {ids:?}"
    );
    let decoded = executor
        .tokenizer()
        .decode(&ids)
        .unwrap_or_else(|e| panic!("decode {ids:?}: {e}"));
    assert!(!decoded.is_empty(), "decoded text is empty for ids {ids:?}");
}
