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
//! `graph_mode_matches_eager_on_cuda` (feature `cuda`) runs the same prompt
//! through `generate` and `generate_with_graphs` on device 0 for 8 greedy
//! tokens and requires identical ids:
//!
//! ```bash
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo test --release -p blazr --features cuda \
//!   --test qwen35_generate graph_mode_matches_eager_on_cuda -- --nocapture
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
#[cfg(feature = "cuda")]
const GRAPH_TOKENS: usize = 8;

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

/// Collect every token id from `stream`, failing on the first error.
#[cfg(feature = "cuda")]
async fn collect_ids<S>(stream: S, path: &str) -> Vec<u32>
where
    S: futures::Stream<Item = anyhow::Result<blazr::engine::GeneratedToken>>,
{
    let mut stream = std::pin::pin!(stream);
    let mut ids = Vec::new();
    while let Some(item) = stream.next().await {
        let token = item.unwrap_or_else(|e| panic!("{path}: {e}"));
        ids.push(token.token_id);
    }
    ids
}

#[cfg(feature = "cuda")]
#[tokio::test]
async fn graph_mode_matches_eager_on_cuda() {
    use boostr::{CudaDevice, CudaRuntime};

    let Some(path) = require_file() else {
        return;
    };
    if !boostr::runtime::cuda::is_cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let device = CudaDevice::new(0);

    let (model, config, tokenizer) = load_gguf_with_tokenizer::<CudaRuntime, _>(&path, &device)
        .unwrap_or_else(|e| panic!("load_gguf_with_tokenizer on {}: {e}", path.display()));
    assert_eq!(config.model_type(), "qwen35");
    assert!(model.supports_gdn_graph_mode());

    let executor = Executor::new(model, config, tokenizer, device, 64)
        .unwrap_or_else(|e| panic!("Executor::new: {e}"));

    let prompt_tokens = executor.tokenizer().encode(PROMPT);
    assert!(!prompt_tokens.is_empty(), "prompt encoded to no tokens");

    let gen_config = GenerationConfig {
        max_tokens: GRAPH_TOKENS,
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..GenerationConfig::default()
    };
    assert!(gen_config.is_greedy());

    let eager = collect_ids(executor.generate(&prompt_tokens, &gen_config), "generate").await;
    let graph = collect_ids(
        executor.generate_with_graphs(&prompt_tokens, &gen_config),
        "generate_with_graphs",
    )
    .await;

    println!("prompt {PROMPT:?} -> eager {eager:?} graph {graph:?}");
    assert_eq!(eager.len(), GRAPH_TOKENS, "eager produced {eager:?}");
    assert_eq!(graph, eager, "graph-mode ids diverge from eager");
}
