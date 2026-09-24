//! Regression test for the batched paged-decode per-row KV bound.
//!
//! `process_decode_batch` used to size every row's attention range by the
//! batch's longest sequence (`max_seq_len`), so a shorter row read past its
//! own cached keys into another row's paged KV blocks. It now passes each
//! row its own key end via `KvSeqLen::PerRow`. This test drives the real
//! `RequestScheduler` + `BatchEngine` path with two sequences of very
//! different lengths in one batch and asserts each one's generated ids
//! equal the same prompt run alone through the single-sequence path.
//!
//! # Fixture file
//!
//! `BLAZR_TEST_LLAMA_MODEL` is the directory of a plain Llama-architecture
//! SafeTensors checkpoint (a `config.json` beside `model.safetensors`),
//! NOT a qwen35 model — `BatchEngine::new` refuses any model that needs
//! GDN state. Unset, `models/llama-3.2-1b` is tried.
//! Either missing, the test prints one `skip:` line and passes.
//!
//! ```bash
//! BLAZR_TEST_LLAMA_MODEL=/path/to/llama-dir cargo test --release -p blazr \
//!   --test batch_decode_per_row -- --nocapture
//! ```
//!
//! # Memory
//!
//! Every dense tensor in the file is decoded to F32, so expect peak RSS
//! well above the file size.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use futures::StreamExt;

use blazr::loader::load_model;
use blazr::{BatchEngine, Executor, GenerationConfig, RequestScheduler};
use boostr::inference::memory::CpuBlockAllocator;
use boostr::inference::scheduler::SchedulerConfig;
use boostr::{CpuDevice, CpuRuntime};

/// A plain Llama-architecture SafeTensors checkpoint directory.
const ENV_DIR: &str = "BLAZR_TEST_LLAMA_MODEL";
/// Tried when `ENV_DIR` is unset.
const FALLBACK_DIR: &str = "models/llama-3.2-1b";
const SHORT_PROMPT: &str = "Hello";
const LONG_PROMPT: &str = "The sea is wide and the wind is cold and the boats go out at dawn \
    to fish beyond the harbour wall before the tide turns again and the gulls follow them home";
const NUM_TOKENS: usize = 5;
const NUM_CTX: usize = 256;
const BLOCK_SIZE: usize = 16;
const TOTAL_BLOCKS: usize = 64;
/// Batched-decode steps allowed before the drive loop gives up.
const STEP_LIMIT: usize = NUM_TOKENS * 8;

/// The checkpoint directory, or `None` after one `skip:` line.
fn require_dir() -> Option<PathBuf> {
    let path = std::env::var(ENV_DIR)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(FALLBACK_DIR));
    if !path.join("config.json").is_file() {
        println!(
            "skip: set {ENV_DIR} to a plain Llama-architecture SafeTensors checkpoint directory \
             (config.json beside model.safetensors), or place one at {FALLBACK_DIR}; {} has no \
             config.json",
            path.display()
        );
        return None;
    }
    Some(path)
}

/// Greedy config: temperature 0, no penalties.
fn greedy_config(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..GenerationConfig::default()
    }
}

/// The executor over the fixture, loaded once for this binary.
///
/// Paged attention and a small block pool are turned on so both the
/// single-sequence baseline and the batched path below exercise the paged
/// decode kernel; `max_batch_size` is set to the two sequences this test
/// admits together.
fn require_executor() -> Option<Arc<Executor<CpuRuntime>>> {
    let dir = require_dir()?;
    static EXECUTOR: OnceLock<Arc<Executor<CpuRuntime>>> = OnceLock::new();
    let executor = EXECUTOR.get_or_init(|| {
        let device = CpuDevice::new();
        let (model, mut config) = load_model::<CpuRuntime, _>(&dir, &device)
            .unwrap_or_else(|e| panic!("load_model on {}: {e}", dir.display()));
        assert!(
            !model.needs_gdn_state(),
            "fixture at {} needs GDN state; BatchEngine::new refuses such models — point \
             {ENV_DIR} at a plain Llama checkpoint instead",
            dir.display()
        );
        config.inference.paged_attention = true;
        config.inference.block_size = BLOCK_SIZE;
        config.inference.kv_pool_blocks = TOTAL_BLOCKS;
        config.inference.max_batch_size = 2;
        let tokenizer = blazr::tokenizer::from_pretrained("llama3")
            .unwrap_or_else(|e| panic!("from_pretrained(llama3): {e}"));
        let executor = Executor::new(model, config, tokenizer, device, NUM_CTX)
            .unwrap_or_else(|e| panic!("Executor::new: {e}"));
        Arc::new(executor)
    });
    Some(executor.clone())
}

/// Ids one prompt produces alone through the single-sequence paged path.
async fn single_sequence_ids(
    executor: &Executor<CpuRuntime>,
    prompt_tokens: &[u32],
    gen_config: &GenerationConfig,
) -> Vec<u32> {
    let mut stream = std::pin::pin!(executor.generate(prompt_tokens, gen_config));
    let mut ids = Vec::new();
    while let Some(item) = stream.next().await {
        let token = item.unwrap_or_else(|e| panic!("generate: {e}"));
        ids.push(token.token_id);
    }
    ids
}

#[tokio::test]
async fn batched_decode_matches_single_sequence_at_mixed_lengths() {
    let Some(executor) = require_executor() else {
        return;
    };

    let short = executor.tokenizer().encode(SHORT_PROMPT);
    let long = executor.tokenizer().encode(LONG_PROMPT);
    assert!(
        !short.is_empty(),
        "prompt {SHORT_PROMPT:?} encoded to no tokens"
    );
    assert!(
        !long.is_empty(),
        "prompt {LONG_PROMPT:?} encoded to no tokens"
    );
    assert!(
        long.len() > short.len() + BLOCK_SIZE,
        "the two prompts must differ by more than one paged-attention block ({BLOCK_SIZE} \
         tokens) so their block tables differ in length, got {} and {} tokens",
        short.len(),
        long.len()
    );
    assert!(
        long.len() + NUM_TOKENS < NUM_CTX,
        "prompt of {} tokens plus {NUM_TOKENS} decode steps does not fit {NUM_CTX} of context",
        long.len()
    );

    let short_config = greedy_config(NUM_TOKENS);
    let long_config = greedy_config(NUM_TOKENS);

    // Baseline: each prompt run alone, on the same paged-attention executor.
    let short_alone = single_sequence_ids(&executor, &short, &short_config).await;
    let long_alone = single_sequence_ids(&executor, &long, &long_config).await;
    assert_eq!(
        short_alone.len(),
        NUM_TOKENS,
        "single-sequence short run produced {short_alone:?}"
    );
    assert_eq!(
        long_alone.len(),
        NUM_TOKENS,
        "single-sequence long run produced {long_alone:?}"
    );

    // Now the same two prompts, admitted together into one continuous batch
    // so their decode steps run through `process_decode_batch` side by side
    // at different sequence lengths.
    let allocator = CpuBlockAllocator::new(TOTAL_BLOCKS, BLOCK_SIZE);
    let sched_config = SchedulerConfig {
        max_batch_size: 2,
        max_batch_tokens: NUM_CTX,
        max_seq_len: NUM_CTX,
        block_size: BLOCK_SIZE,
        ..Default::default()
    };
    let scheduler = RequestScheduler::new(allocator, sched_config);
    let batch_engine = BatchEngine::new(Arc::clone(&executor), Arc::clone(&scheduler))
        .unwrap_or_else(|e| panic!("BatchEngine::new: {e}"));
    tokio::spawn(async move {
        batch_engine.run().await;
    });

    let mut short_handle = scheduler
        .submit(short, short_config)
        .unwrap_or_else(|e| panic!("submit(short): {e}"));
    let mut long_handle = scheduler
        .submit(long, long_config)
        .unwrap_or_else(|e| panic!("submit(long): {e}"));

    let mut short_ids = Vec::new();
    let mut long_ids = Vec::new();
    let mut short_done = false;
    let mut long_done = false;
    let mut steps = 0usize;
    while !short_done || !long_done {
        assert!(
            steps < STEP_LIMIT,
            "batched decode did not finish within {STEP_LIMIT} steps; short {short_ids:?} \
             (done={short_done}) long {long_ids:?} (done={long_done})"
        );
        steps += 1;
        tokio::select! {
            item = short_handle.token_rx.recv(), if !short_done => {
                match item {
                    Some(Ok(token)) => {
                        short_ids.push(token.token_id);
                        if token.finish_reason.is_some() || short_ids.len() >= NUM_TOKENS {
                            short_done = true;
                        }
                    }
                    Some(Err(e)) => panic!("short lane: {e}"),
                    None => short_done = true,
                }
            }
            item = long_handle.token_rx.recv(), if !long_done => {
                match item {
                    Some(Ok(token)) => {
                        long_ids.push(token.token_id);
                        if token.finish_reason.is_some() || long_ids.len() >= NUM_TOKENS {
                            long_done = true;
                        }
                    }
                    Some(Err(e)) => panic!("long lane: {e}"),
                    None => long_done = true,
                }
            }
        }
    }

    println!(
        "batched short {short_ids:?} alone {short_alone:?}; batched long {long_ids:?} alone \
         {long_alone:?}"
    );
    assert_eq!(
        short_ids, short_alone,
        "the short sequence's batched decode diverged from its single-sequence run — a \
         shorter row is reading past its own cached keys"
    );
    assert_eq!(
        long_ids, long_alone,
        "the long sequence's batched decode diverged from its single-sequence run"
    );
}
