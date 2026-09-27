//! Chunked prefill on the `qwen35` lane path: the ids a prompt produces do
//! not depend on how its prefill was sliced, and the lanes already decoding
//! keep producing tokens while a long prompt prefills.
//!
//! Two properties are checked:
//!
//! 1. One prompt admitted through `qwen35_lane_admit_begin` plus
//!    `qwen35_lane_prefill_chunk` produces the same ids at every chunk size
//!    as the same prompt admitted through `qwen35_lane_admit` in one call,
//!    and as the same prompt run alone through `Executor::generate`. The
//!    chunk sizes cover one that divides the prompt length, one that does
//!    not, one equal to it and one past it.
//! 2. A short sequence already decoding receives more than one token while a
//!    long prompt prefills in chunks beside it, counted in tokens rather
//!    than measured in time.
//!
//! # Fixture file
//!
//! `BLAZR_TEST_QWEN35_GGUF` is the path of a `qwen35` GGUF to run
//! against, whichever quantization it is. Unset,
//! `BOOSTR_BONSAI2_DIR` must hold `Ternary-Bonsai-2-27B-PQ2_0.gguf`, and a
//! CUDA device must be present. Either missing, each test prints one `skip:`
//! line and passes.
//!
//! ```bash
//! BLAZR_TEST_QWEN35_GGUF=/path/to/model.gguf cargo test --release -p blazr --features cuda \
//!   --test qwen35_lane_prefill_interleave -- --nocapture
//! ```
//!
//! # Memory
//!
//! Every dense tensor in the file is decoded to F32. The session
//! also allocates a KV cache and GDN state at its lane count, so expect peak
//! device memory well above the file size.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use tokio::sync::Mutex;

use futures::StreamExt;

use blazr::engine::LaneDecodeMode;
use blazr::loader::load_gguf_with_tokenizer;
use blazr::{Executor, GenerationConfig};
use boostr::{CudaDevice, CudaRuntime};

/// The tests below share one loaded executor (see [`require_executor`]), and
/// its lane engine is not safe under two test bodies driving it at once, so
/// every test in this binary holds this lock for its whole run.
static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// The lock, held across the awaits of a whole test body, so it is an async
/// mutex rather than a blocking one.
async fn cuda_lock() -> tokio::sync::MutexGuard<'static, ()> {
    CUDA_LOCK.get_or_init(|| Mutex::new(())).lock().await
}

/// A `qwen35` GGUF to run against, whichever quantization it is.
const ENV_GGUF: &str = "BLAZR_TEST_QWEN35_GGUF";
/// A directory holding [`PQ2_0_FILE`], used when [`ENV_GGUF`] is unset.
const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const PQ2_0_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
/// A prompt long enough to slice several ways.
const LONG_PROMPT: &str =
    "The sea is wide and the wind is cold and the boats go out at dawn to fish beyond the \
     harbour wall before the tide turns again";
/// A prompt short beside `LONG_PROMPT`, and one a greedy decode continues
/// rather than ending at once.
const SHORT_PROMPT: &str = "The capital of France is";
const NUM_TOKENS: usize = 6;
const NUM_CTX: usize = 512;
/// Steps allowed before a drive loop gives up on a stuck lane.
const STEP_LIMIT: usize = NUM_TOKENS * 4;
/// Tokens the short sequence may decode while the long prompt prefills.
const INTERLEAVE_MAX_TOKENS: usize = 64;

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

/// The executor over the fixture, loaded once for this binary and shared by
/// both tests below (each still runs its own lane session on it, serialized
/// by [`cuda_lock`]), or `None` after one `skip:` line.
///
/// The cheap checks (`require_file`, CUDA availability) run on every call,
/// so a missing fixture is reported by whichever test asks first. Only the
/// multi-gigabyte load itself is behind the `OnceLock`, so it happens once
/// no matter how many tests ask for it.
fn require_executor() -> Option<Arc<Executor<CudaRuntime>>> {
    let path = require_file()?;
    if !boostr::runtime::cuda::is_cuda_available() {
        println!("skip: no CUDA device");
        return None;
    }
    static EXECUTOR: OnceLock<Arc<Executor<CudaRuntime>>> = OnceLock::new();
    let executor = EXECUTOR.get_or_init(|| {
        let device = CudaDevice::new(0);
        let (model, config, tokenizer) = load_gguf_with_tokenizer::<CudaRuntime, _>(&path, &device)
            .unwrap_or_else(|e| panic!("load_gguf_with_tokenizer on {}: {e}", path.display()));
        assert_eq!(config.model_type(), "qwen35");
        let executor = Executor::new(model, config, tokenizer, device, NUM_CTX)
            .unwrap_or_else(|e| panic!("Executor::new: {e}"));
        Arc::new(executor)
    });
    Some(executor.clone())
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

/// Prompt ids, asserted non-empty.
fn encode(executor: &Executor<CudaRuntime>, prompt: &str) -> Vec<u32> {
    let ids = executor.tokenizer().encode(prompt);
    assert!(!ids.is_empty(), "prompt {prompt:?} encoded to no tokens");
    ids
}

/// Ids one prompt produces alone through the single-sequence path.
async fn single_sequence_ids(
    executor: &Executor<CudaRuntime>,
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

/// Chunk sizes to admit one prompt of `prompt_len` tokens under: one that
/// divides the length, one that does not, the length itself and one past it.
fn chunk_sizes(prompt_len: usize) -> Vec<usize> {
    let divides = (2..prompt_len)
        .find(|d| prompt_len.is_multiple_of(*d))
        .unwrap_or(prompt_len);
    let ragged = (2..prompt_len)
        .find(|d| !prompt_len.is_multiple_of(*d))
        .unwrap_or(prompt_len);
    assert!(
        divides != ragged,
        "a prompt of {prompt_len} tokens gives no chunk size that divides it and none that does \
         not; lengthen LONG_PROMPT"
    );
    vec![divides, ragged, prompt_len, prompt_len + 3]
}

/// Ids one prompt produces on a one-lane session.
///
/// `chunk_size` of `None` admits the prompt through the one-shot prefill,
/// `Some(n)` through the chunked prefill at `n` tokens a chunk. The session
/// is opened, driven to the lane's finish reason or `max_tokens`, and the
/// lane is released.
async fn lane_ids(
    executor: &Executor<CudaRuntime>,
    prompt_tokens: &[u32],
    gen_config: &GenerationConfig,
    mode: LaneDecodeMode,
    chunk_size: Option<usize>,
) -> Vec<u32> {
    executor
        .qwen35_lane_open(1, mode)
        .await
        .unwrap_or_else(|e| panic!("qwen35_lane_open(1, {mode:?}): {e}"));

    let admitted = match chunk_size {
        None => executor
            .qwen35_lane_admit(prompt_tokens, gen_config)
            .await
            .unwrap_or_else(|e| panic!("qwen35_lane_admit: {e}")),
        Some(size) => {
            executor
                .qwen35_lane_admit_begin(prompt_tokens, gen_config)
                .await
                .unwrap_or_else(|e| panic!("qwen35_lane_admit_begin: {e}"));
            let mut chunks = 0usize;
            loop {
                chunks += 1;
                assert!(
                    chunks <= prompt_tokens.len() + 1,
                    "the chunked prefill ran {chunks} chunks over a prompt of {} tokens at \
                     chunk size {size}",
                    prompt_tokens.len()
                );
                let done = executor
                    .qwen35_lane_prefill_chunk(size)
                    .await
                    .unwrap_or_else(|e| panic!("qwen35_lane_prefill_chunk({size}): {e}"));
                if let Some(admitted) = done {
                    println!(
                        "chunk size {size} prefilled {} tokens in {chunks} chunks",
                        prompt_tokens.len()
                    );
                    break admitted;
                }
            }
        }
    };

    let mut ids = vec![admitted.first_token];
    let mut done = admitted.finish_reason.is_some() || ids.len() >= gen_config.max_tokens;
    let mut steps = 0usize;
    while !done {
        assert!(
            steps < STEP_LIMIT,
            "the lane did not finish within {STEP_LIMIT} steps, ids so far {ids:?}"
        );
        steps += 1;
        let produced = executor
            .qwen35_lane_step()
            .await
            .unwrap_or_else(|e| panic!("qwen35_lane_step: {e}"));
        for token in produced {
            assert_eq!(token.lane, admitted.lane, "a free lane produced a token");
            ids.push(token.token_id);
            done = token.finish_reason.is_some() || ids.len() >= gen_config.max_tokens;
        }
    }

    executor
        .qwen35_lane_release(admitted.lane)
        .await
        .unwrap_or_else(|e| panic!("qwen35_lane_release({}): {e}", admitted.lane));
    ids
}

#[tokio::test]
async fn chunked_prefill_matches_one_shot_prefill_on_cuda() {
    let _guard = cuda_lock().await;
    let Some(executor) = require_executor() else {
        return;
    };
    let prompt = encode(&executor, LONG_PROMPT);
    let greedy = greedy_config(NUM_TOKENS);
    assert!(
        prompt.len() + NUM_TOKENS < NUM_CTX,
        "prompt of {} tokens plus {NUM_TOKENS} decode steps does not fit {NUM_CTX} of context",
        prompt.len()
    );

    let alone = single_sequence_ids(&executor, &prompt, &greedy).await;
    assert_eq!(
        alone.len(),
        NUM_TOKENS,
        "the single-sequence run produced {alone:?}"
    );

    for mode in [LaneDecodeMode::Eager, LaneDecodeMode::Graph] {
        let one_shot = lane_ids(&executor, &prompt, &greedy, mode, None).await;
        assert_eq!(
            one_shot, alone,
            "{mode:?} one-shot prefill diverged from the single-sequence run"
        );
        for size in chunk_sizes(prompt.len()) {
            let chunked = lane_ids(&executor, &prompt, &greedy, mode, Some(size)).await;
            println!("{mode:?} chunk size {size}: {chunked:?} one-shot {one_shot:?}");
            assert_eq!(
                chunked,
                one_shot,
                "{mode:?} prefill in chunks of {size} over a prompt of {} tokens diverged from \
                 the one-shot prefill",
                prompt.len()
            );
        }
    }

    executor.qwen35_lane_close().await;
}

#[tokio::test]
async fn a_decoding_lane_keeps_stepping_while_a_prompt_prefills_on_cuda() {
    let _guard = cuda_lock().await;
    let Some(executor) = require_executor() else {
        return;
    };
    let short = encode(&executor, SHORT_PROMPT);
    let long = encode(&executor, LONG_PROMPT);
    // Two chunks leave one step, so the count below needs at least three.
    let chunk_size = 2;
    let chunks = long.len().div_ceil(chunk_size);
    assert!(
        chunks >= 3,
        "a prompt of {} tokens in chunks of {chunk_size} gives {chunks} chunks; the interleave \
         needs at least 3",
        long.len()
    );

    let mode = LaneDecodeMode::Eager;
    executor
        .qwen35_lane_open(2, mode)
        .await
        .unwrap_or_else(|e| panic!("qwen35_lane_open(2, {mode:?}): {e}"));

    // The short sequence decodes for longer than the prefill takes chunks,
    // so it is still live when the last chunk lands.
    let decoding = greedy_config(INTERLEAVE_MAX_TOKENS);
    assert!(
        INTERLEAVE_MAX_TOKENS > chunks,
        "the decoding lane must outlast the {chunks} chunks of the prefill"
    );
    let short_lane = executor
        .qwen35_lane_admit(&short, &decoding)
        .await
        .unwrap_or_else(|e| panic!("qwen35_lane_admit(short): {e}"));

    executor
        .qwen35_lane_admit_begin(&long, &greedy_config(NUM_TOKENS))
        .await
        .unwrap_or_else(|e| panic!("qwen35_lane_admit_begin(long): {e}"));

    // One step between chunks, exactly as the lane engine's driver runs it.
    // The short lane is released as soon as it reports a finish reason, and
    // the remaining chunks then prefill on their own.
    let mut short_tokens = 0usize;
    let mut short_live = true;
    let mut steps = 0usize;
    let long_lane = loop {
        let done = executor
            .qwen35_lane_prefill_chunk(chunk_size)
            .await
            .unwrap_or_else(|e| panic!("qwen35_lane_prefill_chunk({chunk_size}): {e}"));
        if let Some(admitted) = done {
            break admitted;
        }
        if !short_live {
            continue;
        }
        steps += 1;
        let produced = executor
            .qwen35_lane_step()
            .await
            .unwrap_or_else(|e| panic!("qwen35_lane_step: {e}"));
        for token in produced {
            assert_eq!(
                token.lane, short_lane.lane,
                "lane {} produced a token while only the short sequence was admitted",
                token.lane
            );
            short_tokens += 1;
            if token.finish_reason.is_some() {
                short_live = false;
            }
        }
        if !short_live {
            executor
                .qwen35_lane_release(short_lane.lane)
                .await
                .unwrap_or_else(|e| panic!("qwen35_lane_release({}): {e}", short_lane.lane));
        }
    };

    println!(
        "the short lane took {short_tokens} tokens over {steps} steps while {} prompt tokens \
         prefilled in chunks of {chunk_size}",
        long.len()
    );
    assert!(
        short_tokens > 1,
        "the short lane took {short_tokens} tokens while a prompt of {} tokens prefilled in \
         chunks of {chunk_size}; the prefill did not interleave",
        long.len()
    );
    if short_live {
        assert_ne!(
            long_lane.lane, short_lane.lane,
            "the prefilled prompt took the lane the short sequence still holds"
        );
    }

    if short_live {
        executor
            .qwen35_lane_release(short_lane.lane)
            .await
            .unwrap_or_else(|e| panic!("qwen35_lane_release({}): {e}", short_lane.lane));
    }
    executor
        .qwen35_lane_release(long_lane.lane)
        .await
        .unwrap_or_else(|e| panic!("qwen35_lane_release({}): {e}", long_lane.lane));
    executor.qwen35_lane_close().await;
}
