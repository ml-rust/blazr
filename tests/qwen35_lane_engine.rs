//! Multi-lane `qwen35` decode through `Qwen35LaneSession`, checked against the
//! single-sequence path.
//!
//! Three prompts of different lengths are admitted into one session and
//! stepped together. Each lane's generated ids must equal the ids the same
//! prompt produces alone through `Executor::generate` under the same
//! `GenerationConfig`. Both decode paths are covered — the eager batched
//! forward and the captured multi-lane graph — and one case runs a greedy
//! lane and a sampled lane in the same step.
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
//!   --test qwen35_lane_engine -- --nocapture
//! ```
//!
//! # Memory
//!
//! Every dense tensor in the file is decoded to F32. The session
//! also allocates a KV cache and GDN state at `LANES` lanes, so expect peak
//! device memory well above the file size.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use futures::StreamExt;

use blazr::engine::LaneDecodeMode;
use blazr::loader::load_gguf_with_tokenizer;
use blazr::{Executor, GenerationConfig};
use boostr::{CudaDevice, CudaRuntime};

/// A `qwen35` GGUF to run against, whichever quantization it is.
const ENV_GGUF: &str = "BLAZR_TEST_QWEN35_GGUF";
/// A directory holding [`PQ2_0_FILE`], used when [`ENV_GGUF`] is unset.
const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const PQ2_0_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
/// Three prompts of deliberately different token lengths.
const PROMPTS: [&str; 3] = [
    "Hello",
    "The capital of France is",
    "Write one short sentence about the sea and then stop",
];
const NUM_TOKENS: usize = 6;
const LANES: usize = 3;
const NUM_CTX: usize = 128;
/// Steps allowed before the drive loop gives up on a stuck lane.
const STEP_LIMIT: usize = NUM_TOKENS * 4;

/// The PQ2_0 file, or `None` after one `skip:` line.
fn require_file() -> Option<PathBuf> {
    let path = match std::env::var(ENV_GGUF).ok().map(PathBuf::from) {
        Some(path) => path,
        None => {
            let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
                println!("skip: set {ENV_GGUF} to a qwen35 GGUF, or {ENV_DIR} to a directory holding {PQ2_0_FILE}");
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

/// Greedy config: temperature 0, no penalties.
fn greedy_config() -> GenerationConfig {
    GenerationConfig {
        max_tokens: NUM_TOKENS,
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..GenerationConfig::default()
    }
}

/// Sampled config with a fixed seed, so the ids are reproducible.
fn sampled_config() -> GenerationConfig {
    GenerationConfig {
        max_tokens: NUM_TOKENS,
        temperature: 0.9,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.0,
        seed: Some(20_250_924),
        ..GenerationConfig::default()
    }
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

/// Admit every `(prompt, config)` pair into one session, step them together,
/// and return each pair's generated ids in the order the pairs were given.
///
/// A lane is released as soon as it reports a finish reason or reaches its own
/// `max_tokens`, so a finished lane stops decoding while the others carry on.
async fn lane_ids(
    executor: &Executor<CudaRuntime>,
    work: &[(Vec<u32>, GenerationConfig)],
    mode: LaneDecodeMode,
) -> Vec<Vec<u32>> {
    executor
        .qwen35_lane_open(LANES, mode)
        .await
        .unwrap_or_else(|e| panic!("qwen35_lane_open({LANES}, {mode:?}): {e}"));

    let mut ids: Vec<Vec<u32>> = vec![Vec::new(); work.len()];
    // Lane index -> position in `work`.
    let mut lane_of: Vec<Option<usize>> = vec![None; LANES];
    let mut live = 0usize;

    for (slot, (prompt_tokens, gen_config)) in work.iter().enumerate() {
        let admitted = executor
            .qwen35_lane_admit(prompt_tokens, gen_config)
            .await
            .unwrap_or_else(|e| panic!("qwen35_lane_admit(slot {slot}): {e}"));
        ids[slot].push(admitted.first_token);
        if admitted.finish_reason.is_some() || ids[slot].len() >= gen_config.max_tokens {
            executor
                .qwen35_lane_release(admitted.lane)
                .await
                .unwrap_or_else(|e| panic!("qwen35_lane_release({}): {e}", admitted.lane));
        } else {
            lane_of[admitted.lane] = Some(slot);
            live += 1;
        }
    }

    let mut steps = 0usize;
    while live > 0 {
        assert!(
            steps < STEP_LIMIT,
            "lane decode did not finish within {STEP_LIMIT} steps; {live} lanes still live"
        );
        steps += 1;

        let produced = executor
            .qwen35_lane_step()
            .await
            .unwrap_or_else(|e| panic!("qwen35_lane_step: {e}"));
        let mut done = Vec::new();
        for token in produced {
            let Some(slot) = lane_of[token.lane] else {
                panic!("lane {} produced a token with no request", token.lane);
            };
            ids[slot].push(token.token_id);
            let max_tokens = work[slot].1.max_tokens;
            if token.finish_reason.is_some() || ids[slot].len() >= max_tokens {
                done.push(token.lane);
            }
        }
        for lane in done {
            executor
                .qwen35_lane_release(lane)
                .await
                .unwrap_or_else(|e| panic!("qwen35_lane_release({lane}): {e}"));
            lane_of[lane] = None;
            live -= 1;
        }
    }

    ids
}

/// One greedy lane and one sampled lane in the same step, each matching its
/// own single-sequence run.
async fn check_two_sampling_policies(executor: &Executor<CudaRuntime>, prompts: &[Vec<u32>]) {
    let greedy_prompt = prompts[0].clone();
    let sampled_prompt = prompts[2].clone();
    let greedy = greedy_config();
    let sampled = sampled_config();
    assert!(greedy.is_greedy());
    assert!(!sampled.is_greedy());

    let greedy_alone = single_sequence_ids(executor, &greedy_prompt, &greedy).await;
    let sampled_alone = single_sequence_ids(executor, &sampled_prompt, &sampled).await;
    let sampled_prompt_greedy = single_sequence_ids(executor, &sampled_prompt, &greedy).await;
    if sampled_alone == sampled_prompt_greedy {
        println!(
            "note: the sampled config picked the greedy ids for prompt {:?}, so this case does \
             not separate the two policies",
            PROMPTS[2]
        );
    }

    let work = vec![(greedy_prompt, greedy), (sampled_prompt, sampled)];
    for mode in [LaneDecodeMode::Eager, LaneDecodeMode::Graph] {
        let lanes = lane_ids(executor, &work, mode).await;
        println!(
            "{mode:?} greedy lane {:?} alone {greedy_alone:?}; sampled lane {:?} alone \
             {sampled_alone:?}",
            lanes[0], lanes[1]
        );
        assert_eq!(
            lanes[0], greedy_alone,
            "{mode:?} greedy lane diverged from its single-sequence run"
        );
        assert_eq!(
            lanes[1], sampled_alone,
            "{mode:?} sampled lane diverged from its single-sequence run"
        );
    }
}

#[tokio::test]
async fn lanes_match_the_single_sequence_path_on_cuda() {
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

    let executor = Executor::new(model, config, tokenizer, device, NUM_CTX)
        .unwrap_or_else(|e| panic!("Executor::new: {e}"));

    let prompts: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| {
            let ids = executor.tokenizer().encode(p);
            assert!(!ids.is_empty(), "prompt {p:?} encoded to no tokens");
            ids
        })
        .collect();
    let lengths: Vec<usize> = prompts.iter().map(Vec::len).collect();
    assert!(
        lengths[0] != lengths[1] && lengths[1] != lengths[2] && lengths[0] != lengths[2],
        "the three prompts must differ in length, got {lengths:?}"
    );

    let greedy = greedy_config();
    let baselines: Vec<Vec<u32>> = {
        let mut out = Vec::new();
        for prompt in &prompts {
            out.push(single_sequence_ids(&executor, prompt, &greedy).await);
        }
        out
    };
    for (i, ids) in baselines.iter().enumerate() {
        assert_eq!(
            ids.len(),
            NUM_TOKENS,
            "single-sequence prompt {i} produced {ids:?}"
        );
    }

    let work: Vec<(Vec<u32>, GenerationConfig)> = prompts
        .iter()
        .map(|p| (p.clone(), greedy.clone()))
        .collect();

    for mode in [LaneDecodeMode::Eager, LaneDecodeMode::Graph] {
        let lanes = lane_ids(&executor, &work, mode).await;
        for (i, ids) in lanes.iter().enumerate() {
            println!("{mode:?} prompt {i}: lane {ids:?} alone {:?}", baselines[i]);
            assert_eq!(
                *ids, baselines[i],
                "{mode:?} lane ids diverge from the single-sequence path for prompt {i}"
            );
        }
    }

    check_two_sampling_policies(&executor, &prompts).await;
    executor.qwen35_lane_close().await;
}
