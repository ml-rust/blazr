//! Greedy decode of real checkpoints against Hugging Face `transformers`.
//!
//! Each test loads one SafeTensors checkpoint on the CPU in F32. It checks
//! the detected config and the prompt ids. It decodes 32 greedy tokens through
//! `Executor::generate` and asserts the first 8 ids.
//!
//! The reference ids come from `transformers` on the CPU in F32, with
//! `add_special_tokens=True`, greedy, no repeat penalty. Tiny random-weight
//! checkpoints also compare the last prompt position's first logits, since
//! their text carries no meaning.
//!
//! # Fixtures
//!
//! Each test reads its own variable, else `$BOOSTR_MODELS_DIR/<entry>`. A
//! test with neither prints one skip line and passes. The tests load one
//! checkpoint at a time.
//!
//! ```bash
//! BLAZR_TEST_QWEN2_MODEL=/path/to/qwen2.5-0.5b cargo test --release \
//!   --test arch_reference -- --nocapture
//! ```

use std::path::PathBuf;

use futures::StreamExt;

use blazr::config::{DtypeRequest, DtypeSource};
use blazr::loader::load_model_as;
use blazr::tokenizer::checkpoint_tokenizer;
use blazr::{Executor, GenerationConfig};
use boostr::inference::LayeredKvCache;
use boostr::{CpuDevice, CpuRuntime, DType, Tensor};

mod common;
use common::{model_fixture, skip_notice};

/// Tokens each test decodes.
const NUM_TOKENS: usize = 32;
/// Leading ids that must equal the reference exactly.
const CHECKED_IDS: usize = 8;
/// KV capacity of each run.
const NUM_CTX: usize = 256;
/// Logit tolerance, relative to the largest reference logit.
const LOGIT_RTOL: f32 = 1e-5;

/// One architecture's checkpoint, prompt and reference.
struct Case {
    /// Variable naming the checkpoint directory.
    env: &'static str,
    /// Directory under `$BOOSTR_MODELS_DIR` tried when `env` is unset.
    entry: &'static str,
    /// Model type blazr must route the checkpoint to.
    model_type: &'static str,
    num_layers: usize,
    hidden_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    prompt: &'static str,
    /// HF prompt ids, `add_special_tokens=True`.
    prompt_ids: &'static [u32],
    /// HF greedy ids. The first `CHECKED_IDS` must match when `check_ids`.
    reference_ids: &'static [u32],
    /// Assert the leading ids. Random-weight checkpoints keep their top-2
    /// logits within 1e-3, so a tie there is no defect. They match today, and
    /// their logits check covers them if a near-tie ever flips.
    check_ids: bool,
    /// HF logits at the last prompt position, vocabulary ids `0..n`.
    reference_logits: Option<&'static [f32]>,
}

fn greedy(max_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_tokens,
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..GenerationConfig::default()
    }
}

fn f32_request() -> DtypeRequest {
    DtypeRequest {
        dtype: DType::F32,
        source: DtypeSource::Flag,
    }
}

fn require_dir(case: &Case) -> Option<PathBuf> {
    let Some(path) = model_fixture(case.env, case.entry) else {
        skip_notice(case.entry, case.env);
        return None;
    };
    Some(path)
}

/// The last prompt row's logits from a fresh load, for a logits reference.
fn prompt_logits(dir: &PathBuf, prompt_ids: &[u32], head: usize) -> Vec<f32> {
    let device = CpuDevice::new();
    let request = f32_request();
    let (model, _) = load_model_as::<CpuRuntime, _>(dir, &device, Some(&request))
        .unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));
    let kv_heads = model
        .num_kv_heads()
        .expect("an attention model has KV heads");
    let head_dim = model.head_dim().expect("an attention model has a head dim");
    let mut cache = LayeredKvCache::<CpuRuntime>::new_positional(
        model.num_layers(),
        1,
        kv_heads,
        NUM_CTX,
        NUM_CTX,
        head_dim,
        DType::F32,
        &device,
    )
    .unwrap_or_else(|e| panic!("KV cache: {e}"));
    let ids: Vec<i64> = prompt_ids.iter().map(|&id| i64::from(id)).collect();
    let input = Tensor::<CpuRuntime>::from_slice(&ids, &[1, ids.len()], &device)
        .unwrap_or_else(|e| panic!("input ids: {e}"));
    let logits = model
        .forward_with_kv_cache(&input, &mut cache, 0)
        .unwrap_or_else(|e| panic!("prefill: {e}"));
    let vocab = logits.shape()[2];
    let all: Vec<f32> = logits.contiguous().unwrap().to_vec();
    let last = &all[(ids.len() - 1) * vocab..ids.len() * vocab];
    last[..head].to_vec()
}

/// One checkpoint in memory at a time. Several F32 checkpoints loaded in
/// parallel exceed host RAM: Phi-3.5 alone takes 15 GB.
static ONE_MODEL_AT_A_TIME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn run_case(case: &Case) {
    let Some(dir) = require_dir(case) else {
        return;
    };
    let _serial = ONE_MODEL_AT_A_TIME.lock().await;
    let device = CpuDevice::new();
    let request = f32_request();
    let (model, config) = load_model_as::<CpuRuntime, _>(&dir, &device, Some(&request))
        .unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));

    assert_eq!(config.model_type(), case.model_type, "model type");
    assert_eq!(model.num_layers(), case.num_layers, "layer count");
    assert_eq!(model.hidden_size(), case.hidden_size, "hidden size");
    assert_eq!(model.num_kv_heads(), Some(case.num_kv_heads), "KV heads");
    assert_eq!(model.head_dim(), Some(case.head_dim), "head dim");
    assert_eq!(model.dtype(), DType::F32, "compute dtype");

    let tokenizer = checkpoint_tokenizer(&dir, config.vocab_size())
        .unwrap_or_else(|e| panic!("tokenizer: {e}"));
    let prompt_ids = tokenizer.encode(case.prompt);
    assert_eq!(
        prompt_ids, case.prompt_ids,
        "prompt ids for {:?}",
        case.prompt
    );

    let executor = Executor::new(model, config, tokenizer, device, NUM_CTX)
        .unwrap_or_else(|e| panic!("Executor::new: {e}"));
    let gen_config = greedy(NUM_TOKENS);
    let mut stream = std::pin::pin!(executor.generate(&prompt_ids, &gen_config));
    let mut ids = Vec::new();
    while let Some(item) = stream.next().await {
        ids.push(item.unwrap_or_else(|e| panic!("generate: {e}")).token_id);
    }
    let text = executor.tokenizer().decode_lossy(&ids);
    println!("{}: ids {ids:?}", case.entry);
    println!("{}: text {text:?}", case.entry);
    let matched = ids
        .iter()
        .zip(case.reference_ids)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "{}: {matched} of {} ids match the reference",
        case.entry,
        case.reference_ids.len()
    );
    assert!(
        ids.len() >= CHECKED_IDS,
        "{} decoded {} tokens, fewer than {CHECKED_IDS}",
        case.entry,
        ids.len()
    );
    if case.check_ids {
        assert_eq!(
            &ids[..CHECKED_IDS],
            &case.reference_ids[..CHECKED_IDS],
            "{}: first {CHECKED_IDS} greedy ids differ from transformers",
            case.entry
        );
    }

    if let Some(reference) = case.reference_logits {
        let got = prompt_logits(&dir, case.prompt_ids, reference.len());
        let scale = reference.iter().fold(0.0f32, |m, r| m.max(r.abs()));
        let tol = LOGIT_RTOL * scale + 1e-6;
        let worst = got
            .iter()
            .zip(reference)
            .map(|(g, r)| (g - r).abs())
            .fold(0.0f32, f32::max);
        println!(
            "{}: max logit error {worst:e}, tolerance {tol:e}",
            case.entry
        );
        for (i, (g, r)) in got.iter().zip(reference).enumerate() {
            assert!(
                (g - r).abs() <= tol,
                "{}: logit {i} is {g}, transformers gives {r}",
                case.entry
            );
        }
    }
}

/// Qwen2.5-0.5B: Qwen2 attention biases, tied embeddings, no BOS.
#[tokio::test]
async fn qwen2_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_QWEN2_MODEL",
        entry: "qwen2.5-0.5b",
        model_type: "qwen2",
        num_layers: 24,
        hidden_size: 896,
        num_kv_heads: 2,
        head_dim: 64,
        prompt: "The capital of France is",
        prompt_ids: &[785, 6722, 315, 9625, 374],
        reference_ids: &[
            12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 2086, 7772, 304, 279, 1879,
            13, 1084, 374, 1083, 279, 6722, 315, 9625, 11, 279, 2086, 7772, 3146, 304, 4505, 11,
        ],
        check_ids: true,
        reference_logits: None,
    })
    .await;
}

/// Llama 3.2 1B: the plain Llama block, and its own `tokenizer.json`.
#[tokio::test]
async fn llama_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_LLAMA_MODEL",
        entry: "llama-3.2-1b",
        model_type: "llama",
        num_layers: 16,
        hidden_size: 2048,
        num_kv_heads: 8,
        head_dim: 64,
        prompt: "The capital of France is",
        prompt_ids: &[128000, 791, 6864, 315, 9822, 374],
        reference_ids: &[
            12366, 13, 1102, 374, 279, 1455, 95551, 3363, 304, 9822, 323, 279, 3224, 753, 6864,
            323, 7928, 3363, 13, 12366, 374, 279, 1455, 95551, 16036, 3158, 304, 279, 7665, 9323,
            323, 279,
        ],
        check_ids: true,
        reference_logits: None,
    })
    .await;
}

/// Pythia-160M: GPT-NeoX parallel residual, LayerNorm, 25% rotary, fused QKV.
#[tokio::test]
async fn gpt_neox_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_GPT_NEOX_MODEL",
        entry: "pythia-160m",
        model_type: "gpt_neox",
        num_layers: 12,
        hidden_size: 768,
        num_kv_heads: 12,
        head_dim: 64,
        prompt: "The capital of France is",
        prompt_ids: &[510, 5347, 273, 6181, 310],
        reference_ids: &[
            4441, 275, 253, 2846, 273, 7785, 15, 187, 187, 510, 2846, 310, 671, 1728, 281, 253,
            8530, 5112, 5842, 5453, 13, 253, 7785, 11877, 14, 29816, 404, 15, 187, 187, 12697, 187,
        ],
        check_ids: true,
        reference_logits: Some(&[
            829.086, 13.5148, 828.294, 827.893, 820.958, 822.796, 822.64, 823.495, 827.745,
            826.321, 828.092, 827.77, 823.782, 833.822, 829.844, 831.552,
        ]),
    })
    .await;
}

/// axolotl-ai-co/tiny-mixtral-30m: a trained 30M Mixtral, 8 experts, top-2.
#[tokio::test]
async fn mixtral_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_MIXTRAL_MODEL",
        entry: "tiny-mixtral-30m",
        model_type: "mixtral",
        num_layers: 4,
        hidden_size: 256,
        num_kv_heads: 2,
        head_dim: 32,
        prompt: "The capital of France is",
        prompt_ids: &[1, 415, 5565, 302, 4843, 349],
        reference_ids: &[
            272, 7639, 2990, 297, 272, 1526, 28723, 415, 2990, 349, 5651, 297, 272, 2990, 302,
            4843, 28725, 970, 378, 349, 5651, 297, 272, 2990, 302, 4843, 28723, 415, 2990, 349,
            5651, 297,
        ],
        check_ids: true,
        reference_logits: Some(&[
            -7.14664, -6.83416, 0.970686, -6.49775, -6.51809, -6.72634, -6.57867, -6.63325,
            -6.91664, -6.706, -7.11164, -6.8054, -3.81971, 4.56869, -6.91999, -6.48849,
        ]),
    })
    .await;
}

/// hf-tiny-v2/tiny-random-Qwen2MoeForCausalLM: random weights, shared expert
/// with a sigmoid gate, no top-k renormalization, GELU experts.
#[tokio::test]
async fn qwen2_moe_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_QWEN2_MOE_MODEL",
        entry: "tiny-random-qwen2moe",
        model_type: "qwen2_moe",
        num_layers: 2,
        hidden_size: 32,
        num_kv_heads: 2,
        head_dim: 16,
        prompt: "The capital of France is",
        prompt_ids: &[785, 6722, 315, 9625, 374],
        reference_ids: &[
            115642, 105922, 95138, 84142, 15009, 40207, 79388, 1986, 46666, 65676, 116352, 84761,
            51240, 151438, 20329, 33101, 93924, 80589, 42644, 75669, 27245, 5562, 41635, 129052,
            148199, 20291, 92414, 52869, 144747, 80408, 142928, 8482,
        ],
        check_ids: true,
        reference_logits: Some(&[
            0.106651, -0.0455639, -0.10963, 0.00921962, 0.110781, -0.0586174, 0.0200553, -0.204318,
            -0.0232903, -0.184062, -0.125594, -0.0184496, 0.110217, -0.230093, -0.124723, 0.100303,
        ]),
    })
    .await;
}

/// trl-internal-testing/tiny-DbrxForCausalLM: random weights, fused experts,
/// `clip_qkv`, bias-free LayerNorm.
#[tokio::test]
async fn dbrx_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_DBRX_MODEL",
        entry: "tiny-dbrx",
        model_type: "dbrx",
        num_layers: 2,
        hidden_size: 24,
        num_kv_heads: 2,
        head_dim: 4,
        prompt: "The capital of France is",
        prompt_ids: &[791, 6864, 315, 9822, 374],
        reference_ids: &[
            34216, 63721, 38331, 82627, 63721, 63721, 99658, 26481, 53736, 14688, 14688, 26481,
            26481, 99658, 36069, 26481, 77808, 31350, 11761, 36069, 36069, 31350, 97703, 97703,
            97703, 97703, 69458, 14688, 14688, 14688, 5336, 15785,
        ],
        check_ids: true,
        reference_logits: Some(&[
            -0.0400944,
            0.0240282,
            -0.00654448,
            0.00536384,
            0.0251723,
            -0.0359722,
            -0.0191073,
            -0.0473401,
            -0.00373439,
            0.0220094,
            -0.0240321,
            -0.0220335,
            -0.00360078,
            0.0221171,
            -0.0120466,
            1.27461e-05,
        ]),
    })
    .await;
}

/// trl-internal-testing/tiny-CohereForCausalLM: random weights, parallel
/// block, interleaved RoPE, `logit_scale`, tied embeddings.
#[tokio::test]
async fn cohere_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_COHERE_MODEL",
        entry: "tiny-cohere",
        model_type: "cohere",
        num_layers: 2,
        hidden_size: 8,
        num_kv_heads: 2,
        head_dim: 2,
        prompt: "The capital of France is",
        prompt_ids: &[5, 2162, 7784, 1719, 5334, 1801],
        reference_ids: &[
            158852, 229915, 158852, 158852, 158852, 158852, 229915, 229915, 158852, 158852, 158852,
            158852, 158852, 229915, 158852, 158852, 158852, 158852, 158852, 158852, 158852, 158852,
            158852, 158852, 158852, 158852, 158852, 158852, 158852, 158852, 158852, 158852,
        ],
        check_ids: true,
        reference_logits: Some(&[
            -0.000288658,
            -0.000552856,
            0.00243085,
            -0.00105066,
            0.000920896,
            0.000817211,
            -0.000579644,
            0.00159885,
            0.00118766,
            -0.00171618,
            -8.28271e-05,
            -0.000913497,
            -0.000984962,
            -9.53735e-05,
            -0.000990335,
            -0.00131564,
        ]),
    })
    .await;
}

/// unsloth/gemma-2-2b (ungated mirror of google/gemma-2-2b): sandwich norms,
/// `w + 1` RMSNorm, GeGLU, softcaps, alternating sliding windows.
#[tokio::test]
async fn gemma2_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_GEMMA2_MODEL",
        entry: "gemma-2-2b",
        model_type: "gemma2",
        num_layers: 26,
        hidden_size: 2304,
        num_kv_heads: 4,
        head_dim: 256,
        prompt: "The capital of France is",
        prompt_ids: &[2, 651, 6037, 576, 6081, 603],
        reference_ids: &[
            476, 3413, 576, 82777, 235265, 1165, 603, 476, 3413, 576, 3096, 235269, 6981, 235269,
            578, 4281, 235265, 1165, 603, 1170, 476, 3413, 576, 8012, 235269, 2960, 235269, 578,
            2245, 235265, 7127, 603,
        ],
        check_ids: true,
        reference_logits: Some(&[
            -20.168255,
            2.9185555,
            6.247_96,
            -14.99171,
            -0.936_836_7,
            -18.819948,
            -20.12544,
            -18.570656,
            -20.187_65,
            -15.294464,
            -20.183046,
            -16.081558,
            -18.661778,
            -20.086102,
            -20.16159,
            -20.241024,
        ]),
    })
    .await;
}

/// petals-team/falcon-rw-1b (SafeTensors copy of tiiuae/falcon-rw-1b): ALiBi
/// scaled by `1/sqrt(head_dim)`, per-head fused QKV, LayerNorm, GELU.
#[tokio::test]
async fn falcon_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_FALCON_MODEL",
        entry: "falcon-rw-1b",
        model_type: "falcon",
        num_layers: 24,
        hidden_size: 2048,
        num_kv_heads: 32,
        head_dim: 64,
        prompt: "The capital of France is",
        prompt_ids: &[464, 3139, 286, 4881, 318],
        reference_ids: &[
            257, 1748, 286, 39469, 13, 632, 318, 257, 1748, 286, 2106, 11, 3968, 11, 290, 1242, 13,
            632, 318, 257, 1748, 286, 6977, 11, 2057, 11, 290, 8237, 13, 632, 318, 257,
        ],
        check_ids: true,
        reference_logits: Some(&[
            32.022_79, 28.584_72, 26.926216, 25.938065, 25.234903, 27.017166, 29.052696, 28.695225,
            30.161533, 28.161194, 26.708126, 37.93222, 31.857807, 34.61383, 30.557992, 25.889652,
        ]),
    })
    .await;
}

/// TechxGenus/starcoder2-3b-instruct, a fine-tune of bigcode/starcoder2-3b
/// (12.1 GB in F32, over the 8 GB download cap): LayerNorm with bias,
/// plain GELU MLP with biases, attention biases, sliding window.
#[tokio::test]
async fn starcoder2_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_STARCODER2_MODEL",
        entry: "starcoder2-3b-instruct",
        model_type: "starcoder2",
        num_layers: 30,
        hidden_size: 3072,
        num_kv_heads: 2,
        head_dim: 128,
        prompt: "def fibonacci(n):",
        prompt_ids: &[610, 28319, 34789, 45, 115, 731],
        reference_ids: &[
            303, 434, 329, 2527, 244, 54, 63, 310, 461, 329, 303, 832, 63, 310, 461, 327, 26487,
            34789, 45, 115, 50, 54, 46, 494, 28319, 34789, 45, 115, 50, 55, 509, 222,
        ],
        check_ids: true,
        reference_logits: Some(&[
            3.488407, -5.802506, -6.187615, -6.262324, -3.823742, -5.53915, 0.1788558, -5.157418,
            -2.846774, -3.224383, -5.646985, -5.167936, 0.9537632, -2.517025, -6.347602, -6.253241,
        ]),
    })
    .await;
}

/// microsoft/Phi-3.5-mini-instruct: fused `qkv_proj` and `gate_up_proj`,
/// LongRoPE with the short factors below 4096 positions.
#[tokio::test]
async fn phi3_matches_transformers() {
    run_case(&Case {
        env: "BLAZR_TEST_PHI3_MODEL",
        entry: "phi-3.5-mini-instruct",
        model_type: "phi3",
        num_layers: 32,
        hidden_size: 3072,
        num_kv_heads: 32,
        head_dim: 96,
        prompt: "The capital of France is",
        prompt_ids: &[450, 7483, 310, 3444, 338],
        reference_ids: &[
            3681, 29889, 13, 13, 2177, 275, 338, 278, 7483, 4272, 310, 3444, 29889, 739, 338, 5982,
            297, 278, 6641, 29899, 25171, 760, 310, 278, 4234, 322, 338, 2998, 363, 967, 8261,
            4955,
        ],
        check_ids: true,
        reference_logits: Some(&[
            38.2145, 42.31735, 39.15597, 35.96156, 34.33443, 36.29036, 34.2196, 36.51657, 34.84375,
            34.46718, 36.35577, 38.75328, 46.80462, 52.41314, 38.40689, 40.65531,
        ]),
    })
    .await;
}
