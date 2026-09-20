//! Image prompt end to end through blazr's executor with a vision tower: a
//! real Ternary-Bonsai-2-27B `PQ2_0` GGUF, its Q8_0 mmproj, and the 96x64
//! fixture image, decoded greedily for 4 tokens on the eager path and, with
//! `cuda`, on the graph path.
//!
//! # Fixture files
//!
//! `BOOSTR_BONSAI2_DIR` must hold `Ternary-Bonsai-2-27B-PQ2_0.gguf` and
//! `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`. Unset dir or a missing file:
//! the test prints one `skip:` line and passes.
//!
//! `tests/fixtures/vision/img_96x64.png` is boostr's
//! `tests/fixtures/vision/img_96x64.png`. The prompt ids and the expected
//! greedy ids are `ids` and `greedy` of boostr's
//! `tests/fixtures/vision/img_96x64.logits.json`, the reference decoder's
//! output for that image.
//!
//! ```bash
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo test --release -p blazr \
//!   --test qwen35_vision_generate -- --nocapture
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo test --release -p blazr --features cuda \
//!   --test qwen35_vision_generate -- --nocapture
//! ```
//!
//! # Memory
//!
//! Every dense tensor of the model decodes to F32, and each test loads its
//! own copy, so run the tests one at a time (`--test-threads 1`) when RAM
//! is short.

use std::path::{Path, PathBuf};

use futures::StreamExt;

use blazr::engine::{GeneratedToken, ImageTokenLimits, VisionTower};
use blazr::loader::load_gguf_with_tokenizer;
use blazr::{Executor, GenerationConfig};
use boostr::{DType, Runtime};

const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const MODEL_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";
const MMPROJ_FILE: &str = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf";
const IMAGE_FILE: &str = "tests/fixtures/vision/img_96x64.png";
const TEXT_PROMPT: &str = "Hello";
const NUM_CTX: usize = 64;

/// `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Hi.<|im_end|>\n
/// <|im_start|>assistant\n`: `ids` of `img_96x64.logits.json`.
const IMAGE_PROMPT_IDS: [u32; 13] = [
    248045, 846, 198, 248053, 248056, 248054, 72240, 13, 248046, 198, 248045, 74455, 198,
];
/// `greedy` of `img_96x64.logits.json`: the prefill argmax then 3 steps.
const EXPECTED_GREEDY: [u32; 4] = [248068, 271, 248069, 271];

/// The model and mmproj paths, or `None` after one `skip:` line.
fn require_files() -> Option<(PathBuf, PathBuf)> {
    let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return None;
    };
    let model = dir.join(MODEL_FILE);
    let mmproj = dir.join(MMPROJ_FILE);
    for path in [&model, &mmproj] {
        if !path.is_file() {
            println!("skip: {} not found", path.display());
            return None;
        }
    }
    Some((model, mmproj))
}

fn fixture_image() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(IMAGE_FILE);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn greedy_config() -> GenerationConfig {
    let gen_config = GenerationConfig {
        max_tokens: EXPECTED_GREEDY.len(),
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..GenerationConfig::default()
    };
    assert!(gen_config.is_greedy());
    gen_config
}

/// Load the model into an executor without a tower.
fn load_executor<R>(model: &PathBuf, device: &R::Device) -> Executor<R>
where
    R: Runtime<DType = DType>,
    R::Client: boostr::ops::TensorOps<R>
        + boostr::ScalarOps<R>
        + boostr::ConvOps<R>
        + boostr::NormalizationOps<R>
        + boostr::UnaryOps<R>
        + boostr::ActivationOps<R>
        + boostr::BinaryOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + boostr::model::ModelClient<R>,
{
    let (loaded, config, tokenizer) = load_gguf_with_tokenizer::<R, _>(model, device)
        .unwrap_or_else(|e| panic!("load_gguf_with_tokenizer on {}: {e}", model.display()));
    assert_eq!(config.model_type(), "qwen35");
    let executor = Executor::new(loaded, config, tokenizer, device.clone(), NUM_CTX)
        .unwrap_or_else(|e| panic!("Executor::new: {e}"));
    assert!(executor.vision_tower().is_none(), "no mmproj configured");
    executor
}

/// Load the tower from `mmproj` against the executor's tokenizer and attach
/// it; the marker ids it finds must be the fixture's.
fn attach_tower<R>(executor: Executor<R>, mmproj: &Path, device: &R::Device) -> Executor<R>
where
    R: Runtime<DType = DType>,
    R::Client: boostr::ops::TensorOps<R>
        + boostr::ScalarOps<R>
        + boostr::ConvOps<R>
        + boostr::NormalizationOps<R>
        + boostr::UnaryOps<R>
        + boostr::ActivationOps<R>
        + boostr::BinaryOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + boostr::model::ModelClient<R>,
{
    let tower = VisionTower::<R>::load(
        mmproj,
        device,
        executor.tokenizer(),
        ImageTokenLimits::default(),
    )
    .unwrap_or_else(|e| panic!("VisionTower::load {}: {e}", mmproj.display()));
    let markers = tower.markers();
    assert_eq!(markers.vision_start, IMAGE_PROMPT_IDS[3]);
    assert_eq!(markers.image_pad, IMAGE_PROMPT_IDS[4]);
    assert_eq!(markers.vision_end, IMAGE_PROMPT_IDS[5]);
    let executor = executor.with_vision_tower(tower);
    assert!(executor.vision_tower().is_some());
    executor
}

/// Collect every token id from `stream`, failing on the first error.
async fn collect_ids<S>(stream: S, path: &str) -> Vec<u32>
where
    S: futures::Stream<Item = anyhow::Result<GeneratedToken>>,
{
    let mut stream = std::pin::pin!(stream);
    let mut ids = Vec::new();
    while let Some(item) = stream.next().await {
        let token = item.unwrap_or_else(|e| panic!("{path}: {e}"));
        ids.push(token.token_id);
    }
    ids
}

/// The first error `stream` yields.
async fn first_error<S>(stream: S) -> String
where
    S: futures::Stream<Item = anyhow::Result<GeneratedToken>>,
{
    let mut stream = std::pin::pin!(stream);
    match stream.next().await {
        Some(Err(e)) => e.to_string(),
        Some(Ok(token)) => panic!("expected an error, got token {}", token.token_id),
        None => panic!("expected an error, got an empty stream"),
    }
}

#[tokio::test]
async fn image_prompt_decodes_the_fixture_ids_on_cpu() {
    use boostr::{CpuDevice, CpuRuntime};

    let Some((model, mmproj)) = require_files() else {
        return;
    };
    let device = CpuDevice::new();
    let executor = load_executor::<CpuRuntime>(&model, &device);
    let gen_config = greedy_config();
    let images = [fixture_image()];

    // Without a tower an image request is refused, naming --mmproj.
    let err =
        first_error(executor.generate_with_images(&IMAGE_PROMPT_IDS, &images, &gen_config)).await;
    assert!(err.contains("--mmproj"), "{err}");

    let executor = attach_tower(executor, &mmproj, &device);
    let eager = collect_ids(
        executor.generate_with_images(&IMAGE_PROMPT_IDS, &images, &gen_config),
        "generate_with_images",
    )
    .await;
    println!("image prompt -> eager {eager:?}");
    assert_eq!(eager, EXPECTED_GREEDY);

    // A text prompt through the image entry with no images is `generate`.
    let text_ids = executor.tokenizer().encode(TEXT_PROMPT);
    let plain = collect_ids(executor.generate(&text_ids, &gen_config), "generate").await;
    let via_images = collect_ids(
        executor.generate_with_images(&text_ids, &[], &gen_config),
        "generate_with_images(text)",
    )
    .await;
    println!("text prompt -> generate {plain:?} via images {via_images:?}");
    assert_eq!(plain.len(), EXPECTED_GREEDY.len());
    assert_eq!(via_images, plain);
}

#[cfg(feature = "cuda")]
#[tokio::test]
async fn image_prompt_matches_on_the_graph_path_on_cuda() {
    use boostr::{CudaDevice, CudaRuntime};

    let Some((model, mmproj)) = require_files() else {
        return;
    };
    if !boostr::runtime::cuda::is_cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let device = CudaDevice::new(0);
    let executor = load_executor::<CudaRuntime>(&model, &device);
    let executor = attach_tower(executor, &mmproj, &device);
    let gen_config = greedy_config();
    let images = [fixture_image()];

    let eager = collect_ids(
        executor.generate_with_images(&IMAGE_PROMPT_IDS, &images, &gen_config),
        "generate_with_images",
    )
    .await;
    let graph = collect_ids(
        executor.generate_with_graphs_images(&IMAGE_PROMPT_IDS, &images, &gen_config),
        "generate_with_graphs_images",
    )
    .await;
    println!("image prompt -> eager {eager:?} graph {graph:?}");
    assert_eq!(eager, EXPECTED_GREEDY);
    assert_eq!(graph, EXPECTED_GREEDY);

    // The session captured for the image prompt serves a text prompt too,
    // and the text path is the plain one.
    let text_ids = executor.tokenizer().encode(TEXT_PROMPT);
    let plain = collect_ids(executor.generate(&text_ids, &gen_config), "generate").await;
    let plain_graph = collect_ids(
        executor.generate_with_graphs(&text_ids, &gen_config),
        "generate_with_graphs",
    )
    .await;
    let via_images = collect_ids(
        executor.generate_with_graphs_images(&text_ids, &[], &gen_config),
        "generate_with_graphs_images(text)",
    )
    .await;
    println!("text prompt -> eager {plain:?} graph {plain_graph:?} via images {via_images:?}");
    assert_eq!(plain.len(), EXPECTED_GREEDY.len());
    assert_eq!(plain_graph, plain);
    assert_eq!(via_images, plain);
}
