//! Greedy generation on the wgpu runtime against the CPU runtime.
//!
//! WebGPU computes in F32 only, so the loader decodes F16/BF16 weights to
//! F32 there. The CPU reference decodes the same checkpoint to F32 too, so
//! both runtimes do the same arithmetic. The tests check:
//!
//! - every layer's hidden state, prefill and one decode step, within 1e-3
//!   relative to the CPU state's largest magnitude;
//! - 32 greedy token ids, which must match exactly.
//!
//! # Fixture
//!
//! `BLAZR_TEST_LLAMA_MODEL` is a Llama-family SafeTensors directory
//! (`config.json` beside the weights). Unset, `$BOOSTR_MODELS_DIR/
//! llama-3.2-1b` is tried. `BLAZR_TEST_WGPU_GGUF` is a Llama-family GGUF
//! file. Its CPU reference dequantizes every block to F32, the math WebGPU
//! runs. Each test prints one skip line and passes
//! when its fixture is absent or no wgpu adapter opens.
//!
//! ```bash
//! BLAZR_TEST_LLAMA_MODEL=/path/to/llama-dir cargo test --features wgpu \
//!   --test wgpu_generate -- --nocapture --test-threads 1
//! ```
//!
//! # Memory
//!
//! The CPU reference holds the model in F32: about 4 bytes per parameter in
//! host RAM. The wgpu copy holds the same in device memory.

#![cfg(feature = "wgpu")]

use std::path::{Path, PathBuf};

use futures::StreamExt;

use blazr::config::BlazrConfig;
use blazr::loader::{load_gguf, load_model};
use blazr::{Executor, GenerationConfig};
use boostr::format::SafeTensorsLoader;
use boostr::inference::LayeredKvCache;
use boostr::model::LoadedModel;
use boostr::quant::DequantOps;
use boostr::runtime::wgpu::{WgpuDevice, WgpuRuntime};
use boostr::{CpuClient, CpuDevice, CpuRuntime, DType, Runtime, Tensor, TypeConversionOps};
use boostr::{VarBuilder, VarMap, Weight};
use splintr::AnyTokenizer;

mod common;
use common::{model_fixture, skip_notice};

/// A Llama-family SafeTensors checkpoint directory.
const ENV_DIR: &str = "BLAZR_TEST_LLAMA_MODEL";
/// Tried under `$BOOSTR_MODELS_DIR` when `ENV_DIR` is unset.
const MODELS_DIR_ENTRY: &str = "llama-3.2-1b";
/// A Llama-family GGUF file. No fallback.
const ENV_GGUF: &str = "BLAZR_TEST_WGPU_GGUF";
const PROMPT: &str = "The capital of France is";
const NUM_TOKENS: usize = 32;
const NUM_CTX: usize = 256;
/// Largest per-layer difference, relative to the CPU state's peak.
const LAYER_TOLERANCE: f32 = 1e-3;

/// Adapter 0, or `None` after one skip line.
fn require_wgpu() -> Option<WgpuDevice> {
    match blazr::runtime::open_wgpu_device(0) {
        Ok(device) => Some(device),
        Err(e) => {
            println!("skip: no wgpu adapter: {e}");
            None
        }
    }
}

/// The SafeTensors checkpoint directory, or `None` after one skip line.
fn require_dir() -> Option<PathBuf> {
    let Some(path) = model_fixture(ENV_DIR, MODELS_DIR_ENTRY) else {
        skip_notice("a Llama-family SafeTensors checkpoint", ENV_DIR);
        return None;
    };
    if !path.join("config.json").is_file() {
        println!(
            "skip: {} has no config.json; point {ENV_DIR} at a Llama-family SafeTensors \
             checkpoint directory",
            path.display()
        );
        return None;
    }
    Some(path)
}

/// Greedy, no penalties.
fn greedy_config() -> GenerationConfig {
    GenerationConfig {
        max_tokens: NUM_TOKENS,
        temperature: 0.0,
        repeat_penalty: 1.0,
        ..GenerationConfig::default()
    }
}

/// The checkpoint on the CPU, every float tensor decoded to F32.
///
/// `config` is the wgpu load's config, so both models share one shape.
fn cpu_f32_model(dir: &Path, config: &BlazrConfig) -> LoadedModel<CpuRuntime> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let mut loader = SafeTensorsLoader::open(dir)
        .unwrap_or_else(|e| panic!("SafeTensorsLoader::open({}): {e}", dir.display()));
    let mut var_map = VarMap::<CpuRuntime>::new();
    for name in loader.tensor_names() {
        let tensor = loader
            .load_tensor::<CpuRuntime>(&name, &device)
            .unwrap_or_else(|e| panic!("load_tensor({name}): {e}"));
        let tensor = match tensor.dtype() {
            DType::F16 | DType::BF16 => client
                .cast(&tensor, DType::F32)
                .unwrap_or_else(|e| panic!("cast {name} to F32: {e}")),
            _ => tensor,
        };
        var_map.insert(name, tensor);
    }
    let mut vb = VarBuilder::new(&mut var_map, &device);
    LoadedModel::load(&config.model, &mut vb).unwrap_or_else(|e| panic!("LoadedModel::load: {e}"))
}

/// A GGUF file's model and config on `device`.
fn load_gguf_parts<R>(path: &Path, device: &R::Device) -> (LoadedModel<R>, BlazrConfig)
where
    R: Runtime<DType = DType>,
    R::Client: boostr::ops::TensorOps<R> + DequantOps<R>,
{
    load_gguf::<R, _>(path, device).unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

/// The GGUF file's own tokenizer, else the bundled one for `vocab_size`.
fn gguf_tokenizer(path: &Path, vocab_size: usize) -> AnyTokenizer {
    let embedded = boostr::format::Gguf::open(path)
        .ok()
        .and_then(|gguf| boostr::format::extract_gguf_vocab(gguf.metadata()).ok())
        .and_then(|vocab| splintr::from_gguf_vocab(vocab).ok());
    embedded.unwrap_or_else(|| {
        blazr::tokenizer::from_vocab_size(vocab_size)
            .unwrap_or_else(|e| panic!("tokenizer for vocab {vocab_size}: {e}"))
    })
}

/// A GGUF file on the CPU with every block-quantized weight dequantized to
/// F32. WebGPU dots dequantized F32 weights, so this is its exact
/// reference. The CPU's own K-quant matmul quantizes the activation, which
/// is a different rounding.
fn cpu_f32_from_gguf(path: &Path, config: &BlazrConfig) -> LoadedModel<CpuRuntime> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let mut var_map = VarMap::<CpuRuntime>::from_gguf(path, &device)
        .unwrap_or_else(|e| panic!("VarMap::from_gguf({}): {e}", path.display()));
    let names: Vec<String> = var_map.names().map(str::to_string).collect();
    for name in names {
        let weight = var_map
            .take(&name)
            .unwrap_or_else(|e| panic!("take {name}: {e}"));
        let tensor = match weight {
            Weight::Quantized(qt) => client
                .dequantize(&qt, DType::F32)
                .unwrap_or_else(|e| panic!("dequantize {name}: {e}")),
            Weight::Standard(t) if matches!(t.dtype(), DType::F16 | DType::BF16) => client
                .cast(&t, DType::F32)
                .unwrap_or_else(|e| panic!("cast {name} to F32: {e}")),
            Weight::Standard(t) => t,
            Weight::DecomposedQuant(_) => panic!("{name}: a GGUF holds no AWQ/GPTQ weight"),
        };
        var_map.insert(name, tensor);
    }
    let mut vb = VarBuilder::new(&mut var_map, &device);
    LoadedModel::load(&config.model, &mut vb).unwrap_or_else(|e| panic!("LoadedModel::load: {e}"))
}

/// The greedy ids `executor` produces for `prompt`.
async fn greedy_ids<R>(executor: &Executor<R>, prompt: &[u32]) -> Vec<u32>
where
    R: Runtime<DType = DType>,
    R::Client: boostr::ops::TensorOps<R>
        + boostr::ScalarOps<R>
        + boostr::ConvOps<R>
        + boostr::NormalizationOps<R>
        + boostr::UnaryOps<R>
        + boostr::ActivationOps<R>
        + boostr::BinaryOps<R>
        + TypeConversionOps<R>
        + boostr::SamplingOps<R>
        + boostr::GrammarDfaOps<R>
        + boostr::model::ModelClient<R>
        + boostr::RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    let config = greedy_config();
    let mut stream = std::pin::pin!(executor.generate(prompt, &config));
    let mut ids = Vec::with_capacity(NUM_TOKENS);
    while let Some(item) = stream.next().await {
        ids.push(item.unwrap_or_else(|e| panic!("generate: {e}")).token_id);
    }
    ids
}

/// The first index where `a` and `b` differ, or `None` when they match.
fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

/// The largest absolute difference, over the reference's largest magnitude.
fn relative_gap(reference: &[f32], other: &[f32]) -> f32 {
    assert_eq!(reference.len(), other.len(), "state lengths differ");
    let peak = reference.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let gap = reference
        .iter()
        .zip(other)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    gap / peak
}

/// `embed` for state 0, else the layer the state comes out of.
fn stage_name(state: usize) -> String {
    match state {
        0 => "embed".to_string(),
        n => format!("layer {}", n - 1),
    }
}

/// A flat KV cache for one sequence of `model`.
fn kv_cache<R: Runtime<DType = DType>>(
    model: &LoadedModel<R>,
    device: &R::Device,
) -> LayeredKvCache<R>
where
    R::Client: boostr::model::ModelClient<R>,
{
    LayeredKvCache::new_positional(
        model.num_layers(),
        1,
        model.num_kv_heads().unwrap_or(8),
        NUM_CTX,
        NUM_CTX,
        model.head_dim().unwrap_or(64),
        DType::F32,
        device,
    )
    .unwrap_or_else(|e| panic!("LayeredKvCache::new_positional: {e}"))
}

/// The embedding output, then each layer's hidden state, for `ids` at
/// `position`, as host vectors.
fn layer_states<R>(
    model: &LoadedModel<R>,
    cache: &mut LayeredKvCache<R>,
    ids: &[u32],
    position: usize,
    device: &R::Device,
) -> Vec<Vec<f32>>
where
    R: Runtime<DType = DType>,
    R::Client: boostr::model::ModelClient<R>,
{
    let ids: Vec<i64> = ids.iter().map(|&t| t as i64).collect();
    let input = Tensor::<R>::from_slice(&ids, &[1, ids.len()], device).expect("input ids");
    let mut hidden = model.forward_embed(&input).expect("forward_embed");
    let mut prev = None;
    let mut states = Vec::with_capacity(model.num_layers() + 1);
    states.push(hidden.tensor().contiguous().expect("contiguous").to_vec());
    for layer in 0..model.num_layers() {
        let (h, mlp) = model
            .forward_layers_range(hidden, prev, cache, layer, layer + 1, position)
            .unwrap_or_else(|e| panic!("layer {layer}: {e}"));
        // The residual add of the MLP output is deferred to the next layer.
        let mut state: Vec<f32> = h.tensor().contiguous().expect("contiguous").to_vec();
        if let Some(ref m) = mlp {
            let m: Vec<f32> = m.tensor().contiguous().expect("contiguous").to_vec();
            state.iter_mut().zip(m).for_each(|(s, v)| *s += v);
        }
        states.push(state);
        hidden = h;
        prev = mlp;
    }
    states
}

/// Prefill, then one decode step, layer by layer on both runtimes.
#[test]
fn safetensors_layer_states_track_cpu() {
    let Some(dir) = require_dir() else { return };
    let Some(wgpu_device) = require_wgpu() else {
        return;
    };
    let (wgpu_model, config) = load_model::<WgpuRuntime, _>(&dir, &wgpu_device)
        .unwrap_or_else(|e| panic!("load_model on wgpu: {e}"));
    let cpu_device = CpuDevice::new();
    let cpu_model = cpu_f32_model(&dir, &config);

    let tokenizer = blazr::tokenizer::checkpoint_tokenizer(&dir, config.vocab_size())
        .unwrap_or_else(|e| panic!("checkpoint tokenizer: {e}"));
    let prompt = tokenizer.encode(PROMPT);
    let mut cpu_cache = kv_cache(&cpu_model, &cpu_device);
    let mut wgpu_cache = kv_cache(&wgpu_model, &wgpu_device);

    let steps: [(&[u32], usize); 2] = [(&prompt, 0), (&prompt[..1], prompt.len())];
    for (step, (ids, position)) in steps.into_iter().enumerate() {
        let cpu = layer_states(&cpu_model, &mut cpu_cache, ids, position, &cpu_device);
        let gpu = layer_states(&wgpu_model, &mut wgpu_cache, ids, position, &wgpu_device);
        for (layer, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            let gap = relative_gap(c, g);
            println!("step {step} {}: relative gap {gap:.3e}", stage_name(layer));
            assert!(
                gap < LAYER_TOLERANCE,
                "step {step} {}: wgpu state is {gap:.3e} from the CPU state, \
                 above {LAYER_TOLERANCE:.0e}",
                stage_name(layer)
            );
        }
    }
}

/// Prefill, then one decode step, layer by layer, for a GGUF file, against
/// the dequantized F32 CPU reference.
#[test]
fn gguf_layer_states_track_cpu() {
    let Some(path) = model_fixture(ENV_GGUF, "") else {
        skip_notice("a Llama-family GGUF file", ENV_GGUF);
        return;
    };
    let Some(wgpu_device) = require_wgpu() else {
        return;
    };
    let (wgpu_model, config) = load_gguf_parts::<WgpuRuntime>(&path, &wgpu_device);
    let tokenizer = gguf_tokenizer(&path, config.vocab_size());
    let cpu_device = CpuDevice::new();
    let cpu_model = cpu_f32_from_gguf(&path, &config);

    let prompt = tokenizer.encode(PROMPT);
    let mut cpu_cache = kv_cache(&cpu_model, &cpu_device);
    let mut wgpu_cache = kv_cache(&wgpu_model, &wgpu_device);
    let steps: [(&[u32], usize); 2] = [(&prompt, 0), (&prompt[..1], prompt.len())];
    for (step, (ids, position)) in steps.into_iter().enumerate() {
        let cpu = layer_states(&cpu_model, &mut cpu_cache, ids, position, &cpu_device);
        let gpu = layer_states(&wgpu_model, &mut wgpu_cache, ids, position, &wgpu_device);
        for (layer, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
            let gap = relative_gap(c, g);
            println!("step {step} {}: relative gap {gap:.3e}", stage_name(layer));
            assert!(
                gap < LAYER_TOLERANCE,
                "step {step} {}: wgpu state is {gap:.3e} from the CPU state, \
                 above {LAYER_TOLERANCE:.0e}",
                stage_name(layer)
            );
        }
    }
}

/// 32 greedy ids on wgpu equal the CPU ids for the same F32 weights.
#[tokio::test]
async fn safetensors_greedy_ids_match_cpu() {
    let Some(dir) = require_dir() else { return };
    let Some(wgpu_device) = require_wgpu() else {
        return;
    };
    let (wgpu_model, config) = load_model::<WgpuRuntime, _>(&dir, &wgpu_device)
        .unwrap_or_else(|e| panic!("load_model on wgpu: {e}"));
    let cpu_model = cpu_f32_model(&dir, &config);
    let mut cpu_config = config.clone();
    cpu_config.inference.dtype = "f32".to_string();

    let vocab_size = config.vocab_size();
    let tokenizer = || {
        blazr::tokenizer::checkpoint_tokenizer(&dir, vocab_size)
            .unwrap_or_else(|e| panic!("checkpoint tokenizer: {e}"))
    };
    let wgpu = Executor::new(wgpu_model, config, tokenizer(), wgpu_device, NUM_CTX)
        .unwrap_or_else(|e| panic!("Executor::new on wgpu: {e}"));
    let cpu = Executor::new(
        cpu_model,
        cpu_config,
        tokenizer(),
        CpuDevice::new(),
        NUM_CTX,
    )
    .unwrap_or_else(|e| panic!("Executor::new on cpu: {e}"));

    let prompt = cpu.tokenizer().encode(PROMPT);
    let cpu_ids = greedy_ids(&cpu, &prompt).await;
    let wgpu_ids = greedy_ids(&wgpu, &prompt).await;
    println!("cpu  ids: {cpu_ids:?}");
    println!("wgpu ids: {wgpu_ids:?}");
    assert_eq!(cpu_ids.len(), NUM_TOKENS, "the CPU run stopped early");
    assert_eq!(
        first_divergence(&cpu_ids, &wgpu_ids),
        None,
        "wgpu ids diverge from the CPU ids"
    );
}

/// 32 greedy ids from a GGUF file on wgpu equal the ids of the dequantized
/// F32 CPU reference.
#[tokio::test]
async fn gguf_greedy_ids_match_cpu() {
    let Some(path) = model_fixture(ENV_GGUF, "") else {
        skip_notice("a Llama-family GGUF file", ENV_GGUF);
        return;
    };
    let Some(wgpu_device) = require_wgpu() else {
        return;
    };
    let (wgpu_model, wgpu_config) = load_gguf_parts::<WgpuRuntime>(&path, &wgpu_device);
    let cpu_device = CpuDevice::new();
    let cpu_model = cpu_f32_from_gguf(&path, &wgpu_config);
    let mut cpu_config = wgpu_config.clone();
    cpu_config.inference.dtype = "f32".to_string();
    let wgpu_tok = gguf_tokenizer(&path, wgpu_config.vocab_size());
    let cpu_tok = gguf_tokenizer(&path, wgpu_config.vocab_size());

    let wgpu = Executor::new(wgpu_model, wgpu_config, wgpu_tok, wgpu_device, NUM_CTX)
        .unwrap_or_else(|e| panic!("Executor::new on wgpu: {e}"));
    let cpu = Executor::new(cpu_model, cpu_config, cpu_tok, cpu_device, NUM_CTX)
        .unwrap_or_else(|e| panic!("Executor::new on cpu: {e}"));

    let prompt = cpu.tokenizer().encode(PROMPT);
    let cpu_ids = greedy_ids(&cpu, &prompt).await;
    let wgpu_ids = greedy_ids(&wgpu, &prompt).await;
    println!("cpu  ids: {cpu_ids:?}");
    println!("wgpu ids: {wgpu_ids:?}");
    assert_eq!(cpu_ids.len(), NUM_TOKENS, "the CPU run stopped early");
    assert_eq!(
        first_divergence(&cpu_ids, &wgpu_ids),
        None,
        "wgpu ids diverge from the CPU ids"
    );
}
