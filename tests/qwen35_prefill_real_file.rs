//! A prefill's last-row logits match the last row of the full forward, on the
//! real Qwen3.5 GGUF the server runs.
//!
//! Reads `$BOOSTR_MODELS_DIR/Qwen3.5-4B-Q8_0.gguf`, and skips when it is absent
//! or no CUDA device answers. Run with:
//!   cargo test --features cuda --test qwen35_prefill_real_file

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use blazr::loader::load_gguf_with_tokenizer_as;
use boostr::inference::kv_cache::layered::LayeredKvCacheConfig;
use boostr::inference::{LayeredGdnState, LayeredKvCache};
use boostr::model::LoadedModel;
use boostr::runtime::cuda::{is_cuda_available, CudaDevice, CudaRuntime};
use boostr::{DType, Tensor};

/// Rows of the prompt: two chunks at the server's chunk size of 512.
const ROWS: usize = 600;
const CHUNK: usize = 512;
/// F32 logits of order 10 agree to this bound across the two kernels' shapes.
const TOLERANCE: f32 = 1e-3;

type Caches = (LayeredKvCache<CudaRuntime>, LayeredGdnState<CudaRuntime>);

fn caches(model: &LoadedModel<CudaRuntime>, device: &CudaDevice) -> Caches {
    let config = LayeredKvCacheConfig {
        batch_size: 1,
        num_kv_heads: model.num_kv_heads().expect("KV heads"),
        initial_capacity: ROWS + 1,
        max_seq_len: ROWS + 1,
        head_dim: model.head_dim().expect("head dim"),
        dtype: DType::F32,
    };
    let kv = LayeredKvCache::new(
        model.num_attention_layers().expect("attention layers"),
        &config,
        device,
    )
    .expect("KV cache");
    let gdn = LayeredGdnState::zeros(
        model.num_gdn_layers().expect("GDN layers"),
        model.gdn_config().expect("GDN config"),
        1,
        DType::F32,
        device,
    )
    .expect("GDN state");
    (kv, gdn)
}

fn max_abs_diff(a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> f32 {
    a.to_vec::<f32>()
        .iter()
        .zip(b.to_vec::<f32>().iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn last_row_logits_match_the_full_forward() {
    let Ok(dir) = std::env::var("BOOSTR_MODELS_DIR") else {
        eprintln!("skip: BOOSTR_MODELS_DIR is unset");
        return;
    };
    let file = PathBuf::from(dir).join("Qwen3.5-4B-Q8_0.gguf");
    if !file.is_file() || !is_cuda_available() {
        eprintln!("skip: {} or a CUDA device is absent", file.display());
        return;
    }
    let device = CudaDevice::new(0);
    let (model, _, _) =
        load_gguf_with_tokenizer_as::<CudaRuntime, _>(&file, &device, None).expect("model loads");

    let ids: Vec<i64> = (0..ROWS as i64).map(|i| 1000 + (i * 37) % 5000).collect();
    let input = Tensor::<CudaRuntime>::from_slice(&ids, &[1, ROWS], &device).expect("ids");

    let (mut kv, mut gdn) = caches(&model, &device);
    let full = model
        .forward_qwen35(&input, &mut kv, &mut gdn, 0)
        .expect("full forward");
    let full_last = full
        .narrow(1, ROWS - 1, 1)
        .expect("narrow")
        .contiguous()
        .expect("copy");
    drop(full);
    drop((kv, gdn));

    let (mut kv, mut gdn) = caches(&model, &device);
    let last = model
        .forward_qwen35_last(&input, &mut kv, &mut gdn, 0)
        .expect("last-row forward");
    assert_eq!(last.shape(), full_last.shape());
    let diff = max_abs_diff(&last, &full_last);
    eprintln!("one call: max abs logit diff {diff}");
    assert!(diff < TOLERANCE, "one call: max abs logit diff {diff}");
    drop((kv, gdn));

    // The server's chunked path: no logits for the first chunk.
    let (mut kv, mut gdn) = caches(&model, &device);
    let head = input
        .narrow(1, 0, CHUNK)
        .expect("narrow")
        .contiguous()
        .expect("copy");
    let tail = input
        .narrow(1, CHUNK, ROWS - CHUNK)
        .expect("narrow")
        .contiguous()
        .expect("copy");
    model
        .advance_qwen35(&head, &mut kv, &mut gdn, 0)
        .expect("first chunk");
    let chunked = model
        .forward_qwen35_last(&tail, &mut kv, &mut gdn, CHUNK)
        .expect("last chunk");
    let diff = max_abs_diff(&chunked, &full_last);
    eprintln!("chunked: max abs logit diff {diff}");
    assert!(diff < TOLERANCE, "chunked: max abs logit diff {diff}");
}
