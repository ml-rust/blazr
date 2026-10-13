//! A real PEFT LoRA adapter served as a blazr variant: the base Qwen3.5
//! checkpoint loads through the model scheduler on the CPU in F32, the
//! adapter registers as the variant `<model>:kev` through
//! `Scheduler::register_variant` (the path `--lora` and `POST /v1/lora`
//! take), and a 5-token prefill of each model is checked against
//! `transformers` + `peft` reference logits. The axum router then runs
//! in-process over the same scheduler: a completion addressed to the
//! variant runs on it, and an unknown `lora_adapter` is a 404.
//!
//! # Fixtures
//!
//! - `BLAZR_LORA_BASE_DIR`: the base checkpoint directory, such as
//!   `qwen3.5-4b-base`. Its directory name is the model name.
//! - `BLAZR_LORA_ADAPTER_DIR`: the PEFT adapter trained on that base.
//! - `BLAZR_LORA_REF`: little-endian F32 logits `[5, vocab]` of the base
//!   with the adapter applied unmerged, for the token ids `PROMPT`.
//! - `BLAZR_LORA_BASE_REF`: the same for the base without the adapter.
//!
//! Unset or missing, the test prints one `skip:` line and passes.
//!
//! ```bash
//! BLAZR_LORA_BASE_DIR=/path/to/qwen3.5-4b-base BLAZR_LORA_ADAPTER_DIR=/path/to/kev-4b \
//!   BLAZR_LORA_REF=/path/to/ref-4b-kev.f32 BLAZR_LORA_BASE_REF=/path/to/ref-4b.f32 \
//!   cargo nextest run --test lora_variant_real_file --no-capture
//! ```
//!
//! # Memory
//!
//! The base loads in F32: expect peak RSS near four bytes per parameter of
//! the text model. The variant shares those weights.

#![cfg(all(unix, feature = "server", not(any(feature = "cuda", feature = "wgpu"))))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use blazr::config::{DtypeRequest, DtypeSource};
use blazr::engine::{AdapterSpec, Executor, Scheduler};
use blazr::server::handlers::AppState;
use boostr::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
use boostr::model::LoadedModel;
use boostr::{CpuClient, CpuDevice, CpuRuntime, DType, Tensor};

const ENV_BASE: &str = "BLAZR_LORA_BASE_DIR";
const ENV_ADAPTER: &str = "BLAZR_LORA_ADAPTER_DIR";
const ENV_REF: &str = "BLAZR_LORA_REF";
const ENV_BASE_REF: &str = "BLAZR_LORA_BASE_REF";
const PROMPT: [i64; 5] = [248_044, 17, 42, 99, 7];
/// Largest logit difference from a reference, as boostr's adapter test.
const TOLERANCE: f32 = 5e-2;

/// The value of `var` as an existing path, or `None` after one `skip:` line.
fn env_path(var: &str) -> Option<PathBuf> {
    let Some(path) = std::env::var_os(var).map(PathBuf::from) else {
        println!("skip: {var} not set");
        return None;
    };
    if !path.exists() {
        println!("skip: {var}={} does not exist", path.display());
        return None;
    }
    Some(path)
}

/// The 5-token prefill logits of a qwen35 `model`, `[5 * vocab]`.
fn prefill_logits(model: &LoadedModel<CpuRuntime>, device: &CpuDevice) -> Vec<f32> {
    let LoadedModel::Qwen35(model) = model else {
        panic!("expected a qwen35 model, got {}", model.model_type());
    };
    let client = CpuClient::new(device.clone());
    let attn = model.attention_config();
    let kv_config = LayeredKvCacheConfig {
        batch_size: 1,
        num_kv_heads: attn.num_kv_heads,
        initial_capacity: 16,
        max_seq_len: model.config().max_seq_len,
        head_dim: attn.head_dim,
        dtype: DType::F32,
    };
    let mut kv =
        LayeredKvCache::<CpuRuntime>::new(model.num_attention_layers(), &kv_config, device)
            .unwrap_or_else(|e| panic!("LayeredKvCache::new: {e}"));
    let mut gdn = LayeredGdnState::<CpuRuntime>::zeros(
        model.num_gdn_layers(),
        model.gdn_config(),
        1,
        DType::F32,
        device,
    )
    .unwrap_or_else(|e| panic!("LayeredGdnState::zeros: {e}"));
    let ids = Tensor::<CpuRuntime>::from_slice(&PROMPT, &[1, PROMPT.len()], device)
        .unwrap_or_else(|e| panic!("prompt tensor: {e}"));
    let logits = model
        .forward_qwen35(&client, &ids, &mut kv, &mut gdn, 0)
        .unwrap_or_else(|e| panic!("prefill forward: {e}"));
    let data = logits.to_vec::<f32>();
    let bad = data.iter().position(|v| !v.is_finite());
    assert!(bad.is_none(), "non-finite logit at flat index {bad:?}");
    data
}

/// The reference logits at `path`.
fn reference(path: &Path, vocab: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let (words, tail) = bytes.as_chunks::<4>();
    assert!(
        tail.is_empty(),
        "{} is not a whole number of f32",
        path.display()
    );
    let want: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
    assert_eq!(want.len(), PROMPT.len() * vocab, "{} size", path.display());
    want
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m })
        .0
}

/// At every position the argmax matches `want` and every logit is within
/// [`TOLERANCE`].
fn check(label: &str, got: &[f32], want: &[f32], vocab: usize) {
    for pos in 0..PROMPT.len() {
        let g = &got[pos * vocab..(pos + 1) * vocab];
        let w = &want[pos * vocab..(pos + 1) * vocab];
        let diff = g
            .iter()
            .zip(w)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "{label} pos {pos}: max abs diff {diff}, argmax {} vs {}",
            argmax(g),
            argmax(w)
        );
        assert_eq!(argmax(g), argmax(w), "{label} pos {pos}: argmax differs");
        assert!(
            diff < TOLERANCE,
            "{label} pos {pos}: logits differ by {diff}"
        );
    }
}

/// POST `body` to `uri` and return the status and the JSON body.
async fn post(router: axum::Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap_or_else(|e| panic!("request {uri}: {e}"));
    let response = router
        .oneshot(request)
        .await
        .unwrap_or_else(|e| panic!("{uri}: {e}"));
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap_or_else(|e| panic!("{uri} body: {e}"));
    let value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("{uri}: {e}: {}", String::from_utf8_lossy(&bytes)));
    (status, value)
}

/// GET `uri` and return the JSON body after checking for a 200.
async fn get(router: axum::Router, uri: &str) -> Value {
    let request = Request::builder()
        .uri(uri)
        .body(Body::empty())
        .unwrap_or_else(|e| panic!("request {uri}: {e}"));
    let response = router
        .oneshot(request)
        .await
        .unwrap_or_else(|e| panic!("{uri}: {e}"));
    assert_eq!(response.status(), StatusCode::OK, "{uri}");
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap_or_else(|e| panic!("{uri} body: {e}"));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{uri}: {e}"))
}

/// When `name` was last accessed, from the scheduler's loaded models.
async fn last_access(scheduler: &Scheduler<CpuRuntime>, name: &str) -> std::time::Instant {
    scheduler
        .list_loaded()
        .await
        .into_iter()
        .find(|m| m.name == name)
        .unwrap_or_else(|| panic!("{name} is not loaded"))
        .last_accessed
}

#[tokio::test(flavor = "multi_thread")]
async fn kev_variant_matches_peft_and_leaves_the_base_unchanged() {
    let (Some(base_dir), Some(adapter_dir), Some(kev_ref), Some(base_ref)) = (
        env_path(ENV_BASE),
        env_path(ENV_ADAPTER),
        env_path(ENV_REF),
        env_path(ENV_BASE_REF),
    ) else {
        return;
    };
    // The model directory holds the base under its own name.
    let base_dir = base_dir
        .canonicalize()
        .unwrap_or_else(|e| panic!("{}: {e}", base_dir.display()));
    let name = base_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_else(|| panic!("{} has no directory name", base_dir.display()))
        .to_string();
    let model_dir = tempfile::tempdir().unwrap_or_else(|e| panic!("temp dir: {e}"));
    std::os::unix::fs::symlink(&base_dir, model_dir.path().join(&name))
        .unwrap_or_else(|e| panic!("link {}: {e}", base_dir.display()));

    let device = CpuDevice::new();
    let f32_request = DtypeRequest {
        dtype: DType::F32,
        source: DtypeSource::Flag,
    };
    let scheduler = Arc::new(
        Scheduler::<CpuRuntime>::new(model_dir.path().to_path_buf(), device.clone())
            .with_dtype(Some(f32_request)),
    );

    // The base, before any adapter.
    let base = scheduler
        .get_executor(&name)
        .await
        .unwrap_or_else(|e| panic!("load {name}: {e:#}"));
    let vocab = base.model().vocab_size();
    let base_before = prefill_logits(base.model(), &device);
    check("base", &base_before, &reference(&base_ref, vocab), vocab);

    // The variant, derived from the loaded base through the registry.
    let spec = AdapterSpec::new("kev", &adapter_dir).unwrap_or_else(|e| panic!("spec: {e}"));
    let info = scheduler
        .register_variant(&name, spec)
        .await
        .unwrap_or_else(|e| panic!("register kev: {e}"));
    let id = format!("{name}:kev");
    assert_eq!(info.id, id);
    assert_eq!(info.base, name);
    println!(
        "variant {} rank {} over {} modules",
        info.id, info.rank, info.adapted_modules
    );
    assert!(info.adapted_modules > 0, "the adapter adapted nothing");
    let variant: Arc<Executor<CpuRuntime>> = scheduler
        .get_executor(&id)
        .await
        .unwrap_or_else(|e| panic!("get {id}: {e:#}"));
    assert!(
        !Arc::ptr_eq(&variant, &base),
        "the variant is the base executor"
    );
    assert_eq!(
        variant.model().adapted_modules().len(),
        info.adapted_modules
    );
    assert!(
        base.model().adapted_modules().is_empty(),
        "the base was adapted"
    );
    check(
        "kev",
        &prefill_logits(variant.model(), &device),
        &reference(&kev_ref, vocab),
        vocab,
    );

    // The base is unchanged: bit-identical to its logits before the derive.
    let base_after = prefill_logits(base.model(), &device);
    let moved = base_before
        .iter()
        .zip(&base_after)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(moved, 0, "{moved} base logits moved after the derive");

    // Selection: `lora_adapter` with the base name picks the variant.
    let selected = scheduler
        .select_model(&name, Some("kev"), None)
        .unwrap_or_else(|e| panic!("select: {e}"));
    assert_eq!(selected, id);
    let again = scheduler
        .get_executor(&selected)
        .await
        .unwrap_or_else(|e| panic!("{e:#}"));
    assert!(
        Arc::ptr_eq(&again, &variant),
        "selection reached another executor"
    );

    // The routes over the same scheduler.
    let metrics = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    let state = Arc::new(AppState::new(Arc::clone(&scheduler), metrics));
    let router = blazr::server::api_routes().with_state(state);

    let (status, body) = post(
        router.clone(),
        "/v1/completions",
        json!({"model": name, "prompt": "Hello", "max_tokens": 1, "lora_adapter": "nope"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["type"], "lora_not_found", "{body}");

    let variant_seen = last_access(&scheduler, &id).await;
    let (status, body) = post(
        router.clone(),
        "/v1/completions",
        json!({"model": id, "prompt": "Hello", "max_tokens": 1, "temperature": 0.0}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        last_access(&scheduler, &id).await > variant_seen,
        "the variant did not serve it"
    );

    let models = get(router.clone(), "/v1/models").await;
    let listed = models["data"]
        .as_array()
        .and_then(|data| data.iter().find(|m| m["id"] == id.as_str()))
        .unwrap_or_else(|| panic!("/v1/models lists no {id}: {models}"));
    assert_eq!(listed["lora"]["base"], name.as_str());
    assert_eq!(listed["lora"]["rank"], info.rank);
    assert_eq!(listed["lora"]["adapted_modules"], info.adapted_modules);
    let loras = get(router, &format!("/v1/lora?model={name}")).await;
    assert_eq!(loras["adapters"][0]["id"], id.as_str(), "{loras}");
    assert_eq!(loras["adapters"][0]["loaded"], true, "{loras}");
}
