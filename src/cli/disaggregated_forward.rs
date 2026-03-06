//! Forward function builders and HTTP server for disaggregated prefill/decode.
//!
//! Contains:
//! - `build_prefill_fn` / `run_prefill_forward` — prefill worker compute
//! - `build_decode_step_fn` / `run_decode_step` — decode worker compute
//! - `run_router_http_server` — HTTP server for the router role
//! - `resolve_model_path` — model path resolution helper

use anyhow::Result;
use std::sync::Arc;

use crate::distributed::disaggregated::{DecodeStepFn, PrefillFn};
use crate::distributed::kv_serialize;
use crate::distributed::DisaggRouter;

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

// ─────────────────────────────────────────────────────────────────────────────
// Forward function builders
// ─────────────────────────────────────────────────────────────────────────────

/// Build the prefill forward function for the given loaded model.
///
/// The closure:
/// 1. Decodes `token_ids_bytes` as little-endian `i64` token IDs.
/// 2. Runs the model's full forward pass (embed + all layers + head).
/// 3. Serializes the resulting KV cache for network transfer.
///
/// Returns `(activation_bytes, kv_cache_bytes)` as required by `PrefillWorker`.
pub fn build_prefill_fn(
    model: Arc<boostr::model::LoadedModel<ServerRuntime>>,
    device: <ServerRuntime as boostr::Runtime>::Device,
) -> PrefillFn {
    use boostr::inference::{LayeredKvCache, LayeredKvCacheConfig};
    use boostr::DType;

    use std::sync::Mutex;

    let num_layers = model.num_layers();
    let num_kv_heads = model.num_kv_heads().unwrap_or(1);
    let head_dim = model.head_dim().unwrap_or(64);
    let hidden_size = model.hidden_size();

    // KV cache shared across calls (one request at a time per worker).
    let kv_cache: Option<Mutex<LayeredKvCache<ServerRuntime>>> = if model.needs_kv_cache() {
        let config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads,
            initial_capacity: 4096,
            max_seq_len: 32768,
            head_dim,
            dtype: DType::F32,
        };
        match LayeredKvCache::<ServerRuntime>::new(num_layers, &config, &device) {
            Ok(c) => Some(Mutex::new(c)),
            Err(e) => {
                tracing::error!("Prefill: failed to create KV cache: {}", e);
                None
            }
        }
    } else {
        None
    };

    let kv_cache = Arc::new(kv_cache);
    let model = Arc::clone(&model);
    let device_clone = device.clone();

    Box::new(
        move |token_ids_bytes: &[u8], seq_len: usize| -> (Vec<u8>, Vec<u8>) {
            // Reset KV cache for this new request.
            if let Some(ref mutex) = *kv_cache {
                if let Ok(mut cache) = mutex.lock() {
                    cache.reset();
                }
            }

            let result = run_prefill_forward(
                &model,
                &device_clone,
                token_ids_bytes,
                seq_len,
                hidden_size,
                &kv_cache,
            );

            match result {
                Ok((activation, kv_bytes)) => (activation, kv_bytes),
                Err(e) => {
                    tracing::error!("Prefill forward failed: {}", e);
                    (Vec::new(), Vec::new())
                }
            }
        },
    )
}

/// Run the prefill forward pass and serialize the resulting KV cache.
fn run_prefill_forward(
    model: &boostr::model::LoadedModel<ServerRuntime>,
    device: &<ServerRuntime as boostr::Runtime>::Device,
    token_ids_bytes: &[u8],
    _seq_len: usize,
    _hidden_size: usize,
    kv_cache: &Arc<Option<std::sync::Mutex<boostr::inference::LayeredKvCache<ServerRuntime>>>>,
) -> Result<(Vec<u8>, Vec<u8>)> {
    use boostr::Tensor;

    if !token_ids_bytes
        .len()
        .is_multiple_of(std::mem::size_of::<i64>())
    {
        anyhow::bail!(
            "Token ID buffer length {} is not a multiple of 8",
            token_ids_bytes.len()
        );
    }

    let token_ids: &[i64] = bytemuck::cast_slice(token_ids_bytes);
    let seq_len = token_ids.len();

    let input_tensor = Tensor::<ServerRuntime>::from_slice(token_ids, &[1, seq_len], device);

    // Embed tokens.
    let hidden = model.forward_embed(&input_tensor)?;

    // Run all layers through the KV cache.
    let (hidden_out, _prev_mlp_out) = if let Some(ref mutex) = **kv_cache {
        let mut cache = mutex
            .lock()
            .map_err(|_| anyhow::anyhow!("KV cache lock poisoned"))?;
        let position = cache.seq_len();
        let num_layers = model.num_layers();
        model.forward_layers_range(hidden, None, &mut cache, 0, num_layers, position)?
    } else {
        (hidden, None)
    };

    // Serialize the current KV cache state.
    let kv_bytes = if let Some(ref mutex) = **kv_cache {
        let cache = mutex
            .lock()
            .map_err(|_| anyhow::anyhow!("KV cache lock poisoned"))?;
        kv_serialize::serialize_kv_cache::<ServerRuntime>(&cache)
    } else {
        // No KV cache (e.g. Mamba/SSM): return empty payload.
        Vec::new()
    };

    // Activation: final hidden state serialised as raw f32 bytes.
    // The decode worker doesn't strictly need this (it uses the KV cache),
    // but we return it for completeness / future use.
    let hidden_f32: Vec<f32> = hidden_out.tensor().to_vec();
    let activation_bytes = bytemuck::cast_slice::<f32, u8>(&hidden_f32).to_vec();

    Ok((activation_bytes, kv_bytes))
}

/// Build the decode step function for the given loaded model.
///
/// The closure:
/// 1. Deserializes `kv_cache_bytes` into a `LayeredKvCache`.
/// 2. Embeds `last_token_id`, runs one decode step at `position`.
/// 3. Samples the next token (greedy argmax for simplicity).
/// 4. Serializes the updated KV cache and returns `(next_token_id, updated_kv_bytes)`.
///
/// EOS is signalled by returning `i64::MIN` as the token ID.
pub fn build_decode_step_fn(
    model: Arc<boostr::model::LoadedModel<ServerRuntime>>,
    device: <ServerRuntime as boostr::Runtime>::Device,
) -> DecodeStepFn {
    let model = Arc::clone(&model);
    let device_clone = device.clone();

    Box::new(
        move |kv_cache_bytes: &[u8], last_token_id: i64, _position: u32| -> (i64, Vec<u8>) {
            let result = run_decode_step(&model, &device_clone, kv_cache_bytes, last_token_id);

            match result {
                Ok(pair) => pair,
                Err(e) => {
                    tracing::error!("Decode step failed: {}", e);
                    // Signal EOS on error so the router doesn't hang.
                    (i64::MIN, Vec::new())
                }
            }
        },
    )
}

/// Run a single decode step and return the next token plus the updated KV cache.
fn run_decode_step(
    model: &boostr::model::LoadedModel<ServerRuntime>,
    device: &<ServerRuntime as boostr::Runtime>::Device,
    kv_cache_bytes: &[u8],
    last_token_id: i64,
) -> Result<(i64, Vec<u8>)> {
    use boostr::Tensor;

    // Deserialize the KV cache from bytes.
    let mut kv_cache = kv_serialize::deserialize_kv_cache::<ServerRuntime>(kv_cache_bytes, device)?;

    // Embed the single token.
    let token_slice = [last_token_id];
    let input_tensor = Tensor::<ServerRuntime>::from_slice(&token_slice, &[1, 1], device);
    let hidden = model.forward_embed(&input_tensor)?;

    // Run all layers with the restored KV cache.
    let position = kv_cache.seq_len();
    let num_layers = model.num_layers();
    let (hidden_out, _prev_mlp) =
        model.forward_layers_range(hidden, None, &mut kv_cache, 0, num_layers, position)?;

    // Compute logits.
    let logits = model.forward_head(hidden_out, None)?;

    // Greedy argmax over the last token position.
    let logits_f32: Vec<f32> = logits.to_vec::<f32>();
    let vocab_size = model.vocab_size();

    // Logits shape: [batch=1, seq=1, vocab_size] — last `vocab_size` elements.
    let logit_slice = if logits_f32.len() >= vocab_size {
        &logits_f32[logits_f32.len() - vocab_size..]
    } else {
        &logits_f32[..]
    };

    let next_token = logit_slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i64)
        .unwrap_or(0);

    // Serialize the updated KV cache.
    let updated_kv = kv_serialize::serialize_kv_cache::<ServerRuntime>(&kv_cache);

    Ok((next_token, updated_kv))
}

// ─────────────────────────────────────────────────────────────────────────────
// Router HTTP server
// ─────────────────────────────────────────────────────────────────────────────

/// Start the HTTP server for the router role.
///
/// The router's HTTP server accepts `/v1/completions`-style requests and
/// routes them through `DisaggRouter::route_request`, then streams the
/// generated tokens back as a text response.
pub async fn run_router_http_server(
    router: Arc<DisaggRouter>,
    server_config: crate::config::ServerConfig,
) -> Result<()> {
    use axum::{extract::State, response::IntoResponse, routing::post, Json, Router};

    #[derive(serde::Deserialize)]
    struct CompletionRequest {
        prompt: String,
        #[serde(default = "default_max_tokens")]
        max_tokens: u32,
        #[serde(default)]
        session_id: Option<String>,
    }

    fn default_max_tokens() -> u32 {
        256
    }

    #[derive(serde::Serialize)]
    struct CompletionResponse {
        tokens: Vec<i64>,
        token_count: usize,
    }

    async fn handle_completion(
        State(router): State<Arc<DisaggRouter>>,
        Json(req): Json<CompletionRequest>,
    ) -> impl IntoResponse {
        // Simple tokenization placeholder: encode prompt as UTF-8 bytes
        // mapped to i64 codepoints. In production this would call splintr.
        let token_ids: Vec<i64> = req.prompt.chars().map(|c| c as i64).collect();

        let seq_len = token_ids.len() as u32;
        let token_ids_bytes: Vec<u8> = token_ids.iter().flat_map(|&t| t.to_le_bytes()).collect();

        let session_key = req.session_id.as_deref();

        match router
            .route_request(&token_ids_bytes, seq_len, req.max_tokens, session_key)
            .await
        {
            Ok(tokens) => {
                let count = tokens.len();
                axum::Json(CompletionResponse {
                    tokens,
                    token_count: count,
                })
                .into_response()
            }
            Err(e) => {
                tracing::error!("Router request failed: {}", e);
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    axum::Json(serde_json::json!({
                        "error": { "message": e.to_string(), "type": "server_error" }
                    })),
                )
                    .into_response()
            }
        }
    }

    let app = Router::new()
        .route("/v1/completions", post(handle_completion))
        .route(
            "/health",
            axum::routing::get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }),
        )
        .with_state(router);

    let bind_addr = format!("{}:{}", server_config.host, server_config.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| {
            anyhow::anyhow!("Failed to bind router HTTP server to {}: {}", bind_addr, e)
        })?;

    tracing::info!(
        addr = %bind_addr,
        "Disaggregated router HTTP server listening"
    );

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Router HTTP server error: {}", e))?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

pub fn resolve_model_path(model: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(model);
    if p.is_absolute() || p.exists() {
        p.to_path_buf()
    } else {
        let model_dir = std::env::var("BLAZR_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("./models"));
        model_dir.join(model)
    }
}
