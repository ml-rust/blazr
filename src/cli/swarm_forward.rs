//! Forward function builders for swarm pipeline-parallel workers.
//!
//! Contains:
//! - `serialize_activation` / `deserialize_activation` — wire format helpers
//! - `build_forward_fn` — constructs the per-worker forward closure
//! - `run_forward` — inner forward pass implementation

use anyhow::Result;
use std::sync::Arc;

#[cfg(feature = "cuda")]
type ServerRuntime = boostr::CudaRuntime;
#[cfg(not(feature = "cuda"))]
type ServerRuntime = boostr::CpuRuntime;

// ─────────────────────────────────────────────────────────────────────────────
// Activation serialization helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Serialize a hidden state and optional MLP residual into a byte buffer.
///
/// Wire format: `[u32 hidden_numel][u32 mlp_numel][hidden f32...][mlp f32...]`
pub fn serialize_activation(hidden: &[f32], prev_mlp: Option<&[f32]>) -> Vec<u8> {
    let hidden_n = hidden.len() as u32;
    let mlp_n = prev_mlp.map(|s| s.len() as u32).unwrap_or(0);
    let total_bytes = 8 + (hidden_n as usize + mlp_n as usize) * 4;
    let mut buf = Vec::with_capacity(total_bytes);
    buf.extend_from_slice(&hidden_n.to_le_bytes());
    buf.extend_from_slice(&mlp_n.to_le_bytes());
    buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(hidden));
    if let Some(mlp) = prev_mlp {
        buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(mlp));
    }
    buf
}

/// Deserialize a hidden state and optional MLP residual from a byte buffer.
///
/// Returns `(hidden_f32, Option<mlp_f32>)`.
pub fn deserialize_activation(bytes: &[u8]) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
    if bytes.len() < 8 {
        anyhow::bail!("Activation buffer too short: {} bytes", bytes.len());
    }
    let hidden_n = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let mlp_n = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let needed = 8 + (hidden_n + mlp_n) * 4;
    if bytes.len() < needed {
        anyhow::bail!(
            "Activation buffer truncated: need {} bytes, have {}",
            needed,
            bytes.len()
        );
    }
    let hidden_bytes = &bytes[8..8 + hidden_n * 4];
    let hidden: Vec<f32> = bytemuck::cast_slice::<u8, f32>(hidden_bytes).to_vec();
    let prev_mlp = if mlp_n > 0 {
        let mlp_bytes = &bytes[8 + hidden_n * 4..8 + (hidden_n + mlp_n) * 4];
        Some(bytemuck::cast_slice::<u8, f32>(mlp_bytes).to_vec())
    } else {
        None
    };
    Ok((hidden, prev_mlp))
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward function builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build the forward function for a worker stage.
///
/// The closure captures the loaded model, device, and layer assignment.
/// It deserializes activations from the previous stage (or token IDs from the
/// leader for the first stage), runs the assigned model layers with KV cache,
/// and serializes the output activations (or logits for the final stage).
///
/// KV cache is protected by a Mutex so the `Fn` closure can call it repeatedly.
pub fn build_forward_fn(
    model: Arc<boostr::model::LoadedModel<ServerRuntime>>,
    device: <ServerRuntime as boostr::Runtime>::Device,
    assignment: crate::distributed::transport::LayerAssignment,
    has_prev_rank: bool,
    has_next_rank: bool,
) -> crate::distributed::worker::ForwardFn {
    use boostr::inference::{LayeredKvCache, LayeredSsmState};
    use boostr::DType;
    use std::sync::Mutex;

    let num_layers = model.num_layers();
    let num_kv_heads = model.num_kv_heads().unwrap_or(1);
    let head_dim = model.head_dim().unwrap_or(64);
    let hidden_size = model.hidden_size();
    let start_layer = assignment.start_layer as usize;
    let end_layer = assignment.end_layer as usize;

    // Allocate KV cache (covers all layers; only start_layer..end_layer slots are used)
    let kv_cache: Option<Mutex<LayeredKvCache<ServerRuntime>>> = if model.needs_kv_cache() {
        match LayeredKvCache::<ServerRuntime>::new_positional(
            num_layers,
            1, // batch_size = 1
            num_kv_heads,
            4096,  // initial capacity
            32768, // max_seq_len — generous upper bound
            head_dim,
            DType::F32,
            &device,
        ) {
            Ok(cache) => Some(Mutex::new(cache)),
            Err(e) => {
                tracing::error!("Failed to create KV cache: {}", e);
                None
            }
        }
    } else {
        None
    };

    // SSM state for Mamba2 models
    let ssm_state: Option<Mutex<LayeredSsmState<ServerRuntime>>> = if model.needs_ssm_state() {
        if let Some(mamba_cfg) = model.mamba_config() {
            Some(Mutex::new(LayeredSsmState::new(
                num_layers,
                1, // batch_size = 1
                mamba_cfg,
                DType::F32,
                &device,
            )))
        } else {
            None
        }
    } else {
        None
    };

    let kv_cache = Arc::new(kv_cache);
    let ssm_state = Arc::new(ssm_state);
    let model = Arc::clone(&model);
    let device = device.clone();

    Box::new(move |input_bytes: &[u8], size_hint: usize| -> Vec<u8> {
        let result = run_forward(
            &model,
            &device,
            input_bytes,
            size_hint,
            has_prev_rank,
            has_next_rank,
            start_layer,
            end_layer,
            hidden_size,
            &kv_cache,
            &ssm_state,
        );
        match result {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::error!("Worker forward pass failed: {}", e);
                Vec::new()
            }
        }
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Inner forward pass
// ─────────────────────────────────────────────────────────────────────────────

/// Run the forward pass for the worker's assigned layers.
///
/// Handles all three stage types:
/// - First stage (no prev_rank): input is i64 token IDs → embed → layers → serialize activation
/// - Intermediate stage: input is serialized activation → layers → serialize activation
/// - Last stage (no next_rank): input → layers → norm + lm_head → serialize logits as raw f32
#[allow(clippy::too_many_arguments)]
fn run_forward(
    model: &boostr::model::LoadedModel<ServerRuntime>,
    device: &<ServerRuntime as boostr::Runtime>::Device,
    input_bytes: &[u8],
    _size_hint: usize,
    has_prev_rank: bool,
    has_next_rank: bool,
    start_layer: usize,
    end_layer: usize,
    hidden_size: usize,
    kv_cache: &Arc<Option<std::sync::Mutex<boostr::inference::LayeredKvCache<ServerRuntime>>>>,
    _ssm_state: &Arc<Option<std::sync::Mutex<boostr::inference::LayeredSsmState<ServerRuntime>>>>,
) -> Result<Vec<u8>> {
    use boostr::autograd::Var;
    use boostr::Tensor;

    // --- Decode input ---
    let (hidden_var, prev_mlp_var): (Var<ServerRuntime>, Option<Var<ServerRuntime>>) =
        if !has_prev_rank {
            // First stage: input_bytes are i64 token IDs (little-endian)
            if !input_bytes.len().is_multiple_of(std::mem::size_of::<i64>()) {
                anyhow::bail!(
                    "Token ID buffer length {} is not a multiple of 8",
                    input_bytes.len()
                );
            }
            let token_ids: &[i64] = bytemuck::cast_slice(input_bytes);
            let seq_len = token_ids.len();
            let input_tensor =
                Tensor::<ServerRuntime>::from_slice(token_ids, &[1, seq_len], device);
            // Embed tokens → hidden state
            let hidden = model.forward_embed(&input_tensor)?;
            (hidden, None)
        } else {
            // Intermediate or last stage: input_bytes is serialized activation
            let (hidden_f32, prev_mlp_f32) = deserialize_activation(input_bytes)?;
            let batch_size = 1usize;
            let seq_len = hidden_f32.len() / hidden_size;
            if hidden_f32.len() != batch_size * seq_len * hidden_size {
                anyhow::bail!(
                    "Hidden state size mismatch: got {} f32, expected batch*seq*hidden={}*{}*{}",
                    hidden_f32.len(),
                    batch_size,
                    seq_len,
                    hidden_size
                );
            }
            let hidden_tensor = Tensor::<ServerRuntime>::from_slice(
                &hidden_f32,
                &[batch_size, seq_len, hidden_size],
                device,
            );
            let hidden_var = Var::new(hidden_tensor, false);
            let mlp_var = prev_mlp_f32.map(|mlp_f32| {
                let mlp_tensor = Tensor::<ServerRuntime>::from_slice(
                    &mlp_f32,
                    &[batch_size, seq_len, hidden_size],
                    device,
                );
                Var::new(mlp_tensor, false)
            });
            (hidden_var, mlp_var)
        };

    // --- Run assigned transformer layers ---
    let (hidden_out, prev_mlp_out) = {
        if let Some(ref mutex) = **kv_cache {
            let mut cache = mutex
                .lock()
                .map_err(|_| anyhow::anyhow!("KV cache lock poisoned"))?;
            let position = cache.seq_len();
            model.forward_layers_range(
                hidden_var,
                prev_mlp_var,
                &mut cache,
                start_layer,
                end_layer,
                position,
            )?
        } else {
            // Mamba/SSM path: no KV cache. Return activation unchanged for now since
            // SSM pipeline parallel is handled via forward_with_ssm_state on full model.
            (hidden_var, prev_mlp_var)
        }
    };

    // --- Encode output ---
    if !has_next_rank {
        // Last stage: apply norm + lm_head, output raw f32 logits
        let logits = model.forward_head(hidden_out, prev_mlp_out)?;
        let logits_f32: Vec<f32> = logits.to_vec();
        Ok(bytemuck::cast_slice::<f32, u8>(&logits_f32).to_vec())
    } else {
        // Not the last stage: serialize hidden state + prev_mlp_out and forward to next stage
        let hidden_f32: Vec<f32> = hidden_out.tensor().to_vec();
        let mlp_f32: Option<Vec<f32>> = prev_mlp_out.as_ref().map(|v| v.tensor().to_vec());
        Ok(serialize_activation(&hidden_f32, mlp_f32.as_deref()))
    }
}
