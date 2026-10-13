//! Per-executor CUDA graph sessions: one captured decode step and the
//! full-capacity caches it addresses, kept across requests.
//!
//! A request resets the caches in place, prefills its prompt eagerly into the
//! same buffers, then replays the graph. Every buffer the graph reads keeps
//! its device address, so one capture serves every request with the same
//! capacity. The Llama session build lives here; the `qwen35` build lives in
//! `cuda_graphs_qwen35.rs` beside its arena logic.

#![cfg(feature = "cuda")]

use anyhow::{anyhow, Result};

use boostr::autograd::Var;
use boostr::inference::decode_graph::{
    argmax_to_buf, DecodeGraph, DeviceScalars, Qwen35DecodeGraph,
};
use boostr::inference::{LayeredGdnState, LayeredKvCache};
use boostr::{CudaRuntime, Runtime, Tensor};

use crate::tokenizer::TokenTextStream;

use crate::engine::executor::Executor;
use crate::engine::types::{FinishReason, GeneratedToken};

/// Captured Llama decode step plus the KV cache its kernels address.
///
/// `decode` is declared first so the graph drops before the cache it writes.
pub struct LlamaGraphSession {
    /// The captured graph and its stable-address replay buffers.
    pub decode: DecodeGraph,
    /// Full-capacity KV cache the graph reads and writes.
    pub kv_cache: LayeredKvCache<CudaRuntime>,
    /// Capacity both caches and the graph were built for.
    pub capacity: usize,
}

/// Captured `qwen35` decode step plus the caches its kernels address.
///
/// `decode` is declared first so the graph drops before the caches it writes.
pub struct Qwen35GraphSession {
    /// The captured graph and its stable-address replay buffers.
    pub decode: Qwen35DecodeGraph,
    /// Full-capacity KV cache for the attention layers.
    pub kv_cache: LayeredKvCache<CudaRuntime>,
    /// In-place GDN state for the GDN layers.
    pub gdn_state: LayeredGdnState<CudaRuntime>,
    /// Capacity both caches and the graph were built for.
    pub capacity: usize,
}

impl Executor<CudaRuntime> {
    /// Count of graph captures on this executor, across both arms.
    pub fn graph_captures(&self) -> usize {
        self.graph_captures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record one capture.
    pub(in crate::engine) fn note_graph_capture(&self) {
        self.note_graph_captures(1);
    }

    /// Record `count` captures, as a driver build reports them.
    pub(in crate::engine) fn note_graph_captures(&self, count: usize) {
        self.graph_captures
            .fetch_add(count, std::sync::atomic::Ordering::Relaxed);
    }

    /// Greedy first token from prefill logits, read through one event.
    pub(in crate::engine) fn first_token_from_logits(
        &self,
        logits: &Tensor<CudaRuntime>,
    ) -> Result<u32> {
        let token_gpu = self.argmax_on_gpu(logits)?;
        let event = token_gpu
            .record_event()
            .map_err(|e| anyhow!("Record event failed: {}", e))?;
        Self::read_token_id(&token_gpu, event)
    }

    /// The `GeneratedToken` for an EOS step. It carries only the bytes the
    /// decoder still holds.
    pub(in crate::engine) fn eos_token(
        token_id: u32,
        text: &mut TokenTextStream,
    ) -> GeneratedToken {
        GeneratedToken {
            token_id,
            text: text.finish(),
            logprob: None,
            top_logprobs: None,
            finish_reason: Some(FinishReason::Eos),
        }
    }

    /// The `GeneratedToken` for a non-EOS decode step. `text` is the
    /// request's own decoder, and `finish_reason` marks the last token.
    pub(in crate::engine) fn text_token(
        &self,
        text: &mut TokenTextStream,
        token_id: u32,
        finish_reason: Option<FinishReason>,
    ) -> GeneratedToken {
        let text = text.next_text(self.tokenizer(), token_id, false, finish_reason.is_some());
        GeneratedToken {
            token_id,
            text,
            logprob: None,
            top_logprobs: None,
            finish_reason,
        }
    }

    /// Reset the session caches and eager-prefill `input` at position 0.
    ///
    /// Sets the decode position and writes the device scalars for the first
    /// replay. Returns the prefill logits.
    pub(in crate::engine) fn llama_session_prefill(
        &self,
        session: &mut LlamaGraphSession,
        input: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        session.kv_cache.reset();
        let logits = self
            .model()
            .forward_with_kv_cache(input, &mut session.kv_cache, 0)
            .map_err(|e| anyhow!("Prefill forward pass failed: {}", e))?;
        let client = CudaRuntime::default_client(self.device());
        let seq_len = session.kv_cache.seq_len();
        session.decode.seq_len = seq_len;
        session
            .decode
            .device_scalars
            .update(&client, &[seq_len])
            .map_err(|e| anyhow!("DeviceScalars update failed: {}", e))?;
        Ok(logits)
    }

    /// Build a Llama session for `capacity`: full-capacity cache, eager
    /// prefill, stable buffers, an uncaptured warmup pass, reset plus
    /// re-prefill into the same cache, then capture.
    ///
    /// Returns the session and the prefill logits of the re-prefill.
    pub(in crate::engine) fn llama_build_session(
        &self,
        input: &Tensor<CudaRuntime>,
        capacity: usize,
    ) -> Result<(LlamaGraphSession, Tensor<CudaRuntime>)> {
        let num_layers = self.model().num_layers();
        let num_kv_heads = self.model().num_kv_heads().unwrap_or(8);
        let head_dim = self.model().head_dim().unwrap_or(64);
        let half_dim = head_dim / 2;
        let kv_dtype = self.config().kv_cache_dtype()?;
        let device = self.device();

        let mut kv_cache = LayeredKvCache::new_positional(
            num_layers,
            1,
            num_kv_heads,
            capacity,
            capacity,
            head_dim,
            kv_dtype,
            device,
        )
        .map_err(|e| {
            anyhow!(
                "Failed to create full-capacity KV cache ({capacity}): {}",
                e
            )
        })?;

        let t0 = std::time::Instant::now();
        let warm_logits = self
            .model()
            .forward_with_kv_cache(input, &mut kv_cache, 0)
            .map_err(|e| anyhow!("Prefill forward pass failed: {}", e))?;
        tracing::info!(
            "Graph-mode prefill: {:?} (seq_len={})",
            t0.elapsed(),
            kv_cache.seq_len()
        );
        drop(warm_logits);

        let (rope_cos_var, rope_sin_var) = self
            .model()
            .rope_caches()
            .ok_or_else(|| anyhow!("Model has no RoPE cache — cannot use CUDA graph mode"))?;
        let rope_cos_cache = rope_cos_var.tensor().clone();
        let rope_sin_cache = rope_sin_var.tensor().clone();

        let client = CudaRuntime::default_client(device);

        // Stable-address decode inputs, all allocated before capture.
        let token_buf = Tensor::<CudaRuntime>::zeros(&[1, 1], boostr::DType::I64, device)?;
        let cos_slice = Var::new(
            Tensor::<CudaRuntime>::zeros(&[1, half_dim], rope_cos_cache.dtype(), device)?,
            false,
        );
        let sin_slice = Var::new(
            Tensor::<CudaRuntime>::zeros(&[1, half_dim], rope_sin_cache.dtype(), device)?,
            false,
        );
        let mut device_scalars = DeviceScalars::new(&[kv_cache.seq_len()], device)?;
        let next_token_buf = Tensor::<CudaRuntime>::zeros(&[1], boostr::DType::I64, device)?;

        // Warmup pass: JIT every kernel, not captured.
        device_scalars
            .update(&client, &[kv_cache.seq_len()])
            .map_err(|e| anyhow!("DeviceScalars warmup update failed: {}", e))?;
        let warmup_logits = self
            .model()
            .forward_graph_mode(
                &token_buf,
                &mut kv_cache,
                &device_scalars,
                &cos_slice,
                &sin_slice,
            )
            .map_err(|e| anyhow!("Graph-mode warmup forward failed: {}", e))?;
        argmax_to_buf(&client, &warmup_logits, &next_token_buf)
            .map_err(|e| anyhow!("Warmup argmax_to_buf failed: {}", e))?;
        drop(warmup_logits);

        // The warmup wrote the cache in place: reset and prefill again so
        // capture starts from the prompt state at the same addresses.
        kv_cache.reset();
        let prefill_logits = self
            .model()
            .forward_with_kv_cache(input, &mut kv_cache, 0)
            .map_err(|e| anyhow!("Re-prefill for capture failed: {}", e))?;

        let capture_seq_len = kv_cache.seq_len();
        device_scalars
            .update(&client, &[capture_seq_len])
            .map_err(|e| anyhow!("DeviceScalars pre-capture update failed: {}", e))?;
        device_scalars
            .update_rope_slices(
                &client,
                &rope_cos_cache,
                &rope_sin_cache,
                &cos_slice,
                &sin_slice,
                capture_seq_len,
                half_dim,
            )
            .map_err(|e| anyhow!("RoPE pre-capture update failed: {}", e))?;

        let graph =
            CudaRuntime::capture_graph_into(&client, &[&token_buf], &[&next_token_buf], |c| {
                let logits = self
                    .model()
                    .forward_graph_mode(
                        &token_buf,
                        &mut kv_cache,
                        &device_scalars,
                        &cos_slice,
                        &sin_slice,
                    )
                    .map_err(|e| {
                        boostr::NumrError::Backend(format!("Capture forward failed: {e}"))
                    })?;
                argmax_to_buf(c, &logits, &next_token_buf).map_err(|e| {
                    boostr::NumrError::Backend(format!("Capture argmax_to_buf failed: {e}"))
                })
            })
            .map_err(|e| anyhow!("CUDA graph capture failed: {}", e))?;
        self.note_graph_capture();
        tracing::info!("CUDA graph captured (seq_len={})", capture_seq_len);

        let decode = DecodeGraph {
            graph,
            device_scalars,
            token_buf,
            cos_slice: cos_slice.tensor().clone(),
            sin_slice: sin_slice.tensor().clone(),
            rope_cos_cache,
            rope_sin_cache,
            next_token_buf,
            // `DecodeGraph` keeps the half dimension in `head_dim`.
            head_dim: half_dim,
            seq_len: capture_seq_len,
        };
        let session = LlamaGraphSession {
            decode,
            kv_cache,
            capacity,
        };
        Ok((session, prefill_logits))
    }
}
