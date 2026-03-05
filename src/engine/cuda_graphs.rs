//! CUDA graph-accelerated greedy decode loop.
//!
//! The config flag `inference.graphs` is backend-agnostic. For now the graph-mode
//! forward pass (`forward_graph_mode`) uses CUDA-specific kernel APIs
//! (device-ptr seq_len, kv_insert kernel), so this impl block is CUDA-only.
//! When ROCm/Metal graph support is added, a parallel impl block will be added
//! for those runtimes using their respective graph-mode ops.

#[cfg(feature = "cuda")]
use anyhow::{anyhow, Result};

#[cfg(feature = "cuda")]
use async_stream::stream;

#[cfg(feature = "cuda")]
use crate::config::parse_dtype;

#[cfg(feature = "cuda")]
use super::executor::Executor;
#[cfg(feature = "cuda")]
use super::types::{FinishReason, GeneratedToken};

#[cfg(feature = "cuda")]
impl Executor<boostr::CudaRuntime> {
    /// Run greedy decode using compute graph capture+replay (CUDA backend).
    ///
    /// After prompt prefill, captures the single-token decode forward pass as a
    /// compute graph. Replay cost is ~5µs per token instead of ~13ms kernel dispatch.
    ///
    /// # Preconditions
    /// - Model must be Llama (Mamba2/Hybrid graph mode not yet implemented)
    /// - `gen_config.is_greedy()` must be true (graph replay is deterministic)
    /// - `config.inference.paged_attention` must be false
    pub fn generate_with_graphs<'a>(
        &'a self,
        prompt: &'a str,
        gen_config: &'a crate::config::GenerationConfig,
    ) -> impl futures::Stream<Item = Result<GeneratedToken>> + 'a {
        use boostr::autograd::Var;
        use boostr::inference::decode_graph::{DecodeGraph, DeviceScalars};
        use boostr::runtime::cuda::CudaRuntime as CudaRt;
        use boostr::CudaRuntime;
        use boostr::Runtime;

        stream! {
            let prompt_tokens = self.tokenizer().encode(prompt)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

            if prompt_tokens.is_empty() {
                return;
            }

            let max_seq_len = self.config().max_seq_len();
            // Graph mode pre-allocates KV cache at fixed capacity.
            // Use num_ctx (user-specified) capped to model max_seq_len.
            let graph_capacity = self.num_ctx().min(max_seq_len);
            let max_tokens = gen_config.max_tokens.min(
                graph_capacity.saturating_sub(prompt_tokens.len())
            );

            let input = self.create_input_tensor(&prompt_tokens)?;

            // ── KV cache at full capacity (required for stable device addresses) ──
            let num_layers = self.model().num_layers();
            let num_kv_heads = self.model().num_kv_heads().unwrap_or(8);
            let head_dim = self.model().head_dim().unwrap_or(64);
            let half_dim = head_dim / 2;
            let kv_dtype = parse_dtype(self.config().dtype())?;

            let mut kv_cache = boostr::inference::LayeredKvCache::new_positional(
                num_layers, 1, num_kv_heads, graph_capacity, graph_capacity, head_dim, kv_dtype, self.device(),
            ).map_err(|e| anyhow!("Failed to create full-capacity KV cache: {}", e))?;

            // ── Prefill ──
            let t0 = std::time::Instant::now();
            let prefill_logits = self.model().forward_with_kv_cache(&input, &mut kv_cache, 0)
                .map_err(|e| anyhow!("Prefill forward pass failed: {}", e))?;
            tracing::info!("Graph-mode prefill: {:?} (seq_len={})", t0.elapsed(), prompt_tokens.len());

            // ── Extract RoPE tables from model ──
            let (rope_cos_var, rope_sin_var) = self.model()
                .rope_caches()
                .ok_or_else(|| anyhow!("Model has no RoPE cache — cannot use CUDA graph mode"))?;
            let rope_cos_cache = rope_cos_var.tensor().clone();
            let rope_sin_cache = rope_sin_var.tensor().clone();

            let client = CudaRt::default_client(self.device());

            // ── Stable-address decode inputs (ALL allocated BEFORE graph capture) ──
            let token_buf = boostr::Tensor::<CudaRuntime>::zeros(&[1, 1], boostr::DType::I64, self.device());

            let cos_slice_t = boostr::Tensor::<CudaRuntime>::zeros(&[1, half_dim], boostr::DType::F32, self.device());
            let sin_slice_t = boostr::Tensor::<CudaRuntime>::zeros(&[1, half_dim], boostr::DType::F32, self.device());
            let cos_slice = Var::new(cos_slice_t, false);
            let sin_slice = Var::new(sin_slice_t, false);

            let device_scalars = DeviceScalars::new(kv_cache.seq_len(), self.device());

            let next_token_buf = boostr::Tensor::<CudaRuntime>::zeros(&[1], boostr::DType::I64, self.device());

            // ── Warmup pass: JIT all kernels, do NOT capture yet ──
            device_scalars.update(&client, kv_cache.seq_len())
                .map_err(|e| anyhow!("DeviceScalars warmup update failed: {}", e))?;
            let warmup_logits = self.model().forward_graph_mode(
                &token_buf, &mut kv_cache, &device_scalars, &cos_slice, &sin_slice,
            ).map_err(|e| anyhow!("Graph-mode warmup forward failed: {}", e))?;
            boostr::inference::decode_graph::argmax_to_buf(&client, &warmup_logits, &next_token_buf)
                .map_err(|e| anyhow!("Warmup argmax_to_buf failed: {}", e))?;

            // Re-prefill with a fresh KV cache so capture starts from the correct state
            let mut kv_cache = boostr::inference::LayeredKvCache::new_positional(
                num_layers, 1, num_kv_heads, graph_capacity, graph_capacity, head_dim, kv_dtype, self.device(),
            ).map_err(|e| anyhow!("Failed to re-create KV cache for capture: {}", e))?;
            let _ = self.model().forward_with_kv_cache(&input, &mut kv_cache, 0)
                .map_err(|e| anyhow!("Re-prefill for capture failed: {}", e))?;

            let capture_seq_len = kv_cache.seq_len();
            device_scalars.update(&client, capture_seq_len)
                .map_err(|e| anyhow!("DeviceScalars pre-capture update failed: {}", e))?;
            device_scalars.update_rope_slices(&client, &rope_cos_cache, &rope_sin_cache, &cos_slice, &sin_slice, capture_seq_len, half_dim)
                .map_err(|e| anyhow!("RoPE pre-capture update failed: {}", e))?;

            // ── CUDA graph capture ──
            let (graph, ()) = CudaRt::capture_graph(&client, |c| {
                let logits = self.model().forward_graph_mode(
                    &token_buf, &mut kv_cache, &device_scalars, &cos_slice, &sin_slice,
                ).map_err(|e| boostr::NumrError::Backend(format!("Capture forward failed: {e}")))?;
                boostr::inference::decode_graph::argmax_to_buf(c, &logits, &next_token_buf)
                    .map_err(|e| boostr::NumrError::Backend(format!("Capture argmax_to_buf failed: {e}")))
            }).map_err(|e| anyhow!("CUDA graph capture failed: {}", e))?;

            tracing::info!("CUDA graph captured (seq_len={})", kv_cache.seq_len());

            // Build DecodeGraph with all stable-address state
            let mut decode_graph = DecodeGraph {
                graph,
                device_scalars,
                token_buf,
                cos_slice: cos_slice.tensor().clone(),
                sin_slice: sin_slice.tensor().clone(),
                rope_cos_cache,
                rope_sin_cache,
                next_token_buf,
                head_dim: half_dim, // DecodeGraph stores half_dim in head_dim field
                seq_len: kv_cache.seq_len(),
            };

            // ── First token from prefill logits ──
            let first_token_gpu = self.argmax_on_gpu(&prefill_logits)?;
            let first_event = first_token_gpu.record_event()
                .map_err(|e| anyhow!("Record event failed: {}", e))?;
            let first_token = Self::read_token_id(&first_token_gpu, first_event)?;

            if self.tokenizer().is_eos(first_token) {
                yield Ok(GeneratedToken { token_id: first_token, text: String::new(), logprob: None, top_logprobs: None, finish_reason: Some(FinishReason::Eos) });
                return;
            }
            let text = self.tokenizer().decode(&[first_token]).unwrap_or_default();
            yield Ok(GeneratedToken { token_id: first_token, text, logprob: None, top_logprobs: None, finish_reason: None });

            // Seed next_token_buf with the first token
            decode_graph.seed_next_token(&client, first_token as i64)
                .map_err(|e| anyhow!("Failed to seed next_token_buf: {}", e))?;

            // ── Graph decode loop (event-based sync, no full device sync) ──
            for i in 0..max_tokens.saturating_sub(1) {
                let t1 = std::time::Instant::now();

                decode_graph.pre_replay_and_launch(&client)
                    .map_err(|e| anyhow!("Graph pre-replay+launch failed: {}", e))?;
                let fwd_time = t1.elapsed();

                let event = decode_graph.next_token_buf.record_event()
                    .map_err(|e| anyhow!("Record event failed: {}", e))?;

                let t2 = std::time::Instant::now();
                let next_token = Self::read_token_id(&decode_graph.next_token_buf, event)?;
                tracing::info!("Graph token {}: fwd={:?} sync={:?}", i + 1, fwd_time, t2.elapsed());

                if self.tokenizer().is_eos(next_token) {
                    yield Ok(GeneratedToken { token_id: next_token, text: String::new(), logprob: None, top_logprobs: None, finish_reason: Some(FinishReason::Eos) });
                    break;
                }

                let graph_max = max_tokens.saturating_sub(1);
                let is_last = i + 1 == graph_max;
                let text = self.tokenizer().decode(&[next_token]).unwrap_or_default();
                yield Ok(GeneratedToken { token_id: next_token, text, logprob: None, top_logprobs: None, finish_reason: if is_last { Some(FinishReason::Length) } else { None } });
            }
        }
    }
}
