//! `qwen35` arm of [`Executor::generate_with_graphs`]: one captured decode
//! step over a full-capacity KV cache and in-place GDN state, replayed per
//! token through `Qwen35DecodeGraph`.
//!
//! The flow mirrors the Llama arm in `cuda_graphs.rs`: eager prefill, an
//! uncaptured warmup pass of the graph-mode forward, a re-prefill on fresh
//! caches, then capture. One `qwen35` decode step records on the order of a
//! thousand intermediate allocations, so capture goes through the arena
//! variant; the arena is sized generously first and tightened from the
//! reported footprint when the first guess is more than twice too large.

#![cfg(feature = "cuda")]

use anyhow::{anyhow, Result};
use async_stream::stream;

use boostr::inference::decode_graph::{
    argmax_to_buf, DeviceScalars, MropeScalars, Qwen35DecodeGraph,
};
use boostr::inference::{LayeredGdnState, LayeredKvCache};
use boostr::runtime::cuda::CudaRuntime as CudaRt;
use boostr::{CudaClient, CudaRuntime, Runtime, Tensor};

use super::executor::Executor;
use super::types::{FinishReason, GeneratedToken};

/// Upper bound for the first capture arena.
const ARENA_CAP_BYTES: usize = 1 << 30;
/// The first arena is at most this fraction of free device memory.
const ARENA_FREE_DIVISOR: u64 = 4;
/// Recapture granularity for the tightened arena.
const ARENA_ROUND_BYTES: usize = 1 << 20;

/// Fresh caches and the prefill logits for one capture attempt.
struct Qwen35Prefill {
    kv_cache: LayeredKvCache<CudaRuntime>,
    gdn_state: LayeredGdnState<CudaRuntime>,
    logits: Tensor<CudaRuntime>,
}

/// Stable-address buffers every replay reads or writes.
struct Qwen35GraphInputs {
    token_buf: Tensor<CudaRuntime>,
    next_token_buf: Tensor<CudaRuntime>,
    device_scalars: DeviceScalars,
    mrope: MropeScalars,
}

impl Executor<CudaRuntime> {
    /// Arena for the first capture: `min(free / 4, 1 GiB)`.
    fn qwen35_first_arena_bytes(&self) -> Result<usize> {
        let (free, _total) = self
            .device()
            .memory_info()
            .map_err(|e| anyhow!("Device memory query failed: {}", e))?;
        let quarter = usize::try_from(free / ARENA_FREE_DIVISOR).unwrap_or(ARENA_CAP_BYTES);
        Ok(quarter.clamp(ARENA_ROUND_BYTES, ARENA_CAP_BYTES))
    }

    /// Tightened arena from a measured footprint: `used * 5 / 4`, rounded up
    /// to 1 MiB.
    fn qwen35_tight_arena_bytes(used: usize) -> usize {
        let padded = (used * 5).div_ceil(4);
        padded.div_ceil(ARENA_ROUND_BYTES).max(1) * ARENA_ROUND_BYTES
    }

    /// Fresh full-capacity caches plus an eager prefill of `input`.
    fn qwen35_graph_prefill(
        &self,
        input: &Tensor<CudaRuntime>,
        graph_capacity: usize,
    ) -> Result<Qwen35Prefill> {
        let (mut kv_cache, mut gdn_state) = self.qwen35_caches(graph_capacity, graph_capacity)?;
        let logits = self
            .model()
            .forward_qwen35(input, &mut kv_cache, &mut gdn_state, 0)
            .map_err(|e| anyhow!("qwen35 graph-mode prefill failed: {}", e))?;
        Ok(Qwen35Prefill {
            kv_cache,
            gdn_state,
            logits,
        })
    }

    /// Write both scalar buffers for `seq_len`.
    fn qwen35_write_scalars(
        client: &CudaClient,
        inputs: &Qwen35GraphInputs,
        seq_len: usize,
    ) -> Result<()> {
        inputs
            .device_scalars
            .update(client, seq_len)
            .map_err(|e| anyhow!("DeviceScalars update failed: {}", e))?;
        inputs
            .mrope
            .update(client, seq_len)
            .map_err(|e| anyhow!("MropeScalars update failed: {}", e))?;
        Ok(())
    }

    /// Capture one decode step into an arena of `arena_bytes`.
    fn qwen35_capture(
        &self,
        client: &CudaClient,
        inputs: &Qwen35GraphInputs,
        prefill: &Qwen35Prefill,
        arena_bytes: usize,
    ) -> Result<boostr::runtime::CapturedGraph<CudaRuntime>> {
        Self::qwen35_write_scalars(client, inputs, prefill.kv_cache.seq_len())?;
        CudaRt::capture_graph_into_with_arena(
            client,
            &[&inputs.token_buf],
            &[&inputs.next_token_buf],
            arena_bytes,
            |c| {
                let logits = self
                    .model()
                    .forward_qwen35_graph_mode(
                        c,
                        &inputs.token_buf,
                        &prefill.kv_cache,
                        &prefill.gdn_state,
                        &inputs.device_scalars,
                        &inputs.mrope,
                    )
                    .map_err(|e| {
                        boostr::NumrError::Backend(format!("Capture forward failed: {e}"))
                    })?;
                argmax_to_buf(c, &logits, &inputs.next_token_buf)
            },
        )
        .map_err(|e| {
            anyhow!(
                "CUDA graph capture (arena {} bytes) failed: {}",
                arena_bytes,
                e
            )
        })
    }

    /// Greedy graph-mode decode for a `qwen35` model.
    ///
    /// # Preconditions
    /// - `self.model().supports_gdn_graph_mode()` is true
    /// - `gen_config.is_greedy()` is true
    pub(super) fn generate_with_graphs_qwen35<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        gen_config: &'a crate::config::GenerationConfig,
    ) -> impl futures::Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            let max_seq_len = self.config().max_seq_len();
            let graph_capacity = self.num_ctx().min(max_seq_len);
            let max_tokens = gen_config.max_tokens.min(
                graph_capacity.saturating_sub(prompt_tokens.len())
            );

            let input = self.create_input_tensor(prompt_tokens)?;
            let client = CudaRt::default_client(self.device());

            // ── Prefill ──
            let t0 = std::time::Instant::now();
            let warm = self.qwen35_graph_prefill(&input, graph_capacity)?;
            tracing::info!("qwen35 graph-mode prefill: {:?} (seq_len={})", t0.elapsed(), prompt_tokens.len());
            let capture_seq_len = warm.kv_cache.seq_len();

            // ── Stable-address decode inputs, all allocated before capture ──
            let inputs = Qwen35GraphInputs {
                token_buf: Tensor::<CudaRuntime>::zeros(&[1, 1], boostr::DType::I64, self.device())?,
                next_token_buf: Tensor::<CudaRuntime>::zeros(&[1], boostr::DType::I64, self.device())?,
                device_scalars: DeviceScalars::new(capture_seq_len, self.device())?,
                mrope: MropeScalars::new(capture_seq_len, self.device())?,
            };

            // ── Warmup pass: JIT kernels and probe the tuner, not captured ──
            Self::qwen35_write_scalars(&client, &inputs, capture_seq_len)?;
            let warmup_logits = self.model().forward_qwen35_graph_mode(
                &client, &inputs.token_buf, &warm.kv_cache, &warm.gdn_state,
                &inputs.device_scalars, &inputs.mrope,
            ).map_err(|e| anyhow!("qwen35 graph-mode warmup forward failed: {}", e))?;
            argmax_to_buf(&client, &warmup_logits, &inputs.next_token_buf)
                .map_err(|e| anyhow!("Warmup argmax_to_buf failed: {}", e))?;
            drop(warmup_logits);
            // The warmup pass wrote the GDN state in place: free those caches
            // and prefill fresh ones for capture.
            drop(warm);

            // ── First capture with a generous arena ──
            let prefill = self.qwen35_graph_prefill(&input, graph_capacity)?;
            let first_arena = self.qwen35_first_arena_bytes()?;
            let mut graph = self.qwen35_capture(&client, &inputs, &prefill, first_arena)?;
            let used = graph.arena_bytes_used().unwrap_or(first_arena);
            tracing::info!("qwen35 CUDA graph captured (arena={} used={})", first_arena, used);

            // ── Tighten: recapture when the arena is more than twice the footprint ──
            // Capture records without executing, so the caches still hold the
            // prefill state and the same `prefill` serves the second capture.
            if used < first_arena / 2 {
                let tight_arena = Self::qwen35_tight_arena_bytes(used);
                drop(graph);
                graph = self.qwen35_capture(&client, &inputs, &prefill, tight_arena)?;
                tracing::info!(
                    "qwen35 CUDA graph recaptured (arena={} used={:?})",
                    tight_arena, graph.arena_bytes_used()
                );
            }

            // `prefill` owns the caches the graph writes; it is declared before
            // `decode_graph`, so the graph drops first.
            let Qwen35GraphInputs { token_buf, next_token_buf, device_scalars, mrope } = inputs;
            let mut decode_graph = Qwen35DecodeGraph {
                graph,
                device_scalars,
                mrope,
                token_buf,
                next_token_buf,
                seq_len: prefill.kv_cache.seq_len(),
            };

            // ── First token from prefill logits ──
            let first_token_gpu = self.argmax_on_gpu(&prefill.logits)?;
            let first_event = first_token_gpu.record_event()
                .map_err(|e| anyhow!("Record event failed: {}", e))?;
            let first_token = Self::read_token_id(&first_token_gpu, first_event)?;

            if self.tokenizer().is_eos(first_token) {
                yield Ok(GeneratedToken { token_id: first_token, text: String::new(), logprob: None, top_logprobs: None, finish_reason: Some(FinishReason::Eos) });
                return;
            }
            let text = self.tokenizer().decode(&[first_token]).unwrap_or_default();
            yield Ok(GeneratedToken { token_id: first_token, text, logprob: None, top_logprobs: None, finish_reason: None });

            decode_graph.seed_next_token(&client, first_token as i64)
                .map_err(|e| anyhow!("Failed to seed next_token_buf: {}", e))?;

            // ── Graph decode loop (event-gated readback, no device sync) ──
            let graph_max = max_tokens.saturating_sub(1);
            for i in 0..graph_max {
                let t1 = std::time::Instant::now();

                decode_graph.pre_replay_and_launch(&client)
                    .map_err(|e| anyhow!("qwen35 graph pre-replay+launch failed: {}", e))?;
                let fwd_time = t1.elapsed();

                let event = decode_graph.next_token_buf.record_event()
                    .map_err(|e| anyhow!("Record event failed: {}", e))?;

                let t2 = std::time::Instant::now();
                let next_token = Self::read_token_id(&decode_graph.next_token_buf, event)?;
                tracing::info!("qwen35 graph token {}: fwd={:?} sync={:?}", i + 1, fwd_time, t2.elapsed());

                if self.tokenizer().is_eos(next_token) {
                    yield Ok(GeneratedToken { token_id: next_token, text: String::new(), logprob: None, top_logprobs: None, finish_reason: Some(FinishReason::Eos) });
                    break;
                }

                let is_last = i + 1 == graph_max;
                let text = self.tokenizer().decode(&[next_token]).unwrap_or_default();
                yield Ok(GeneratedToken { token_id: next_token, text, logprob: None, top_logprobs: None, finish_reason: if is_last { Some(FinishReason::Length) } else { None } });
            }
        }
    }
}
