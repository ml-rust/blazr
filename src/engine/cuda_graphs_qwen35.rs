//! `qwen35` arm of [`Executor::generate_with_graphs`]: one captured decode
//! step over a full-capacity KV cache and in-place GDN state, replayed per
//! token through `Qwen35DecodeGraph` and kept in `qwen35_graph_session`
//! across requests.
//!
//! The first request at a capacity builds the session: eager prefill, an
//! uncaptured warmup pass of the graph-mode forward, a reset plus re-prefill
//! into the same caches, then capture. Later requests reset the caches in
//! place, prefill their prompt, and replay. One `qwen35` decode step records
//! on the order of a thousand intermediate allocations, so capture goes
//! through the arena variant; the arena is sized generously first and
//! tightened from the reported footprint when the first guess is more than
//! twice too large.

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
use super::graph_session::Qwen35GraphSession;
use super::types::{FinishReason, GeneratedToken};

/// Upper bound for the first capture arena.
const ARENA_CAP_BYTES: usize = 1 << 30;
/// The first arena is at most this fraction of free device memory.
const ARENA_FREE_DIVISOR: u64 = 4;
/// Recapture granularity for the tightened arena.
const ARENA_ROUND_BYTES: usize = 1 << 20;

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

    /// Reset both caches in place and eager-prefill `input` at position 0.
    fn qwen35_reset_and_prefill(
        &self,
        client: &CudaClient,
        input: &Tensor<CudaRuntime>,
        kv_cache: &mut LayeredKvCache<CudaRuntime>,
        gdn_state: &mut LayeredGdnState<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        kv_cache.reset();
        gdn_state
            .reset(client)
            .map_err(|e| anyhow!("qwen35 GDN state reset failed: {}", e))?;
        self.model()
            .forward_qwen35(input, kv_cache, gdn_state, 0)
            .map_err(|e| anyhow!("qwen35 graph-mode prefill failed: {}", e))
    }

    /// Write both scalar buffers for `seq_len`: a text-only context, so the
    /// rope position equals the KV slot.
    fn qwen35_write_scalars(
        client: &CudaClient,
        device_scalars: &DeviceScalars,
        mrope: &MropeScalars,
        seq_len: usize,
    ) -> Result<()> {
        device_scalars
            .update(client, seq_len)
            .map_err(|e| anyhow!("DeviceScalars update failed: {}", e))?;
        mrope
            .update(client, seq_len)
            .map_err(|e| anyhow!("MropeScalars update failed: {}", e))?;
        Ok(())
    }

    /// Capture one decode step over `kv_cache` and `gdn_state` into an arena
    /// of `arena_bytes`.
    fn qwen35_capture(
        &self,
        client: &CudaClient,
        inputs: &Qwen35GraphInputs,
        kv_cache: &LayeredKvCache<CudaRuntime>,
        gdn_state: &LayeredGdnState<CudaRuntime>,
        arena_bytes: usize,
    ) -> Result<boostr::runtime::CapturedGraph<CudaRuntime>> {
        Self::qwen35_write_scalars(
            client,
            &inputs.device_scalars,
            &inputs.mrope,
            kv_cache.seq_len(),
        )?;
        let graph = CudaRt::capture_graph_into_with_arena(
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
                        kv_cache,
                        gdn_state,
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
        })?;
        self.note_graph_capture();
        Ok(graph)
    }

    /// Reset the session caches, prefill `input`, and set the decode position
    /// and scalars for the first replay. Returns the prefill logits.
    fn qwen35_session_prefill(
        &self,
        client: &CudaClient,
        session: &mut Qwen35GraphSession,
        input: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let logits = self.qwen35_reset_and_prefill(
            client,
            input,
            &mut session.kv_cache,
            &mut session.gdn_state,
        )?;
        let seq_len = session.kv_cache.seq_len();
        session.decode.seq_len = seq_len;
        // Text-only prompt: the rope position is the KV slot.
        session.decode.rope_pos = seq_len;
        Self::qwen35_write_scalars(
            client,
            &session.decode.device_scalars,
            &session.decode.mrope,
            seq_len,
        )?;
        Ok(logits)
    }

    /// Build a `qwen35` session for `capacity`: full-capacity caches, eager
    /// prefill, stable buffers, an uncaptured warmup pass, reset plus
    /// re-prefill into the same caches, then capture with arena tightening.
    ///
    /// Returns the session and the prefill logits of the re-prefill.
    fn qwen35_build_session(
        &self,
        client: &CudaClient,
        input: &Tensor<CudaRuntime>,
        capacity: usize,
    ) -> Result<(Qwen35GraphSession, Tensor<CudaRuntime>)> {
        let (mut kv_cache, mut gdn_state) = self.qwen35_caches(capacity, capacity)?;

        let t0 = std::time::Instant::now();
        let warm_logits = self
            .model()
            .forward_qwen35(input, &mut kv_cache, &mut gdn_state, 0)
            .map_err(|e| anyhow!("qwen35 graph-mode prefill failed: {}", e))?;
        tracing::info!(
            "qwen35 graph-mode prefill: {:?} (seq_len={})",
            t0.elapsed(),
            kv_cache.seq_len()
        );
        drop(warm_logits);
        let capture_seq_len = kv_cache.seq_len();

        // Stable-address decode inputs, all allocated before capture.
        let inputs = Qwen35GraphInputs {
            token_buf: Tensor::<CudaRuntime>::zeros(&[1, 1], boostr::DType::I64, self.device())?,
            next_token_buf: Tensor::<CudaRuntime>::zeros(&[1], boostr::DType::I64, self.device())?,
            device_scalars: DeviceScalars::new(capture_seq_len, self.device())?,
            mrope: MropeScalars::new(capture_seq_len, self.device())?,
        };

        // Warmup pass: JIT kernels and probe the tuner, not captured.
        Self::qwen35_write_scalars(
            client,
            &inputs.device_scalars,
            &inputs.mrope,
            capture_seq_len,
        )?;
        let warmup_logits = self
            .model()
            .forward_qwen35_graph_mode(
                client,
                &inputs.token_buf,
                &kv_cache,
                &gdn_state,
                &inputs.device_scalars,
                &inputs.mrope,
            )
            .map_err(|e| anyhow!("qwen35 graph-mode warmup forward failed: {}", e))?;
        argmax_to_buf(client, &warmup_logits, &inputs.next_token_buf)
            .map_err(|e| anyhow!("Warmup argmax_to_buf failed: {}", e))?;
        drop(warmup_logits);

        // The warmup wrote the KV cache and GDN state in place: reset both
        // and prefill again so capture starts from the prompt state at the
        // same addresses.
        let prefill_logits =
            self.qwen35_reset_and_prefill(client, input, &mut kv_cache, &mut gdn_state)?;

        // First capture with a generous arena.
        let first_arena = self.qwen35_first_arena_bytes()?;
        let mut graph = self.qwen35_capture(client, &inputs, &kv_cache, &gdn_state, first_arena)?;
        let used = graph.arena_bytes_used().unwrap_or(first_arena);
        tracing::info!(
            "qwen35 CUDA graph captured (arena={} used={})",
            first_arena,
            used
        );

        // Tighten: recapture when the arena is more than twice the footprint.
        // Capture records without executing, so the caches still hold the
        // prefill state and serve the second capture unchanged.
        if used < first_arena / 2 {
            let tight_arena = Self::qwen35_tight_arena_bytes(used);
            drop(graph);
            graph = self.qwen35_capture(client, &inputs, &kv_cache, &gdn_state, tight_arena)?;
            tracing::info!(
                "qwen35 CUDA graph recaptured (arena={} used={:?})",
                tight_arena,
                graph.arena_bytes_used()
            );
        }

        let Qwen35GraphInputs {
            token_buf,
            next_token_buf,
            device_scalars,
            mrope,
        } = inputs;
        let decode = Qwen35DecodeGraph {
            graph,
            device_scalars,
            mrope,
            token_buf,
            next_token_buf,
            seq_len: kv_cache.seq_len(),
            rope_pos: kv_cache.seq_len(),
        };
        let session = Qwen35GraphSession {
            decode,
            kv_cache,
            gdn_state,
            capacity,
        };
        Ok((session, prefill_logits))
    }

    /// Greedy graph-mode decode for a `qwen35` model.
    ///
    /// The session lock is held for the whole stream, so graph-mode requests
    /// on one executor run one at a time.
    ///
    /// # Preconditions
    /// - `self.model().supports_gdn_graph_mode()` is true
    /// - `gen_config.is_greedy()` is true and `gen_config.has_penalties()` is false: the replay reads only the argmax token
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

            let mut guard = self.qwen35_graph_session.lock().await;
            let reuse = guard.as_ref().is_some_and(|s| s.capacity == graph_capacity);
            let (session, prefill_logits) = if reuse {
                let session = guard.as_mut()
                    .ok_or_else(|| anyhow!("qwen35 graph session missing after reuse check"))?;
                let logits = self.qwen35_session_prefill(&client, session, &input)?;
                (session, logits)
            } else {
                *guard = None;
                let (session, logits) = self.qwen35_build_session(&client, &input, graph_capacity)?;
                (guard.insert(session), logits)
            };
            let decode_graph = &mut session.decode;

            // ── First token from prefill logits ──
            let first_token = self.first_token_from_logits(&prefill_logits)?;
            drop(prefill_logits);

            if self.tokenizer().is_eos(first_token) {
                yield Ok(Self::eos_token(first_token));
                return;
            }
            yield Ok(self.text_token(first_token, None));

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
                    yield Ok(Self::eos_token(next_token));
                    break;
                }

                let is_last = i + 1 == graph_max;
                let finish_reason = if is_last { Some(FinishReason::Length) } else { None };
                yield Ok(self.text_token(next_token, finish_reason));
            }
        }
    }
}
