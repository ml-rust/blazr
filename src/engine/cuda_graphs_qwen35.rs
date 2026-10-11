//! `qwen35` arm of [`Executor::generate_with_graphs`]: one captured decode
//! step over a full-capacity KV cache and in-place GDN state, replayed per
//! token through `Qwen35DecodeGraph` and kept in `qwen35_graph_session`
//! across requests.
//!
//! The first request at a capacity builds the session: eager prefill, an
//! uncaptured warmup pass of the graph-mode forward, a reset plus re-prefill
//! into the same caches, then capture. Later requests reset the caches in
//! place, prefill their prompt, and replay. An image prompt prefills its
//! [`Qwen35PromptPlan`] instead of its ids and starts the replay at the rope
//! position the plan leaves, behind the KV slot. The capture stops at the
//! logits copy; this arm samples every step greedily on the host and writes
//! the token back for the next replay. The arena it captures into is sized by
//! boostr's `first_arena_bytes` and tightened by `tight_arena_bytes`.

#![cfg(feature = "cuda")]

use anyhow::{anyhow, Result};
use async_stream::stream;

use boostr::inference::decode_graph::{
    arena_is_loose, copy_into_stable, first_arena_bytes, tight_arena_bytes, DeviceScalars,
    Qwen35DecodeGraph,
};
use boostr::inference::{LaneTable, LayeredGdnState, LayeredKvCache};
use boostr::model::qwen35::Qwen35PromptPlan;
use boostr::runtime::cuda::CudaRuntime as CudaRt;
use boostr::{CudaClient, CudaRuntime, Runtime, Tensor};

use super::executor::Executor;
use super::graph_session::Qwen35GraphSession;
use super::types::{FinishReason, GeneratedToken};
use crate::tokenizer::TokenTextStream;

/// The one lane a `qwen35` graph session decodes in.
const GRAPH_LANE: usize = 0;
/// The sequence id that lane carries; one request at a time per session.
const GRAPH_SEQ: u64 = 0;

/// Stable-address inputs every replay reads.
struct Qwen35GraphInputs {
    token_buf: Tensor<CudaRuntime>,
    device_scalars: DeviceScalars,
    lanes: LaneTable<CudaRuntime>,
}

impl Executor<CudaRuntime> {
    /// Reset both caches in place and eager-prefill at position 0: `plan`
    /// through the embeddings forward, else `input` through the token
    /// forward.
    fn qwen35_reset_and_prefill(
        &self,
        client: &CudaClient,
        input: &Tensor<CudaRuntime>,
        plan: Option<&Qwen35PromptPlan<CudaRuntime>>,
        kv_cache: &mut LayeredKvCache<CudaRuntime>,
        gdn_state: &mut LayeredGdnState<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        kv_cache.reset();
        gdn_state
            .reset(client)
            .map_err(|e| anyhow!("qwen35 GDN state reset failed: {}", e))?;
        self.qwen35_prefill(input, plan, kv_cache, gdn_state)
            .map_err(|e| anyhow!("qwen35 graph-mode prefill failed: {}", e))
    }

    /// Write the single lane's state: the KV slot `seq_len` and the rope
    /// position `rope_pos` of the next token. Equal for a text-only
    /// context; `rope_pos` lags after an image.
    ///
    /// The prefill wrote straight into lane 0 of the session caches, so the
    /// lane is claimed rather than admitted.
    fn qwen35_write_scalars(
        client: &CudaClient,
        device_scalars: &mut DeviceScalars,
        lanes: &mut LaneTable<CudaRuntime>,
        seq_len: usize,
        rope_pos: usize,
        capacity: usize,
    ) -> Result<()> {
        device_scalars
            .update_lanes(client, &[seq_len], capacity)
            .map_err(|e| anyhow!("DeviceScalars update_lanes failed: {}", e))?;
        lanes
            .claim(client, GRAPH_LANE, GRAPH_SEQ, seq_len, rope_pos)
            .map_err(|e| anyhow!("LaneTable claim failed: {}", e))?;
        Ok(())
    }

    /// Rope positions the loaded `qwen35` model's IMROPE table holds, read
    /// through the LLM backbone whether the model is bare or multimodal.
    pub(super) fn qwen35_rope_table_len(&self) -> Result<usize> {
        Ok(self.qwen35_backbone()?.rope_table_len())
    }

    /// Capture one decode step over `kv_cache` and `gdn_state` into an arena
    /// of `arena_bytes`.
    ///
    /// The capture ends at the logits copy into `logits_buf`. It picks no
    /// token: the decode loop samples each replay's logits on the host and
    /// writes the result back as the next input token.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_capture(
        &self,
        client: &CudaClient,
        inputs: &mut Qwen35GraphInputs,
        logits_buf: &Tensor<CudaRuntime>,
        kv_cache: &LayeredKvCache<CudaRuntime>,
        gdn_state: &LayeredGdnState<CudaRuntime>,
        rope_pos: usize,
        arena_bytes: usize,
        capacity: usize,
    ) -> Result<boostr::runtime::CapturedGraph<CudaRuntime>> {
        Self::qwen35_write_scalars(
            client,
            &mut inputs.device_scalars,
            &mut inputs.lanes,
            kv_cache.seq_len(),
            rope_pos,
            capacity,
        )?;
        let graph = CudaRt::capture_graph_into_with_arena(
            client,
            &[&inputs.token_buf],
            &[logits_buf],
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
                        &inputs.lanes,
                    )
                    .map_err(|e| {
                        boostr::NumrError::Backend(format!("Capture forward failed: {e}"))
                    })?;
                copy_into_stable(c, &logits, logits_buf)
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

    /// Reset the session caches, prefill `input` or `plan`, and set the
    /// decode position and scalars for the first replay. Returns the prefill
    /// logits.
    fn qwen35_session_prefill(
        &self,
        client: &CudaClient,
        session: &mut Qwen35GraphSession,
        input: &Tensor<CudaRuntime>,
        plan: Option<&Qwen35PromptPlan<CudaRuntime>>,
    ) -> Result<Tensor<CudaRuntime>> {
        let logits = self.qwen35_reset_and_prefill(
            client,
            input,
            plan,
            &mut session.kv_cache,
            &mut session.gdn_state,
        )?;
        let seq_len = session.kv_cache.seq_len();
        let rope_pos = Self::qwen35_first_rope_pos(plan, &session.kv_cache);
        Self::qwen35_write_scalars(
            client,
            &mut session.decode.device_scalars,
            &mut session.decode.lanes,
            seq_len,
            rope_pos,
            session.capacity,
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
        plan: Option<&Qwen35PromptPlan<CudaRuntime>>,
        capacity: usize,
    ) -> Result<(Qwen35GraphSession, Tensor<CudaRuntime>)> {
        let (mut kv_cache, mut gdn_state) = self.qwen35_caches(capacity, capacity)?;

        let t0 = std::time::Instant::now();
        let warm_logits = self
            .qwen35_prefill(input, plan, &mut kv_cache, &mut gdn_state)
            .map_err(|e| anyhow!("qwen35 graph-mode prefill failed: {}", e))?;
        tracing::info!(
            "qwen35 graph-mode prefill: {:?} (seq_len={})",
            t0.elapsed(),
            kv_cache.seq_len()
        );
        drop(warm_logits);
        let capture_seq_len = kv_cache.seq_len();
        let capture_rope_pos = Self::qwen35_first_rope_pos(plan, &kv_cache);

        // Single-sequence lane table: one lane, bounded by the model's own
        // rope table.
        let rope_table_len = self.qwen35_rope_table_len()?;

        // Stable-address decode inputs, all allocated before capture.
        let mut inputs = Qwen35GraphInputs {
            token_buf: Tensor::<CudaRuntime>::zeros(&[1, 1], boostr::DType::I64, self.device())?,
            device_scalars: DeviceScalars::new(&[capture_seq_len], self.device())?,
            lanes: LaneTable::<CudaRuntime>::new(1, rope_table_len, self.device())
                .map_err(|e| anyhow!("LaneTable::new failed: {}", e))?,
        };

        // Warmup pass: JIT kernels and probe the tuner, not captured.
        Self::qwen35_write_scalars(
            client,
            &mut inputs.device_scalars,
            &mut inputs.lanes,
            capture_seq_len,
            capture_rope_pos,
            capacity,
        )?;
        let warmup_logits = self
            .model()
            .forward_qwen35_graph_mode(
                client,
                &inputs.token_buf,
                &kv_cache,
                &gdn_state,
                &inputs.device_scalars,
                &inputs.lanes,
            )
            .map_err(|e| anyhow!("qwen35 graph-mode warmup forward failed: {}", e))?;

        // Stable logits destination, shaped and typed by the warmup's own
        // output and allocated before capture so its address holds.
        let logits_buf = Tensor::<CudaRuntime>::zeros(
            warmup_logits.shape(),
            warmup_logits.dtype(),
            self.device(),
        )?;
        copy_into_stable(client, &warmup_logits, &logits_buf)
            .map_err(|e| anyhow!("Warmup logits copy failed: {}", e))?;
        drop(warmup_logits);

        // The warmup wrote the KV cache and GDN state in place: reset both
        // and prefill again so capture starts from the prompt state at the
        // same addresses.
        let prefill_logits =
            self.qwen35_reset_and_prefill(client, input, plan, &mut kv_cache, &mut gdn_state)?;

        // First capture with a generous arena.
        let first_arena = first_arena_bytes(self.device())?;
        let mut graph = self.qwen35_capture(
            client,
            &mut inputs,
            &logits_buf,
            &kv_cache,
            &gdn_state,
            capture_rope_pos,
            first_arena,
            capacity,
        )?;
        let used = graph.arena_bytes_used().unwrap_or(first_arena);
        tracing::info!(
            "qwen35 CUDA graph captured (arena={} used={})",
            first_arena,
            used
        );

        // Tighten: recapture when the arena is more than twice the footprint.
        // Capture records without executing, so the caches still hold the
        // prefill state and serve the second capture unchanged.
        if arena_is_loose(used, first_arena) {
            let tight_arena = tight_arena_bytes(used);
            drop(graph);
            graph = self.qwen35_capture(
                client,
                &mut inputs,
                &logits_buf,
                &kv_cache,
                &gdn_state,
                capture_rope_pos,
                tight_arena,
                capacity,
            )?;
            tracing::info!(
                "qwen35 CUDA graph recaptured (arena={} used={:?})",
                tight_arena,
                graph.arena_bytes_used()
            );
        }

        let Qwen35GraphInputs {
            token_buf,
            device_scalars,
            lanes,
        } = inputs;
        let decode = Qwen35DecodeGraph::new(
            graph,
            device_scalars,
            capacity,
            lanes,
            token_buf,
            logits_buf,
        )
        .map_err(|e| anyhow!("qwen35 decode graph binding failed: {}", e))?;
        let session = Qwen35GraphSession {
            decode,
            kv_cache,
            gdn_state,
            capacity,
        };
        Ok((session, prefill_logits))
    }

    /// Greedy graph-mode decode for a `qwen35` model. `images` holds the
    /// encoded bytes of one image per `<|image_pad|>` in `prompt_tokens`;
    /// empty for a text prompt.
    ///
    /// The session lock is held for the whole stream, so graph-mode requests
    /// on one executor run one at a time.
    ///
    /// # Preconditions
    /// - `self.model().supports_gdn_graph_mode()` is true
    /// - `gen_config.is_greedy()` is true and `gen_config.has_penalties()` is false: this loop samples each step greedily
    pub(super) fn generate_with_graphs_qwen35<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        images: &'a [Vec<u8>],
        gen_config: &'a crate::config::GenerationConfig,
    ) -> impl futures::Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            let max_seq_len = self.config().max_seq_len();
            let graph_capacity = self.num_ctx().min(max_seq_len);
            let plan = if images.is_empty() {
                None
            } else {
                Some(self.qwen35_image_plan(prompt_tokens, images)?)
            };
            let prompt_rows = plan.as_ref().map_or(prompt_tokens.len(), Qwen35PromptPlan::seq_len);
            if prompt_rows > graph_capacity {
                yield Err(anyhow!(
                    "prompt of {prompt_rows} rows exceeds the graph capacity of {graph_capacity}"
                ));
                return;
            }
            let max_tokens = gen_config.max_tokens.min(
                graph_capacity.saturating_sub(prompt_rows)
            );

            let input = self.create_input_tensor(prompt_tokens)?;
            let client = CudaRt::default_client(self.device());

            let mut guard = self.qwen35_graph_session.lock().await;
            let reuse = guard.as_ref().is_some_and(|s| s.capacity == graph_capacity);
            let (session, prefill_logits) = if reuse {
                let session = guard.as_mut()
                    .ok_or_else(|| anyhow!("qwen35 graph session missing after reuse check"))?;
                let logits = self.qwen35_session_prefill(&client, session, &input, plan.as_ref())?;
                (session, logits)
            } else {
                *guard = None;
                let (session, logits) =
                    self.qwen35_build_session(&client, &input, plan.as_ref(), graph_capacity)?;
                (guard.insert(session), logits)
            };
            drop(plan);
            let decode_graph = &mut session.decode;

            let mut text = TokenTextStream::new(self.tokenizer(), prompt_tokens);
            // ── First token from prefill logits ──
            let first_token = self.first_token_from_logits(&prefill_logits)?;
            drop(prefill_logits);

            if self.tokenizer().is_eos(first_token) {
                yield Ok(Self::eos_token(first_token, &mut text));
                return;
            }
            yield Ok(self.text_token(&mut text, first_token, None));

            decode_graph.set_next_tokens(&client, &[first_token as i64])
                .map_err(|e| anyhow!("Failed to seed the graph lane's first token: {}", e))?;

            // ── Graph decode loop ──
            // Replay, then sample the lane's logits on the host through the
            // same greedy path the eager arm uses, then write the chosen
            // token back as the next replay's input. One event-gated readback
            // per step, no device sync.
            let graph_max = max_tokens.saturating_sub(1);
            for i in 0..graph_max {
                let t1 = std::time::Instant::now();

                decode_graph.pre_replay_and_launch(&client)
                    .map_err(|e| anyhow!("qwen35 graph pre-replay+launch failed: {}", e))?;
                let fwd_time = t1.elapsed();

                let t2 = std::time::Instant::now();
                let next_token = self.first_token_from_logits(decode_graph.logits())?;
                decode_graph.set_next_tokens(&client, &[next_token as i64])
                    .map_err(|e| anyhow!("Failed to write the graph lane's next token: {}", e))?;
                tracing::info!("qwen35 graph token {}: fwd={:?} sync={:?}", i + 1, fwd_time, t2.elapsed());

                if self.tokenizer().is_eos(next_token) {
                    yield Ok(Self::eos_token(next_token, &mut text));
                    break;
                }

                let is_last = i + 1 == graph_max;
                let finish_reason = if is_last { Some(FinishReason::Length) } else { None };
                yield Ok(self.text_token(&mut text, next_token, finish_reason));
            }
        }
    }
}
