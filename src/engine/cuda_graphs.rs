//! CUDA graph-accelerated greedy decode loop.
//!
//! The config flag `inference.graphs` is backend-agnostic. The graph-mode
//! forward passes use CUDA-specific kernel APIs (device-ptr seq_len,
//! kv_insert kernel), so this impl block is CUDA-only. Another runtime with
//! graph support gets a parallel impl block over its own graph-mode ops.

#![cfg(feature = "cuda")]

use anyhow::{anyhow, Result};
use async_stream::stream;

use boostr::autograd::Var;
use boostr::inference::decode_graph::{DeviceScalars, PagedDecodeGraph};
use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::{BlockTable, CpuBlockAllocator};
use boostr::tensor::Tensor;
use boostr::{CudaRuntime, Runtime};

use crate::config::parse_dtype;

use super::executor::Executor;
use super::types::{FinishReason, GeneratedToken};

impl Executor<CudaRuntime> {
    /// Run greedy decode using compute graph capture+replay (CUDA backend).
    ///
    /// The first request at a given capacity captures the single-token decode
    /// step into `llama_graph_session`. Later requests reset the session's
    /// KV cache in place, prefill their prompt eagerly, and replay the same
    /// graph. Replay cost is one graph launch per token instead of
    /// per-kernel dispatch. The session lock is held for the whole stream.
    ///
    /// A `qwen35` model takes the arm in `cuda_graphs_qwen35.rs`, which
    /// captures `forward_qwen35_graph_mode` with an arena.
    ///
    /// # Preconditions
    /// - Model must be Llama or `qwen35` (Mamba2/Hybrid graph mode not yet implemented)
    /// - `gen_config.is_greedy()` must be true and `gen_config.has_penalties()` false: the replay reads only the argmax token
    /// - `config.inference.paged_attention` must be false (use `generate_with_graphs_paged` for paged mode)
    pub fn generate_with_graphs<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        gen_config: &'a crate::config::GenerationConfig,
    ) -> impl futures::Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            if prompt_tokens.is_empty() {
                return;
            }
            if self.model().needs_gdn_state() {
                if !self.model().supports_gdn_graph_mode() {
                    yield Err(super::qwen35_unsupported("CUDA graph", None, Some("run without --graphs")));
                    return;
                }
                let inner = self.generate_with_graphs_qwen35(prompt_tokens, &[], gen_config);
                for await item in inner {
                    yield item;
                }
                return;
            }

            let max_seq_len = self.config().max_seq_len();
            // Graph mode pre-allocates the KV cache at fixed capacity:
            // num_ctx capped to the model's max_seq_len.
            let graph_capacity = self.num_ctx().min(max_seq_len);
            let max_tokens = gen_config.max_tokens.min(
                graph_capacity.saturating_sub(prompt_tokens.len())
            );

            let input = self.create_input_tensor(prompt_tokens)?;
            let client = CudaRuntime::default_client(self.device());

            // One captured graph per executor: the guard is held for the
            // whole request, so graph-mode requests run one at a time.
            let mut guard = self.llama_graph_session.lock().await;
            let reuse = guard.as_ref().is_some_and(|s| s.capacity == graph_capacity);
            let (session, prefill_logits) = if reuse {
                let session = guard.as_mut()
                    .ok_or_else(|| anyhow!("Llama graph session missing after reuse check"))?;
                let logits = self.llama_session_prefill(session, &input)?;
                (session, logits)
            } else {
                *guard = None;
                let (session, logits) = self.llama_build_session(&input, graph_capacity)?;
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

            // Seed next_token_buf with the first token
            decode_graph.seed_next_token(&client, first_token as i64)
                .map_err(|e| anyhow!("Failed to seed next_token_buf: {}", e))?;

            // ── Graph decode loop (event-based sync, no full device sync) ──
            let graph_max = max_tokens.saturating_sub(1);
            for i in 0..graph_max {
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
                    yield Ok(Self::eos_token(next_token));
                    break;
                }

                let is_last = i + 1 == graph_max;
                let finish_reason = if is_last { Some(FinishReason::Length) } else { None };
                yield Ok(self.text_token(next_token, finish_reason));
            }
        }
    }

    /// Graph-mode form of [`Executor::generate_with_images`]: the prompt
    /// plan prefills the session caches, then the captured step replays with
    /// the rope position the plan left. Empty `images` is
    /// [`Executor::generate_with_graphs`].
    ///
    /// # Preconditions
    /// - Model must be `qwen35` with graph mode support
    /// - `gen_config.is_greedy()` must be true and `gen_config.has_penalties()` false
    pub fn generate_with_graphs_images<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        images: &'a [Vec<u8>],
        gen_config: &'a crate::config::GenerationConfig,
    ) -> impl futures::Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            if images.is_empty() {
                let inner = self.generate_with_graphs(prompt_tokens, gen_config);
                for await item in inner {
                    yield item;
                }
                return;
            }
            if prompt_tokens.is_empty() {
                yield Err(anyhow!("image prompt has no ids: the markers alone are missing"));
                return;
            }
            if !self.model().needs_gdn_state() {
                yield Err(anyhow!(
                    "image input needs a qwen35 model; the loaded model is {}",
                    self.model().model_type()
                ));
                return;
            }
            if !self.model().supports_gdn_graph_mode() {
                yield Err(super::qwen35_unsupported("CUDA graph", None, Some("run without --graphs")));
                return;
            }
            let inner = self.generate_with_graphs_qwen35(prompt_tokens, images, gen_config);
            for await item in inner {
                yield item;
            }
        }
    }

    /// Run greedy decode using CUDA graph capture+replay with **paged KV cache**.
    ///
    /// Same idea as `generate_with_graphs` but uses `forward_graph_paged` and
    /// `PagedDecodeGraph`, which read `seq_len_k` and write `slot_mapping`
    /// from device memory so these values can change between graph replays.
    ///
    /// # Preconditions
    /// - Model must be Llama
    /// - `gen_config.is_greedy()` must be true and `gen_config.has_penalties()` false
    /// - `config.inference.paged_attention` must be true
    pub fn generate_with_graphs_paged<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        gen_config: &'a crate::config::GenerationConfig,
    ) -> impl futures::Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            if prompt_tokens.is_empty() {
                return;
            }
            if self.model().needs_gdn_state() {
                yield Err(super::qwen35_unsupported("CUDA graph", None, Some("run without --graphs")));
                return;
            }

            let max_seq_len = self.config().max_seq_len();
            let graph_capacity = self.num_ctx().min(max_seq_len);
            let max_tokens = gen_config.max_tokens.min(
                graph_capacity.saturating_sub(prompt_tokens.len())
            );

            let input = self.create_input_tensor(prompt_tokens)?;

            // ── Paged KV cache setup ──
            let num_layers = self.model().num_layers();
            let num_kv_heads = self.model().num_kv_heads().unwrap_or(8);
            let head_dim = self.model().head_dim().unwrap_or(64);
            let half_dim = head_dim / 2;
            let kv_dtype = parse_dtype(self.config().dtype())?;
            let block_size = self.config().inference.block_size;

            // Pre-allocate enough blocks for full capacity
            let total_tokens = prompt_tokens.len() + max_tokens;
            let num_blocks = BlockTable::blocks_needed(total_tokens, block_size) + 4;
            let allocator = CpuBlockAllocator::new(num_blocks, block_size);

            let mut paged_cache = LayeredPagedKvCache::new(
                num_layers, num_blocks, block_size, num_kv_heads, head_dim, kv_dtype, self.device(),
            )?;

            // Allocate all blocks upfront for stable addresses during graph capture
            paged_cache.allocate_blocks(total_tokens, &allocator)
                .map_err(|e| anyhow!("Failed to pre-allocate paged blocks: {}", e))?;

            // ── Prefill via normal paged path ──
            let slot_mapping_vec = paged_cache.compute_slot_mapping(0, prompt_tokens.len())
                .map_err(|e| anyhow!("Failed to compute prefill slot mapping: {}", e))?;
            let prefill_slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[prompt_tokens.len()], self.device())?;

            let bt_vec = paged_cache.block_table_device_format(0);
            let max_num_blocks_bt = bt_vec.len();
            let block_table = Tensor::from_slice(&bt_vec, &[1, max_num_blocks_bt], self.device())?;

            let seq_len_k = prompt_tokens.len();
            paged_cache.set_seq_len(seq_len_k);

            let t0 = std::time::Instant::now();
            let prefill_logits = self.model().forward_with_paged_kv_cache(
                &input, &paged_cache, &prefill_slot_mapping, &block_table,
                seq_len_k, 0, None, None,
            ).map_err(|e| anyhow!("Paged graph prefill failed: {}", e))?;
            tracing::info!("Paged graph prefill: {:?} (seq_len={})", t0.elapsed(), prompt_tokens.len());

            // ── Extract RoPE tables ──
            let (rope_cos_var, rope_sin_var) = self.model()
                .rope_caches()
                .ok_or_else(|| anyhow!("Model has no RoPE cache — cannot use CUDA graph mode"))?;
            let rope_cos_cache = rope_cos_var.tensor().clone();
            let rope_sin_cache = rope_sin_var.tensor().clone();

            let client = CudaRuntime::default_client(self.device());

            // ── Stable-address tensors (ALL allocated BEFORE graph capture) ──
            let token_buf = boostr::Tensor::<CudaRuntime>::zeros(&[1, 1], boostr::DType::I64, self.device())?;
            let cos_slice_t = boostr::Tensor::<CudaRuntime>::zeros(&[1, half_dim], rope_cos_cache.dtype(), self.device())?;
            let sin_slice_t = boostr::Tensor::<CudaRuntime>::zeros(&[1, half_dim], rope_sin_cache.dtype(), self.device())?;
            let cos_slice = Var::new(cos_slice_t, false);
            let sin_slice = Var::new(sin_slice_t, false);

            // Slot mapping for single-token decode — stable address, value updated per step
            let slot_mapping = boostr::Tensor::<CudaRuntime>::from_slice(
                &[seq_len_k as i32], &[1], self.device(),
            )?;

            let mut device_scalars = DeviceScalars::new(&[seq_len_k], self.device())?;
            let next_token_buf = boostr::Tensor::<CudaRuntime>::zeros(&[1], boostr::DType::I64, self.device())?;

            // ── Warmup pass (JIT kernels) ──
            device_scalars.update(&client, &[seq_len_k])
                .map_err(|e| anyhow!("DeviceScalars warmup update failed: {}", e))?;
            let warmup_logits = self.model().forward_graph_paged(
                &client, &token_buf, &paged_cache, &slot_mapping, &block_table,
                &device_scalars, &cos_slice, &sin_slice,
            ).map_err(|e| anyhow!("Paged graph warmup forward failed: {}", e))?;
            boostr::inference::decode_graph::argmax_to_buf(&client, &warmup_logits, &next_token_buf)
                .map_err(|e| anyhow!("Warmup argmax_to_buf failed: {}", e))?;

            // Re-prefill with fresh paged cache for capture
            let mut paged_cache = LayeredPagedKvCache::new(
                num_layers, num_blocks, block_size, num_kv_heads, head_dim, kv_dtype, self.device(),
            )?;
            paged_cache.allocate_blocks(total_tokens, &allocator)
                .map_err(|e| anyhow!("Failed to re-allocate paged blocks for capture: {}", e))?;

            let slot_mapping_vec = paged_cache.compute_slot_mapping(0, prompt_tokens.len())
                .map_err(|e| anyhow!("Failed to re-compute prefill slot mapping: {}", e))?;
            let prefill_slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[prompt_tokens.len()], self.device())?;
            let bt_vec = paged_cache.block_table_device_format(0);
            let block_table = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], self.device())?;
            paged_cache.set_seq_len(seq_len_k);

            let _ = self.model().forward_with_paged_kv_cache(
                &input, &paged_cache, &prefill_slot_mapping, &block_table,
                seq_len_k, 0, None, None,
            ).map_err(|e| anyhow!("Re-prefill for capture failed: {}", e))?;

            let capture_seq_len = paged_cache.seq_len();
            device_scalars.update(&client, &[capture_seq_len])
                .map_err(|e| anyhow!("DeviceScalars pre-capture update failed: {}", e))?;
            device_scalars.update_rope_slices(&client, &rope_cos_cache, &rope_sin_cache, &cos_slice, &sin_slice, capture_seq_len, half_dim)
                .map_err(|e| anyhow!("RoPE pre-capture update failed: {}", e))?;

            // ── CUDA graph capture ──
            let graph = CudaRuntime::capture_graph_into(&client, &[&token_buf], &[&next_token_buf], |c| {
                let logits = self.model().forward_graph_paged(
                    c, &token_buf, &paged_cache, &slot_mapping, &block_table,
                    &device_scalars, &cos_slice, &sin_slice,
                ).map_err(|e| boostr::NumrError::Backend(format!("Capture paged forward failed: {e}")))?;
                boostr::inference::decode_graph::argmax_to_buf(c, &logits, &next_token_buf)
                    .map_err(|e| boostr::NumrError::Backend(format!("Capture argmax_to_buf failed: {e}")))
            }).map_err(|e| anyhow!("CUDA paged graph capture failed: {}", e))?;

            tracing::info!("Paged CUDA graph captured (seq_len={})", paged_cache.seq_len());

            let mut decode_graph = PagedDecodeGraph {
                graph,
                device_scalars,
                token_buf,
                cos_slice: cos_slice.tensor().clone(),
                sin_slice: sin_slice.tensor().clone(),
                rope_cos_cache,
                rope_sin_cache,
                next_token_buf,
                slot_mapping,
                head_dim: half_dim,
                seq_len: paged_cache.seq_len(),
            };

            // ── First token from prefill logits ──
            let first_token_gpu = self.argmax_on_gpu(&prefill_logits)?;
            let first_event = first_token_gpu.record_event()
                .map_err(|e| anyhow!("Record event failed: {}", e))?;
            let first_token = Self::read_token_id(&first_token_gpu, first_event)?;

            if self.tokenizer().is_eos(first_token) {
                yield Ok(Self::eos_token(first_token));
                return;
            }
            yield Ok(self.text_token(first_token, None));

            decode_graph.seed_next_token(&client, first_token as i64)
                .map_err(|e| anyhow!("Failed to seed next_token_buf: {}", e))?;

            // ── Graph decode loop ──
            let graph_max = max_tokens.saturating_sub(1);
            for i in 0..graph_max {
                let t1 = std::time::Instant::now();

                decode_graph.pre_replay_and_launch(&client)
                    .map_err(|e| anyhow!("Paged graph pre-replay+launch failed: {}", e))?;
                let fwd_time = t1.elapsed();

                let event = decode_graph.next_token_buf.record_event()
                    .map_err(|e| anyhow!("Record event failed: {}", e))?;

                let t2 = std::time::Instant::now();
                let next_token = Self::read_token_id(&decode_graph.next_token_buf, event)?;
                tracing::info!("Paged graph token {}: fwd={:?} sync={:?}", i + 1, fwd_time, t2.elapsed());

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
