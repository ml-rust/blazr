//! Batched CUDA graph capture and replay for multi-sequence decode.
//!
//! Provides `capture_batched_graph` and `replay_batched_graph` on
//! `Executor<CudaRuntime>`, plus the `BatchedGraphState` type that holds all
//! stable-address tensors required by the captured graph.

#[cfg(feature = "cuda")]
use anyhow::{anyhow, Result};

#[cfg(feature = "cuda")]
use boostr::runtime::Graph;

#[cfg(feature = "cuda")]
use super::executor::Executor;

#[cfg(feature = "cuda")]
impl Executor<boostr::CudaRuntime> {
    /// Capture a CUDA graph that decodes one token for up to `max_batch_size` sequences
    /// simultaneously using paged attention.
    ///
    /// # Batched graph protocol
    ///
    /// The captured graph operates on padded tensors of fixed shape:
    /// - `token_buf`:   `[max_batch_size, 1]`  — input token IDs
    /// - `block_table`: `[max_batch_size, max_blocks_per_seq]` — per-sequence block tables
    /// - `slot_mapping`: `[max_batch_size]` — slot indices for the current decode step
    /// - `next_token_buf`: `[max_batch_size]` — output: argmax token per sequence
    ///
    /// At replay time (see `replay_batched_graph`):
    /// - Fill the first `actual_batch` rows with real request data.
    /// - Zero-pad rows `actual_batch..max_batch_size` (dummy sequences are harmlessly
    ///   written to dummy slots and their output tokens are ignored).
    ///
    /// # Arguments
    /// - `paged_cache` — Pre-populated paged KV cache (must have `max_batch_size` block
    ///   tables, each of length `max_blocks_per_seq`). The caller must have already run
    ///   prefill so that `paged_cache.seq_len()` reflects the initial decode position.
    /// - `initial_seq_len_k` — The `seq_len_k` value to use during capture (must equal
    ///   `paged_cache.seq_len()`).
    ///
    /// # Returns
    /// A `BatchedGraphState` ready for replay via `replay_batched_graph`.
    pub fn capture_batched_graph(
        &self,
        paged_cache: &boostr::inference::kv_cache::LayeredPagedKvCache<boostr::CudaRuntime>,
        max_batch_size: usize,
        max_blocks_per_seq: usize,
        initial_seq_len_k: usize,
    ) -> Result<BatchedGraphState> {
        use boostr::autograd::Var;
        use boostr::inference::decode_graph::DeviceScalars;
        use boostr::runtime::cuda::CudaRuntime as CudaRt;
        use boostr::CudaRuntime;
        use boostr::Runtime;

        let head_dim = self.model().head_dim().unwrap_or(64);
        let half_dim = head_dim / 2;

        let client = CudaRt::default_client(self.device());

        // ── Extract RoPE tables ──
        let (rope_cos_var, rope_sin_var) = self.model().rope_caches().ok_or_else(|| {
            anyhow!("Model has no RoPE cache — cannot use batched CUDA graph mode")
        })?;
        let rope_cos_cache = rope_cos_var.tensor().clone();
        let rope_sin_cache = rope_sin_var.tensor().clone();

        // ── Stable-address tensors — ALL allocated BEFORE graph capture ──
        // token_buf: [max_batch_size, 1] — one token per sequence per step
        let token_buf = boostr::Tensor::<CudaRuntime>::zeros(
            &[max_batch_size, 1],
            boostr::DType::I64,
            self.device(),
        );

        // cos/sin slices for the shared decode position
        let cos_slice_t =
            boostr::Tensor::<CudaRuntime>::zeros(&[1, half_dim], boostr::DType::F32, self.device());
        let sin_slice_t =
            boostr::Tensor::<CudaRuntime>::zeros(&[1, half_dim], boostr::DType::F32, self.device());
        let cos_slice = Var::new(cos_slice_t, false);
        let sin_slice = Var::new(sin_slice_t, false);

        // slot_mapping: [max_batch_size] — one decode slot per sequence
        let slot_mapping = boostr::Tensor::<CudaRuntime>::zeros(
            &[max_batch_size],
            boostr::DType::I32,
            self.device(),
        );

        // block_table: [max_batch_size, max_blocks_per_seq]
        let block_table = boostr::Tensor::<CudaRuntime>::zeros(
            &[max_batch_size, max_blocks_per_seq],
            boostr::DType::I32,
            self.device(),
        );

        // output: [max_batch_size] — argmax token per sequence
        let next_token_buf = boostr::Tensor::<CudaRuntime>::zeros(
            &[max_batch_size],
            boostr::DType::I64,
            self.device(),
        );

        let device_scalars = DeviceScalars::new(initial_seq_len_k, self.device());

        // ── Warmup pass: JIT all kernels, do NOT capture yet ──
        device_scalars
            .update(&client, initial_seq_len_k)
            .map_err(|e| anyhow!("BatchedGraph DeviceScalars warmup failed: {}", e))?;
        device_scalars
            .update_rope_slices(
                &client,
                &rope_cos_cache,
                &rope_sin_cache,
                &cos_slice,
                &sin_slice,
                initial_seq_len_k,
                half_dim,
            )
            .map_err(|e| anyhow!("BatchedGraph RoPE warmup failed: {}", e))?;

        let warmup_logits = self
            .model()
            .forward_graph_paged(
                &client,
                &token_buf,
                paged_cache,
                &slot_mapping,
                &block_table,
                &device_scalars,
                &cos_slice,
                &sin_slice,
            )
            .map_err(|e| anyhow!("BatchedGraph warmup forward failed: {}", e))?;
        boostr::inference::decode_graph::batch_argmax_to_buf(
            &client,
            &warmup_logits,
            &next_token_buf,
            max_batch_size,
        )
        .map_err(|e| anyhow!("BatchedGraph warmup batch_argmax failed: {}", e))?;

        tracing::info!(
            "BatchedGraph warmup complete (batch={}, max_blocks_per_seq={}, seq_len_k={})",
            max_batch_size,
            max_blocks_per_seq,
            initial_seq_len_k
        );

        // ── CUDA graph capture ──
        let (graph, ()) = CudaRt::capture_graph(&client, |c| {
            let logits = self
                .model()
                .forward_graph_paged(
                    c,
                    &token_buf,
                    paged_cache,
                    &slot_mapping,
                    &block_table,
                    &device_scalars,
                    &cos_slice,
                    &sin_slice,
                )
                .map_err(|e| {
                    boostr::NumrError::Backend(format!("BatchedGraph capture forward failed: {e}"))
                })?;
            boostr::inference::decode_graph::batch_argmax_to_buf(
                c,
                &logits,
                &next_token_buf,
                max_batch_size,
            )
            .map_err(|e| {
                boostr::NumrError::Backend(format!("BatchedGraph capture argmax failed: {e}"))
            })
        })
        .map_err(|e| anyhow!("BatchedGraph CUDA graph capture failed: {}", e))?;

        tracing::info!(
            "BatchedGraph CUDA graph captured (batch={}, max_blocks={}, seq_len_k={})",
            max_batch_size,
            max_blocks_per_seq,
            initial_seq_len_k
        );

        Ok(BatchedGraphState {
            graph,
            device_scalars,
            token_buf,
            cos_slice: cos_slice.tensor().clone(),
            sin_slice: sin_slice.tensor().clone(),
            rope_cos_cache,
            rope_sin_cache,
            slot_mapping,
            block_table,
            next_token_buf,
            head_dim: half_dim,
            max_batch_size,
            max_blocks_per_seq,
            seq_len: initial_seq_len_k,
        })
    }

    /// Replay a batched CUDA graph for a single decode step.
    ///
    /// Before calling this, the caller must copy the actual request data into the
    /// `BatchedGraphState` buffers:
    /// - `state.token_buf` rows `0..actual_batch` — current token IDs
    /// - `state.block_table` rows `0..actual_batch` — block tables
    /// - `state.slot_mapping` elements `0..actual_batch` — decode slot indices
    ///
    /// Rows `actual_batch..max_batch_size` should be zeroed (dummy sequences).
    ///
    /// After replay, `state.next_token_buf[0..actual_batch]` contains the argmax
    /// token for each active sequence.
    pub fn replay_batched_graph(
        &self,
        state: &mut BatchedGraphState,
        seq_len_k: usize,
    ) -> Result<()> {
        use boostr::runtime::cuda::CudaRuntime as CudaRt;
        use boostr::Runtime;

        let client = CudaRt::default_client(self.device());

        // Update device-side scalars (seq_len_k for attention kernel)
        state
            .device_scalars
            .update(&client, seq_len_k)
            .map_err(|e| anyhow!("BatchedGraph DeviceScalars update failed: {}", e))?;

        // Update RoPE slices for the current position
        let cos_slice_var = boostr::autograd::Var::new(state.cos_slice.clone(), false);
        let sin_slice_var = boostr::autograd::Var::new(state.sin_slice.clone(), false);
        state
            .device_scalars
            .update_rope_slices(
                &client,
                &state.rope_cos_cache,
                &state.rope_sin_cache,
                &cos_slice_var,
                &sin_slice_var,
                seq_len_k,
                state.head_dim,
            )
            .map_err(|e| anyhow!("BatchedGraph RoPE slice update failed: {}", e))?;

        // Replay the captured graph (stream-ordered after the pre-launch copies above)
        state
            .graph
            .launch()
            .map_err(|e| anyhow!("BatchedGraph replay failed: {}", e))?;

        state.seq_len = seq_len_k + 1;
        Ok(())
    }
}

/// State for a batched CUDA graph decode step.
///
/// Holds all stable-address tensors used during graph capture. The graph
/// operates on `max_batch_size` sequences simultaneously; unused slots are
/// zero-padded by the caller before each replay.
///
/// # CUDA graph stability requirements
///
/// Every tensor field in this struct was allocated *before* graph capture and
/// must remain at the same device address for the lifetime of the graph.
/// Do NOT replace any field with a freshly allocated tensor — update the
/// *contents* of each buffer in place before calling `replay_batched_graph`.
#[cfg(feature = "cuda")]
pub struct BatchedGraphState {
    /// Captured CUDA graph (the decode kernel chain for the full batch)
    pub graph: boostr::runtime::cuda::CudaGraph,
    /// Device-side scalar values (seq_len_k pointer used by paged attention kernel)
    pub device_scalars: boostr::inference::decode_graph::DeviceScalars,
    /// Input token IDs: `[max_batch_size, 1]` — fill before each replay
    pub token_buf: boostr::Tensor<boostr::CudaRuntime>,
    /// RoPE cosine slice at the current decode position: `[1, head_dim/2]`
    pub cos_slice: boostr::Tensor<boostr::CudaRuntime>,
    /// RoPE sine slice at the current decode position: `[1, head_dim/2]`
    pub sin_slice: boostr::Tensor<boostr::CudaRuntime>,
    /// Full RoPE cosine cache (source for updating `cos_slice` via D2D copy)
    pub rope_cos_cache: boostr::Tensor<boostr::CudaRuntime>,
    /// Full RoPE sine cache (source for updating `sin_slice` via D2D copy)
    pub rope_sin_cache: boostr::Tensor<boostr::CudaRuntime>,
    /// Decode slot mapping: `[max_batch_size]` — one slot index per sequence
    pub slot_mapping: boostr::Tensor<boostr::CudaRuntime>,
    /// Block table: `[max_batch_size, max_blocks_per_seq]` — fill before each replay
    pub block_table: boostr::Tensor<boostr::CudaRuntime>,
    /// Output token buffer: `[max_batch_size]` — argmax token per sequence after replay
    pub next_token_buf: boostr::Tensor<boostr::CudaRuntime>,
    /// Half of the full head dimension (stored as `head_dim` in `DeviceScalars` convention)
    pub head_dim: usize,
    /// Maximum number of sequences the graph was captured for
    pub max_batch_size: usize,
    /// Maximum number of blocks per sequence in the block table
    pub max_blocks_per_seq: usize,
    /// Current `seq_len_k` (updated by `replay_batched_graph` after each step)
    pub seq_len: usize,
}
