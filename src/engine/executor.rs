//! Inference executor
//!
//! Runs inference on loaded models using boostr's LoadedModel.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::{BlockTable, CpuBlockAllocator};
use boostr::inference::{LayeredKvCache, LayeredSsmState};
use boostr::model::LoadedModel;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::{parse_dtype, BlazrConfig, GenerationConfig};
use crate::model::chat_template::ChatTemplate;
use crate::tokenizer::{BoxedTokenizer, TokenizerTrait};

/// Inference executor
///
/// Wraps a loaded model and provides text generation capabilities.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct Executor<R: Runtime<DType = DType>> {
    /// The loaded model
    model: Arc<LoadedModel<R>>,
    /// Model configuration
    config: BlazrConfig,
    /// Tokenizer (boxed to allow different tokenizer types)
    tokenizer: BoxedTokenizer,
    /// Device
    device: R::Device,
    /// Initial context size for KV cache (like Ollama's num_ctx)
    num_ctx: usize,
    /// Chat template for this model
    chat_template: ChatTemplate,
}

impl<R: Runtime<DType = DType>> Executor<R>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ConvOps<R>
        + NormalizationOps<R>
        + UnaryOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + TypeConversionOps<R>
        + SamplingOps<R>
        + ModelClient<R>,
{
    /// Create a new executor
    ///
    /// # Arguments
    /// * `model` - The loaded model
    /// * `config` - Model configuration
    /// * `tokenizer` - Tokenizer for encoding/decoding (boxed trait object)
    /// * `device` - Device to run on
    /// * `num_ctx` - Initial context size for KV cache (like Ollama's num_ctx).
    ///   The KV cache will grow dynamically if more tokens are needed,
    ///   up to the model's max_position_embeddings.
    pub fn new<T: TokenizerTrait + 'static>(
        model: LoadedModel<R>,
        config: BlazrConfig,
        tokenizer: T,
        device: R::Device,
        num_ctx: usize,
    ) -> Result<Self> {
        let chat_template = ChatTemplate::from_model_type(config.model_type());
        Ok(Self {
            model: Arc::new(model),
            config,
            tokenizer: Box::new(tokenizer),
            device,
            num_ctx,
            chat_template,
        })
    }

    /// Create a new executor with a specific chat template detected from model directory
    pub fn with_chat_template(mut self, template: ChatTemplate) -> Self {
        self.chat_template = template;
        self
    }

    /// Get the chat template for this model
    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// Get model configuration
    pub fn config(&self) -> &BlazrConfig {
        &self.config
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &dyn TokenizerTrait {
        self.tokenizer.as_ref()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// Warm up the model by running a dummy forward pass
    ///
    /// This pre-loads all CUDA PTX modules and triggers JIT compilation,
    /// eliminating the ~90ms first-run overhead from TTFT.
    /// Call this after model loading, before the first generation.
    pub fn warmup(&self) -> Result<()> {
        tracing::debug!("Warming up model kernels...");
        let start = std::time::Instant::now();

        // Create a single-token input
        let warmup_input = Tensor::from_slice(&[1i64], &[1, 1], &self.device);

        if self.model.needs_ssm_state() {
            // Mamba2: warmup with SSM state
            let mamba_config = self
                .model
                .mamba_config()
                .ok_or_else(|| anyhow!("Mamba2 model missing mamba config"))?;
            let num_layers = self.model.num_layers();

            let warmup_dtype = parse_dtype(self.config.dtype())?;

            let mut ssm_state =
                LayeredSsmState::new(num_layers, 1, mamba_config, warmup_dtype, &self.device);

            let _ = self
                .model
                .forward_with_ssm_state(&warmup_input, &mut ssm_state)
                .map_err(|e| anyhow!("Warmup forward pass failed: {}", e))?;
        } else if self.config.inference.paged_attention {
            // Llama: warmup with paged KV cache (exercises paged decode + reshape_and_cache kernels)
            let num_layers = self.model.num_layers();
            let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
            let head_dim = self.model.head_dim().unwrap_or(64);
            let block_size = self.config.inference.block_size;
            let kv_dtype = parse_dtype(self.config.dtype())?;

            let num_blocks = 4; // enough for warmup
            let allocator = CpuBlockAllocator::new(num_blocks, block_size);

            let mut paged_cache = LayeredPagedKvCache::new(
                num_layers,
                num_blocks,
                block_size,
                num_kv_heads,
                head_dim,
                kv_dtype,
                &self.device,
            );

            // Prefill warmup (1 token) — warms up paged_flash_attention + reshape_and_cache
            paged_cache
                .allocate_blocks(1, &allocator)
                .map_err(|e| anyhow!("Warmup paged block alloc failed: {}", e))?;
            let slot_vec = paged_cache
                .compute_slot_mapping(0, 1)
                .map_err(|e| anyhow!("Warmup slot mapping failed: {}", e))?;
            let slot_mapping = Tensor::from_slice(&slot_vec, &[1], &self.device);
            let bt_vec = paged_cache.block_table_device_format(0);
            let block_table = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], &self.device);
            paged_cache.set_seq_len(1);

            let _ = self
                .model
                .forward_with_paged_kv_cache(
                    &warmup_input,
                    &paged_cache,
                    &slot_mapping,
                    &block_table,
                    1,
                    0,
                )
                .map_err(|e| anyhow!("Warmup paged prefill failed: {}", e))?;

            // Decode warmup (1 token at position 1) — warms up paged_decode_attention kernel
            paged_cache
                .allocate_blocks(1, &allocator)
                .map_err(|e| anyhow!("Warmup paged decode block alloc failed: {}", e))?;
            let slot_vec = paged_cache
                .compute_slot_mapping(1, 1)
                .map_err(|e| anyhow!("Warmup decode slot mapping failed: {}", e))?;
            let slot_mapping = Tensor::from_slice(&slot_vec, &[1], &self.device);
            let bt_vec = paged_cache.block_table_device_format(0);
            let block_table = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], &self.device);
            paged_cache.set_seq_len(2);

            let _ = self
                .model
                .forward_with_paged_kv_cache(
                    &warmup_input,
                    &paged_cache,
                    &slot_mapping,
                    &block_table,
                    2,
                    1,
                )
                .map_err(|e| anyhow!("Warmup paged decode failed: {}", e))?;
        } else {
            // Llama: warmup with contiguous KV cache
            let num_layers = self.model.num_layers();
            let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
            let head_dim = self.model.head_dim().unwrap_or(64);

            let kv_dtype = parse_dtype(self.config.dtype())?;

            let mut kv_cache = LayeredKvCache::new_positional(
                num_layers,
                1,
                num_kv_heads,
                16,
                16,
                head_dim,
                kv_dtype,
                &self.device,
            )
            .map_err(|e| anyhow!("Failed to create warmup KV cache: {}", e))?;

            let _ = self
                .model
                .forward_with_kv_cache(&warmup_input, &mut kv_cache, 0)
                .map_err(|e| anyhow!("Warmup prefill forward pass failed: {}", e))?;

            // Decode warmup (1 token at position 1) — warms up decode_attention + GEMV kernels
            let _ = self
                .model
                .forward_with_kv_cache(&warmup_input, &mut kv_cache, 1)
                .map_err(|e| anyhow!("Warmup decode forward pass failed: {}", e))?;
        }

        // Warmup argmax + pipelined D2H copy to JIT those kernels too
        // Create a small dummy logits tensor matching the vocab size
        let vocab_size = self.model.vocab_size();
        let dummy_logits = Tensor::zeros(&[1, 1, vocab_size], DType::F32, &self.device);
        let token_gpu = self.argmax_on_gpu(&dummy_logits)?;
        let event = token_gpu
            .record_event()
            .map_err(|e| anyhow!("Warmup record event failed: {}", e))?;
        let _ = Self::read_token_id(&token_gpu, event)?;

        tracing::debug!("Model warmup complete in {:?}", start.elapsed());
        Ok(())
    }

    /// Generate text from a prompt
    ///
    /// Returns a stream of generated tokens.
    pub fn generate<'a>(
        &'a self,
        prompt: &'a str,
        gen_config: &'a GenerationConfig,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            // Encode prompt
            let prompt_tokens = self.tokenizer.encode(prompt)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

            if gen_config.verbose_prompt {
                eprintln!("\nprompt: '{}'", prompt);
                eprintln!("number of tokens in prompt = {}", prompt_tokens.len());
                for &tok in &prompt_tokens {
                    let piece = self.tokenizer.decode(&[tok]).unwrap_or_default();
                    eprintln!("{:>6} -> '{}'", tok, piece);
                }
                eprintln!();
            }

            if prompt_tokens.is_empty() {
                return;
            }

            let max_seq_len = self.config.max_seq_len();
            let max_tokens = gen_config.max_tokens.min(
                max_seq_len.saturating_sub(prompt_tokens.len())
            );

            // Create input tensor
            let input = self.create_input_tensor(&prompt_tokens)?;

            // Token history for repetition penalty
            let mut token_history: Vec<u32> = prompt_tokens.clone();

            if self.model.needs_ssm_state() {
                // ── Mamba2 path: SSM state instead of KV cache ──
                let mamba_config = self.model.mamba_config()
                    .ok_or_else(|| anyhow!("Mamba2 model missing mamba config"))?;
                let num_layers = self.model.num_layers();

                let state_dtype = parse_dtype(self.config.dtype())?;

                let mut ssm_state = LayeredSsmState::new(
                    num_layers,
                    1,
                    mamba_config,
                    state_dtype,
                    &self.device,
                );

                // Prefill
                tracing::debug!("Starting Mamba2 prefill...");
                let mut logits = self.model.forward_with_ssm_state(&input, &mut ssm_state)
                    .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                tracing::debug!("Mamba2 prefill complete");

                // Generate tokens — unified pipelined decode
                for i in 0..max_tokens {
                    tracing::debug!("Generating token {} / {}", i + 1, max_tokens);

                    let token_gpu = self.logits_to_token_on_device(&logits, &token_history, gen_config)?;
                    let event = token_gpu.record_event()
                        .map_err(|e| anyhow!("Record event failed: {}", e))?;

                    // Decode step: feed token on GPU directly (no CPU round-trip)
                    let next_input = token_gpu.reshape(&[1, 1])?;
                    logits = self.model.forward_with_ssm_state(&next_input, &mut ssm_state)
                        .map_err(|e| anyhow!("Forward pass failed: {}", e))?;

                    // Overlap: read token via pipelined copy while forward pass runs
                    let next_token = Self::read_token_id(&token_gpu, event)?;
                    token_history.push(next_token);

                    if self.tokenizer.is_eos(next_token) {
                        tracing::debug!("Hit EOS token, stopping generation");
                        yield Ok(GeneratedToken {
                            token_id: next_token,
                            text: String::new(),
                            logprob: None,
                            finish_reason: Some(FinishReason::Eos),
                        });
                        break;
                    }

                    let is_last = i + 1 == max_tokens;
                    let text = self.tokenizer.decode(&[next_token]).unwrap_or_default();

                    yield Ok(GeneratedToken {
                        token_id: next_token,
                        text,
                        logprob: None,
                        finish_reason: if is_last { Some(FinishReason::Length) } else { None },
                    });
                }
            } else if self.config.inference.paged_attention {
                // ── Llama path: Paged KV cache ──
                let num_layers = self.model.num_layers();
                let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
                let head_dim = self.model.head_dim().unwrap_or(64);
                let block_size = self.config.inference.block_size;
                let kv_dtype = parse_dtype(self.config.dtype())?;

                // Compute number of blocks needed
                let total_tokens = prompt_tokens.len() + max_tokens;
                let num_blocks = if self.config.inference.num_blocks > 0 {
                    self.config.inference.num_blocks
                } else {
                    BlockTable::blocks_needed(total_tokens, block_size) + 4 // small headroom
                };

                let allocator = CpuBlockAllocator::new(num_blocks, block_size);

                let mut paged_cache = LayeredPagedKvCache::new(
                    num_layers,
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    head_dim,
                    kv_dtype,
                    &self.device,
                );

                // Allocate blocks for prompt
                paged_cache.allocate_blocks(prompt_tokens.len(), &allocator)
                    .map_err(|e| anyhow!("Failed to allocate blocks for prefill: {}", e))?;

                // Compute slot mapping for prompt tokens
                let slot_mapping_vec = paged_cache.compute_slot_mapping(0, prompt_tokens.len())
                    .map_err(|e| anyhow!("Failed to compute slot mapping: {}", e))?;
                let slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[prompt_tokens.len()], &self.device);

                // Block table tensor [1, max_num_blocks]
                let bt_vec = paged_cache.block_table_device_format(0);
                let max_num_blocks = bt_vec.len();
                let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, max_num_blocks], &self.device);

                let seq_len_k = prompt_tokens.len();
                paged_cache.set_seq_len(seq_len_k);

                // Prefill
                let t0 = std::time::Instant::now();
                let mut logits = self.model.forward_with_paged_kv_cache(
                    &input, &paged_cache, &slot_mapping, &block_table_tensor, seq_len_k, 0,
                ).map_err(|e| anyhow!("Paged prefill failed: {}", e))?;
                tracing::info!("Paged prefill: {:?} (seq_len={}, blocks={})", t0.elapsed(), prompt_tokens.len(), num_blocks);

                // ── Unified pipelined paged decode (all paths: greedy, penalties, non-greedy) ──
                for i in 0..max_tokens {
                    let t1 = std::time::Instant::now();

                    let token_gpu = self.logits_to_token_on_device(&logits, &token_history, gen_config)?;
                    let event = token_gpu.record_event()
                        .map_err(|e| anyhow!("Record event failed: {}", e))?;

                    let cur_seq_len = paged_cache.seq_len();
                    paged_cache.allocate_blocks(1, &allocator)
                        .map_err(|e| anyhow!("Failed to allocate block for decode: {}", e))?;

                    let slot_vec = paged_cache.compute_slot_mapping(cur_seq_len, 1)
                        .map_err(|e| anyhow!("Failed to compute decode slot mapping: {}", e))?;
                    let slot_mapping = Tensor::from_slice(&slot_vec, &[1], &self.device);

                    let bt_vec = paged_cache.block_table_device_format(0);
                    let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], &self.device);

                    let new_seq_len_k = cur_seq_len + 1;
                    paged_cache.set_seq_len(new_seq_len_k);

                    let next_input = token_gpu.reshape(&[1, 1])?;
                    let next_logits = self.model.forward_with_paged_kv_cache(
                        &next_input, &paged_cache, &slot_mapping, &block_table_tensor,
                        new_seq_len_k, cur_seq_len,
                    ).map_err(|e| anyhow!("Paged decode failed: {}", e))?;
                    let fwd_time = t1.elapsed();

                    let t2 = std::time::Instant::now();
                    let next_token = Self::read_token_id(&token_gpu, event)?;
                    token_history.push(next_token);
                    tracing::info!("Paged token {}: fwd_launch={:?} sync={:?} total={:?}", i+1, fwd_time, t2.elapsed(), t1.elapsed());

                    if self.tokenizer.is_eos(next_token) {
                        yield Ok(GeneratedToken { token_id: next_token, text: String::new(), logprob: None, finish_reason: Some(FinishReason::Eos) });
                        break;
                    }

                    let is_last = i + 1 == max_tokens;
                    let text = self.tokenizer.decode(&[next_token]).unwrap_or_default();
                    yield Ok(GeneratedToken { token_id: next_token, text, logprob: None, finish_reason: if is_last { Some(FinishReason::Length) } else { None } });

                    logits = next_logits;
                }
            } else {
                // ── Llama path: contiguous KV cache ──
                let num_layers = self.model.num_layers();
                let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
                let head_dim = self.model.head_dim().unwrap_or(64);
                // Allocate exactly enough for prompt + max generation tokens.
                // This avoids expensive contiguous() copies when flash attention
                // reads the narrowed KV cache view (narrow of a larger buffer is
                // non-contiguous across heads).
                let initial_capacity = (prompt_tokens.len() + max_tokens).min(max_seq_len);

                let kv_dtype = parse_dtype(self.config.dtype())?;

                let mut kv_cache = LayeredKvCache::new_positional(
                    num_layers,
                    1,
                    num_kv_heads,
                    initial_capacity,
                    max_seq_len,
                    head_dim,
                    kv_dtype,
                    &self.device,
                ).map_err(|e| anyhow!("Failed to create KV cache: {}", e))?;

                // Prefill
                let t0 = std::time::Instant::now();
                let mut logits = self.model.forward_with_kv_cache(&input, &mut kv_cache, 0)
                    .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                tracing::info!("Prefill: {:?} (seq_len={}, kv={})", t0.elapsed(), prompt_tokens.len(), kv_cache.seq_len());

                // ── Unified pipelined decode (all paths: greedy, penalties, non-greedy) ──
                for i in 0..max_tokens {
                    let t1 = std::time::Instant::now();

                    let token_gpu = self.logits_to_token_on_device(&logits, &token_history, gen_config)?;
                    let event = token_gpu.record_event()
                        .map_err(|e| anyhow!("Record event failed: {}", e))?;

                    let next_input = token_gpu.reshape(&[1, 1])?;
                    let position = kv_cache.seq_len();
                    let next_logits = self.model.forward_with_kv_cache(&next_input, &mut kv_cache, position)
                        .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                    let fwd_time = t1.elapsed();

                    let t2 = std::time::Instant::now();
                    let next_token = Self::read_token_id(&token_gpu, event)?;
                    token_history.push(next_token);
                    tracing::trace!("Token {}: id={} fwd_launch={:?} sync={:?} total={:?}", i+1, next_token, fwd_time, t2.elapsed(), t1.elapsed());

                    if self.tokenizer.is_eos(next_token) {
                        yield Ok(GeneratedToken { token_id: next_token, text: String::new(), logprob: None, finish_reason: Some(FinishReason::Eos) });
                        break;
                    }

                    let is_last = i + 1 == max_tokens;
                    let text = self.tokenizer.decode(&[next_token]).unwrap_or_default();
                    yield Ok(GeneratedToken { token_id: next_token, text, logprob: None, finish_reason: if is_last { Some(FinishReason::Length) } else { None } });

                    logits = next_logits;
                }
            }

            tracing::debug!("Generation loop complete");
        }
    }

    /// Generate text and return the complete result with metadata
    pub async fn generate_text(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<GenerationResult> {
        use futures::StreamExt;

        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?
            .len();

        let mut result = String::new();
        let mut completion_tokens = 0usize;
        let mut finish_reason = FinishReason::Length; // default if stream ends without explicit reason
        let mut stream = std::pin::pin!(self.generate(prompt, gen_config));
        let prefill_start = std::time::Instant::now();
        let mut prompt_eval_duration_ms = 0u64;

        while let Some(token_result) = stream.next().await {
            let token = token_result?;
            completion_tokens += 1;

            // First token marks end of prefill
            if completion_tokens == 1 {
                prompt_eval_duration_ms = prefill_start.elapsed().as_millis() as u64;
            }

            result.push_str(&token.text);

            if let Some(reason) = token.finish_reason {
                finish_reason = reason;
            }

            // Check stop sequences
            for stop in &gen_config.stop_sequences {
                if result.ends_with(stop) {
                    result.truncate(result.len() - stop.len());
                    return Ok(GenerationResult {
                        text: result,
                        prompt_tokens,
                        completion_tokens,
                        finish_reason: FinishReason::Stop,
                        prompt_eval_duration_ms,
                    });
                }
            }
        }

        Ok(GenerationResult {
            text: result,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            prompt_eval_duration_ms,
        })
    }

    /// Create input tensor from token IDs
    fn create_input_tensor(&self, tokens: &[u32]) -> Result<Tensor<R>> {
        // Model embedding expects I64 indices
        let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        Ok(Tensor::from_slice(
            &tokens_i64,
            &[1, tokens.len()],
            &self.device,
        ))
    }

    /// Greedy argmax on GPU — returns the token as a GPU tensor [1] i64.
    /// No CPU sync happens here; the result stays on device.
    /// Used for warmup and graph-mode prefill token extraction.
    fn argmax_on_gpu(&self, logits: &Tensor<R>) -> Result<Tensor<R>> {
        let seq_len = logits.dim(1).map_err(|e| anyhow!("{}", e))?;
        let narrowed = logits
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| anyhow!("{}", e))?;
        let squeezed = narrowed.squeeze(Some(1)).squeeze(Some(0));
        squeezed.argmax(0, false).map_err(|e| anyhow!("{}", e))
    }

    /// Read a scalar i64 GPU tensor to CPU using the pipelined copy stream.
    /// Only syncs the copy stream — compute stream continues running.
    fn read_token_id(token_gpu: &Tensor<R>, event: u64) -> Result<u32> {
        let v: Vec<i64> = token_gpu
            .to_vec_pipelined(event)
            .map_err(|e| anyhow!("Pipelined D2H copy failed: {}", e))?;
        Ok(v[0] as u32)
    }

    /// Compute unique token IDs and counts from the penalty window, returning
    /// (token_ids, token_counts) as slices ready for the penalty kernel.
    fn penalty_window(recent_tokens: &[u32], repeat_last_n: usize) -> (Vec<i64>, Vec<i32>) {
        let window = if repeat_last_n > 0 && repeat_last_n < recent_tokens.len() {
            &recent_tokens[recent_tokens.len() - repeat_last_n..]
        } else {
            recent_tokens
        };

        let mut counts = std::collections::HashMap::<u32, u32>::new();
        for &tok in window {
            *counts.entry(tok).or_insert(0) += 1;
        }

        let mut ids: Vec<i64> = Vec::with_capacity(counts.len());
        let mut cnts: Vec<i32> = Vec::with_capacity(counts.len());
        for (&tok, &count) in &counts {
            ids.push(tok as i64);
            cnts.push(count as i32);
        }
        (ids, cnts)
    }

    /// Fused logits-to-token: narrow + cast + penalties + argmax/sample in a single kernel launch.
    ///
    /// Returns `[1]` I64 tensor on device. All decode paths (greedy, greedy+penalties,
    /// non-greedy) return the same type, enabling uniform pipelined decode.
    fn logits_to_token_on_device(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<Tensor<R>> {
        // Apply logit_bias before sampling if any biases are specified
        let logits = if !gen_config.logit_bias.is_empty() {
            self.apply_logit_bias(logits, &gen_config.logit_bias)?
        } else {
            logits.clone()
        };

        let (ids, cnts) = Self::penalty_window(recent_tokens, gen_config.repeat_last_n);

        let ids_tensor = Tensor::from_slice(&ids, &[ids.len()], &self.device);
        let cnts_tensor = Tensor::from_slice(&cnts, &[cnts.len()], &self.device);

        let client = R::default_client(&self.device);
        let temperature = if gen_config.is_greedy() {
            0.0
        } else {
            gen_config.temperature
        };

        client
            .logits_to_token(
                &logits,
                &ids_tensor,
                &cnts_tensor,
                ids.len(),
                gen_config.repeat_penalty,
                gen_config.frequency_penalty,
                gen_config.presence_penalty,
                temperature,
                gen_config.top_k,
                gen_config.top_p,
                gen_config.min_p,
                gen_config.seed,
            )
            .map_err(|e| anyhow!("logits_to_token failed: {}", e))
    }

    /// Apply per-token logit biases by creating a sparse bias tensor and adding to logits.
    fn apply_logit_bias(
        &self,
        logits: &Tensor<R>,
        bias_map: &std::collections::HashMap<u32, f32>,
    ) -> Result<Tensor<R>> {
        let vocab_size = self.model.vocab_size();
        let mut bias_vec = vec![0.0f32; vocab_size];
        for (&token_id, &bias) in bias_map {
            if (token_id as usize) < vocab_size {
                bias_vec[token_id as usize] = bias;
            }
        }
        let bias_tensor = Tensor::from_slice(&bias_vec, &[1, 1, vocab_size], &self.device);
        logits
            .add(&bias_tensor)
            .map_err(|e| anyhow!("logit_bias add failed: {}", e))
    }
}

/// Graph-accelerated greedy decode loop (CUDA backend implementation).
///
/// The config flag `inference.graphs` is backend-agnostic. For now the graph-mode
/// forward pass (`forward_graph_mode`) uses CUDA-specific kernel APIs
/// (device-ptr seq_len, kv_insert kernel), so this impl block is CUDA-only.
/// When ROCm/Metal graph support is added, a parallel impl block will be added
/// for those runtimes using their respective graph-mode ops.
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
        gen_config: &'a GenerationConfig,
    ) -> impl futures::Stream<Item = Result<GeneratedToken>> + 'a {
        use boostr::autograd::Var;
        use boostr::inference::decode_graph::{DecodeGraph, DeviceScalars};
        use boostr::runtime::cuda::CudaRuntime as CudaRt;
        use boostr::CudaRuntime;
        use boostr::Runtime;

        stream! {
            let prompt_tokens = self.tokenizer.encode(prompt)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

            if prompt_tokens.is_empty() {
                return;
            }

            let max_seq_len = self.config.max_seq_len();
            // Graph mode pre-allocates KV cache at fixed capacity.
            // Use num_ctx (user-specified) capped to model max_seq_len.
            let graph_capacity = self.num_ctx.min(max_seq_len);
            let max_tokens = gen_config.max_tokens.min(
                graph_capacity.saturating_sub(prompt_tokens.len())
            );

            let input = self.create_input_tensor(&prompt_tokens)?;

            // ── KV cache at full capacity (required for stable device addresses) ──
            let num_layers = self.model.num_layers();
            let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
            let head_dim = self.model.head_dim().unwrap_or(64);
            let half_dim = head_dim / 2;
            let kv_dtype = parse_dtype(self.config.dtype())?;

            let mut kv_cache = boostr::inference::LayeredKvCache::new_positional(
                num_layers, 1, num_kv_heads, graph_capacity, graph_capacity, head_dim, kv_dtype, &self.device,
            ).map_err(|e| anyhow!("Failed to create full-capacity KV cache: {}", e))?;

            // ── Prefill ──
            let t0 = std::time::Instant::now();
            let prefill_logits = self.model.forward_with_kv_cache(&input, &mut kv_cache, 0)
                .map_err(|e| anyhow!("Prefill forward pass failed: {}", e))?;
            tracing::info!("Graph-mode prefill: {:?} (seq_len={})", t0.elapsed(), prompt_tokens.len());

            // ── Extract RoPE tables from model ──
            let (rope_cos_var, rope_sin_var) = self.model
                .rope_caches()
                .ok_or_else(|| anyhow!("Model has no RoPE cache — cannot use CUDA graph mode"))?;
            let rope_cos_cache = rope_cos_var.tensor().clone();
            let rope_sin_cache = rope_sin_var.tensor().clone();

            let client = CudaRt::default_client(&self.device);

            // ── Stable-address decode inputs (ALL allocated BEFORE graph capture) ──
            // Any tensor allocated after capture starts has a graph-managed address
            // that is only valid inside the graph's execution, not from the CPU.
            //
            // token_buf: [1, 1] i64 — filled via D2D from next_token_buf each step
            let token_buf = boostr::Tensor::<CudaRuntime>::zeros(&[1, 1], boostr::DType::I64, &self.device);

            // cos_slice / sin_slice: [1, half_dim] f32 — updated via D2D before each replay
            let cos_slice_t = boostr::Tensor::<CudaRuntime>::zeros(&[1, half_dim], boostr::DType::F32, &self.device);
            let sin_slice_t = boostr::Tensor::<CudaRuntime>::zeros(&[1, half_dim], boostr::DType::F32, &self.device);
            let cos_slice = Var::new(cos_slice_t, false);
            let sin_slice = Var::new(sin_slice_t, false);

            let device_scalars = DeviceScalars::new(kv_cache.seq_len(), &self.device);

            // next_token_buf: [1] i64 — output written by graph via captured cuMemcpyAsync.
            // Allocated before capture so it has a stable CPU-accessible address.
            let next_token_buf = boostr::Tensor::<CudaRuntime>::zeros(&[1], boostr::DType::I64, &self.device);

            // ── Warmup pass: JIT all kernels, do NOT capture yet ──
            device_scalars.update(&client, kv_cache.seq_len())
                .map_err(|e| anyhow!("DeviceScalars warmup update failed: {}", e))?;
            let warmup_logits = self.model.forward_graph_mode(
                &token_buf, &mut kv_cache, &device_scalars, &cos_slice, &sin_slice,
            ).map_err(|e| anyhow!("Graph-mode warmup forward failed: {}", e))?;
            boostr::inference::decode_graph::argmax_to_buf(&client, &warmup_logits, &next_token_buf)
                .map_err(|e| anyhow!("Warmup argmax_to_buf failed: {}", e))?;

            // Re-prefill with a fresh KV cache so capture starts from the correct state
            let mut kv_cache = boostr::inference::LayeredKvCache::new_positional(
                num_layers, 1, num_kv_heads, graph_capacity, graph_capacity, head_dim, kv_dtype, &self.device,
            ).map_err(|e| anyhow!("Failed to re-create KV cache for capture: {}", e))?;
            let _ = self.model.forward_with_kv_cache(&input, &mut kv_cache, 0)
                .map_err(|e| anyhow!("Re-prefill for capture failed: {}", e))?;

            let capture_seq_len = kv_cache.seq_len();
            device_scalars.update(&client, capture_seq_len)
                .map_err(|e| anyhow!("DeviceScalars pre-capture update failed: {}", e))?;
            device_scalars.update_rope_slices(&client, &rope_cos_cache, &rope_sin_cache, &cos_slice, &sin_slice, capture_seq_len, half_dim)
                .map_err(|e| anyhow!("RoPE pre-capture update failed: {}", e))?;

            // ── CUDA graph capture ──
            // The closure runs forward_graph_mode and then argmax_to_buf.
            // argmax_to_buf records a cuMemcpyAsync node that writes the argmax result
            // into next_token_buf (stable, pre-allocated address).
            // CUDA internally patches the graph-internal argmax source address on replay.
            let (graph, ()) = CudaRt::capture_graph(&client, |c| {
                let logits = self.model.forward_graph_mode(
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

            if self.tokenizer.is_eos(first_token) {
                yield Ok(GeneratedToken { token_id: first_token, text: String::new(), logprob: None, finish_reason: Some(FinishReason::Eos) });
                return;
            }
            let text = self.tokenizer.decode(&[first_token]).unwrap_or_default();
            yield Ok(GeneratedToken { token_id: first_token, text, logprob: None, finish_reason: None });

            // Seed next_token_buf with the first token so pre_replay_and_launch can
            // D2D-copy it into token_buf for the first decode step.
            decode_graph.seed_next_token(&client, first_token as i64)
                .map_err(|e| anyhow!("Failed to seed next_token_buf: {}", e))?;

            // ── Graph decode loop (event-based sync, no full device sync) ──
            for i in 0..max_tokens.saturating_sub(1) {
                let t1 = std::time::Instant::now();

                // D2D: next_token_buf → token_buf, update scalars + RoPE, launch graph.
                // Graph writes argmax into next_token_buf via captured cuMemcpyAsync node.
                decode_graph.pre_replay_and_launch(&client)
                    .map_err(|e| anyhow!("Graph pre-replay+launch failed: {}", e))?;
                let fwd_time = t1.elapsed();

                // Record event after graph launch — fires when graph completes on compute stream
                let event = decode_graph.next_token_buf.record_event()
                    .map_err(|e| anyhow!("Record event failed: {}", e))?;

                // Read token via copy stream (waits on event, not full device sync)
                let t2 = std::time::Instant::now();
                let next_token = Self::read_token_id(&decode_graph.next_token_buf, event)?;
                tracing::info!("Graph token {}: fwd={:?} sync={:?}", i + 1, fwd_time, t2.elapsed());

                if self.tokenizer.is_eos(next_token) {
                    yield Ok(GeneratedToken { token_id: next_token, text: String::new(), logprob: None, finish_reason: Some(FinishReason::Eos) });
                    break;
                }

                let graph_max = max_tokens.saturating_sub(1);
                let is_last = i + 1 == graph_max;
                let text = self.tokenizer.decode(&[next_token]).unwrap_or_default();
                yield Ok(GeneratedToken { token_id: next_token, text, logprob: None, finish_reason: if is_last { Some(FinishReason::Length) } else { None } });
            }
        }
    }
}

/// Why generation stopped
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit an EOS token
    Eos,
    /// Reached max_tokens limit
    Length,
    /// Matched a stop sequence
    Stop,
}

impl FinishReason {
    /// OpenAI API string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Eos => "stop",
            FinishReason::Length => "length",
            FinishReason::Stop => "stop",
        }
    }
}

/// A generated token with metadata
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    /// Token ID
    pub token_id: u32,
    /// Decoded text
    pub text: String,
    /// Log probability (if computed)
    pub logprob: Option<f32>,
    /// Set on the final token to indicate why generation ended
    pub finish_reason: Option<FinishReason>,
}

/// Result of a complete (non-streaming) generation
#[derive(Debug)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,
    /// Number of prompt tokens
    pub prompt_tokens: usize,
    /// Number of completion tokens generated
    pub completion_tokens: usize,
    /// Why generation finished
    pub finish_reason: FinishReason,
    /// Time to first token (prefill duration) in milliseconds
    pub prompt_eval_duration_ms: u64,
}
