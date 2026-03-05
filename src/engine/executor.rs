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

use super::sampling::MirostatState;
use super::types::{is_valid_json, FinishReason, GeneratedToken, GenerationResult};

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

    /// Get the device
    pub(crate) fn device(&self) -> &R::Device {
        &self.device
    }

    /// Get the loaded model
    #[cfg(feature = "cuda")]
    pub(crate) fn model(&self) -> &LoadedModel<R> {
        &self.model
    }

    /// Get num_ctx
    #[cfg(feature = "cuda")]
    pub(crate) fn num_ctx(&self) -> usize {
        self.num_ctx
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
            // Llama: warmup with paged KV cache
            let num_layers = self.model.num_layers();
            let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
            let head_dim = self.model.head_dim().unwrap_or(64);
            let block_size = self.config.inference.block_size;
            let kv_dtype = parse_dtype(self.config.dtype())?;

            let num_blocks = 4;
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

            // Prefill warmup
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

            // Decode warmup
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

            let _ = self
                .model
                .forward_with_kv_cache(&warmup_input, &mut kv_cache, 1)
                .map_err(|e| anyhow!("Warmup decode forward pass failed: {}", e))?;
        }

        // Warmup argmax + pipelined D2H copy
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

            let input = self.create_input_tensor(&prompt_tokens)?;

            // Token history for repetition penalty
            let mut token_history: Vec<u32> = prompt_tokens.clone();

            // Mirostat state (if enabled)
            let mut mirostat: Option<MirostatState> = if gen_config.mirostat_mode >= 2 {
                Some(MirostatState::new(gen_config.mirostat_tau, gen_config.mirostat_eta, gen_config.seed))
            } else {
                None
            };

            if self.model.needs_ssm_state() {
                // ── Mamba2 path: SSM state instead of KV cache ──
                let mamba_config = self.model.mamba_config()
                    .ok_or_else(|| anyhow!("Mamba2 model missing mamba config"))?;
                let num_layers = self.model.num_layers();

                let state_dtype = parse_dtype(self.config.dtype())?;

                let mut ssm_state = LayeredSsmState::new(
                    num_layers, 1, mamba_config, state_dtype, &self.device,
                );

                // Prefill
                tracing::info!(phase = "prefill_start", backend = "mamba2", prompt_tokens = prompt_tokens.len());
                let mut logits = self.model.forward_with_ssm_state(&input, &mut ssm_state)
                    .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                tracing::info!(phase = "prefill_end", backend = "mamba2");
                tracing::info!(phase = "decode_start", backend = "mamba2", max_tokens = max_tokens);
                for i in 0..max_tokens {
                    let cur_logits = &logits;
                    let (token_gpu, miro_id) = self.sample_token_dispatch(cur_logits, &token_history, gen_config, &mut mirostat)?;
                    let event = token_gpu.record_event()
                        .map_err(|e| anyhow!("Record event failed: {}", e))?;

                    let next_input = token_gpu.reshape(&[1, 1])?;
                    let next_logits = self.model.forward_with_ssm_state(&next_input, &mut ssm_state)
                        .map_err(|e| anyhow!("Forward pass failed: {}", e))?;

                    let next_token = miro_id.map_or_else(
                        || Self::read_token_id(&token_gpu, event),
                        Ok,
                    )?;
                    token_history.push(next_token);

                    let is_last = i + 1 == max_tokens;
                    let finish = if self.tokenizer.is_eos(next_token) {
                        Some(FinishReason::Eos)
                    } else if is_last {
                        Some(FinishReason::Length)
                    } else {
                        None
                    };

                    yield Ok(self.make_token(next_token, cur_logits, gen_config, finish)?);
                    if finish == Some(FinishReason::Eos) { break; }
                    logits = next_logits;
                }
                tracing::info!(phase = "decode_end", backend = "mamba2");
            } else if self.config.inference.paged_attention {
                // ── Llama path: Paged KV cache ──
                let num_layers = self.model.num_layers();
                let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
                let head_dim = self.model.head_dim().unwrap_or(64);
                let block_size = self.config.inference.block_size;
                let kv_dtype = parse_dtype(self.config.dtype())?;

                let total_tokens = prompt_tokens.len() + max_tokens;
                let num_blocks = if self.config.inference.num_blocks > 0 {
                    self.config.inference.num_blocks
                } else {
                    BlockTable::blocks_needed(total_tokens, block_size) + 4
                };

                let allocator = CpuBlockAllocator::new(num_blocks, block_size);

                let mut paged_cache = LayeredPagedKvCache::new(
                    num_layers, num_blocks, block_size, num_kv_heads, head_dim, kv_dtype, &self.device,
                );

                paged_cache.allocate_blocks(prompt_tokens.len(), &allocator)
                    .map_err(|e| anyhow!("Failed to allocate blocks for prefill: {}", e))?;

                let slot_mapping_vec = paged_cache.compute_slot_mapping(0, prompt_tokens.len())
                    .map_err(|e| anyhow!("Failed to compute slot mapping: {}", e))?;
                let slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[prompt_tokens.len()], &self.device);

                let bt_vec = paged_cache.block_table_device_format(0);
                let max_num_blocks = bt_vec.len();
                let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, max_num_blocks], &self.device);

                let seq_len_k = prompt_tokens.len();
                paged_cache.set_seq_len(seq_len_k);

                tracing::info!(phase = "prefill_start", backend = "paged", prompt_tokens = prompt_tokens.len(), num_blocks = num_blocks);
                let t0 = std::time::Instant::now();
                let mut logits = self.model.forward_with_paged_kv_cache(
                    &input, &paged_cache, &slot_mapping, &block_table_tensor, seq_len_k, 0,
                ).map_err(|e| anyhow!("Paged prefill failed: {}", e))?;
                tracing::info!("Paged prefill: {:?} (seq_len={}, blocks={})", t0.elapsed(), prompt_tokens.len(), num_blocks);
                tracing::info!(phase = "prefill_end", backend = "paged");
                tracing::info!(phase = "decode_start", backend = "paged", max_tokens = max_tokens);
                for i in 0..max_tokens {
                    let t1 = std::time::Instant::now();

                    let cur_logits = &logits;
                    let (token_gpu, miro_id) = self.sample_token_dispatch(cur_logits, &token_history, gen_config, &mut mirostat)?;
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
                    let next_token = miro_id.map_or_else(
                        || Self::read_token_id(&token_gpu, event),
                        Ok,
                    )?;
                    token_history.push(next_token);
                    tracing::info!("Paged token {}: fwd_launch={:?} sync={:?} total={:?}", i+1, fwd_time, t2.elapsed(), t1.elapsed());

                    let is_last = i + 1 == max_tokens;
                    let finish = if self.tokenizer.is_eos(next_token) {
                        Some(FinishReason::Eos)
                    } else if is_last {
                        Some(FinishReason::Length)
                    } else {
                        None
                    };

                    yield Ok(self.make_token(next_token, cur_logits, gen_config, finish)?);
                    if finish == Some(FinishReason::Eos) { break; }
                    logits = next_logits;
                }
                tracing::info!(phase = "decode_end", backend = "paged");
            } else {
                // ── Llama path: contiguous KV cache ──
                let num_layers = self.model.num_layers();
                let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
                let head_dim = self.model.head_dim().unwrap_or(64);
                let initial_capacity = (prompt_tokens.len() + max_tokens).min(max_seq_len);

                let kv_dtype = parse_dtype(self.config.dtype())?;

                let mut kv_cache = LayeredKvCache::new_positional(
                    num_layers, 1, num_kv_heads, initial_capacity, max_seq_len,
                    head_dim, kv_dtype, &self.device,
                ).map_err(|e| anyhow!("Failed to create KV cache: {}", e))?;

                tracing::info!(phase = "prefill_start", backend = "contiguous", prompt_tokens = prompt_tokens.len());
                let t0 = std::time::Instant::now();
                let mut logits = self.model.forward_with_kv_cache(&input, &mut kv_cache, 0)
                    .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                tracing::info!("Prefill: {:?} (seq_len={}, kv={})", t0.elapsed(), prompt_tokens.len(), kv_cache.seq_len());
                tracing::info!(phase = "prefill_end", backend = "contiguous");
                tracing::info!(phase = "decode_start", backend = "contiguous", max_tokens = max_tokens);
                for i in 0..max_tokens {
                    let t1 = std::time::Instant::now();

                    let cur_logits = &logits;
                    let (token_gpu, miro_id) = self.sample_token_dispatch(cur_logits, &token_history, gen_config, &mut mirostat)?;
                    let event = token_gpu.record_event()
                        .map_err(|e| anyhow!("Record event failed: {}", e))?;

                    let next_input = token_gpu.reshape(&[1, 1])?;
                    let position = kv_cache.seq_len();
                    let next_logits = self.model.forward_with_kv_cache(&next_input, &mut kv_cache, position)
                        .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                    let fwd_time = t1.elapsed();

                    let t2 = std::time::Instant::now();
                    let next_token = miro_id.map_or_else(
                        || Self::read_token_id(&token_gpu, event),
                        Ok,
                    )?;
                    token_history.push(next_token);
                    tracing::trace!("Token {}: id={} fwd_launch={:?} sync={:?} total={:?}", i+1, next_token, fwd_time, t2.elapsed(), t1.elapsed());

                    let is_last = i + 1 == max_tokens;
                    let finish = if self.tokenizer.is_eos(next_token) {
                        Some(FinishReason::Eos)
                    } else if is_last {
                        Some(FinishReason::Length)
                    } else {
                        None
                    };

                    yield Ok(self.make_token(next_token, cur_logits, gen_config, finish)?);
                    if finish == Some(FinishReason::Eos) { break; }
                    logits = next_logits;
                }
                tracing::info!(phase = "decode_end", backend = "contiguous");
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
        let max_attempts = if gen_config.json_mode { 3 } else { 1 };
        for attempt in 0..max_attempts {
            let result = self.generate_text_once(prompt, gen_config).await?;
            if !gen_config.json_mode || is_valid_json(&result.text) {
                return Ok(result);
            }
            tracing::warn!(
                "JSON mode: attempt {}/{} produced invalid JSON, retrying",
                attempt + 1,
                max_attempts
            );
        }
        self.generate_text_once(prompt, gen_config).await
    }

    /// Single generation attempt (used by generate_text for JSON retry)
    async fn generate_text_once(
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
        let mut finish_reason = FinishReason::Length;
        let mut stream = std::pin::pin!(self.generate(prompt, gen_config));
        let prefill_start = std::time::Instant::now();
        let mut prompt_eval_duration_ms = 0u64;
        let mut token_logprobs: Vec<GeneratedToken> = Vec::new();

        while let Some(token_result) = stream.next().await {
            let token = token_result?;
            completion_tokens += 1;

            if completion_tokens == 1 {
                prompt_eval_duration_ms = prefill_start.elapsed().as_millis() as u64;
            }

            result.push_str(&token.text);

            if let Some(reason) = token.finish_reason {
                finish_reason = reason;
            }

            if gen_config.logprobs {
                token_logprobs.push(token);
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
                        token_logprobs: if gen_config.logprobs {
                            Some(token_logprobs)
                        } else {
                            None
                        },
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
            token_logprobs: if gen_config.logprobs {
                Some(token_logprobs)
            } else {
                None
            },
        })
    }

    /// Create input tensor from token IDs
    pub(crate) fn create_input_tensor(&self, tokens: &[u32]) -> Result<Tensor<R>> {
        let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        Ok(Tensor::from_slice(
            &tokens_i64,
            &[1, tokens.len()],
            &self.device,
        ))
    }

    /// Greedy argmax on GPU — returns the token as a GPU tensor [1] i64.
    /// No CPU sync happens here; the result stays on device.
    pub(crate) fn argmax_on_gpu(&self, logits: &Tensor<R>) -> Result<Tensor<R>> {
        let seq_len = logits.dim(1).map_err(|e| anyhow!("{}", e))?;
        let narrowed = logits
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| anyhow!("{}", e))?;
        let squeezed = narrowed.squeeze(Some(1)).squeeze(Some(0));
        squeezed.argmax(0, false).map_err(|e| anyhow!("{}", e))
    }

    /// Read a scalar i64 GPU tensor to CPU using the pipelined copy stream.
    pub(crate) fn read_token_id(token_gpu: &Tensor<R>, event: u64) -> Result<u32> {
        let v: Vec<i64> = token_gpu
            .to_vec_pipelined(event)
            .map_err(|e| anyhow!("Pipelined D2H copy failed: {}", e))?;
        Ok(v[0] as u32)
    }
}
