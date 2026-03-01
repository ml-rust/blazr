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
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, Tensor,
    TypeConversionOps, UnaryOps,
};

use crate::config::{parse_dtype, BlazrConfig, GenerationConfig};
use crate::tokenizer::{BoxedTokenizer, TokenizerTrait};

/// Inference executor
///
/// Wraps a loaded model and provides text generation capabilities.
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
        Ok(Self {
            model: Arc::new(model),
            config,
            tokenizer: Box::new(tokenizer),
            device,
            num_ctx,
        })
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

            if prompt_tokens.is_empty() {
                return;
            }

            let max_seq_len = self.config.max_seq_len();
            let max_tokens = gen_config.max_tokens.min(
                max_seq_len.saturating_sub(prompt_tokens.len())
            );

            // Create input tensor
            let input = self.create_input_tensor(&prompt_tokens)?;

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

                // Generate tokens
                for i in 0..max_tokens {
                    tracing::debug!("Generating token {} / {}", i + 1, max_tokens);

                    let next_token = self.sample_token_stochastic(&logits, gen_config)?;

                    if self.tokenizer.is_eos(next_token) {
                        tracing::debug!("Hit EOS token, stopping generation");
                        break;
                    }

                    let text = self.tokenizer.decode(&[next_token]).unwrap_or_default();

                    yield Ok(GeneratedToken {
                        token_id: next_token,
                        text,
                        logprob: None,
                    });

                    // Decode step: single token, SSM state carries context
                    let next_input = Tensor::from_slice(&[next_token as i64], &[1, 1], &self.device);
                    logits = self.model.forward_with_ssm_state(&next_input, &mut ssm_state)
                        .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
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

                // Decode loop (greedy only for now to keep it simple)
                for i in 0..max_tokens {
                    let t1 = std::time::Instant::now();

                    let next_token = if gen_config.is_greedy() {
                        let token_gpu = self.argmax_on_gpu(&logits)?;
                        let event = token_gpu.record_event()
                            .map_err(|e| anyhow!("Record event failed: {}", e))?;
                        Self::read_token_id(&token_gpu, event)?
                    } else {
                        self.sample_token_stochastic(&logits, gen_config)?
                    };

                    if self.tokenizer.is_eos(next_token) {
                        tracing::debug!("Hit EOS token, stopping generation");
                        break;
                    }

                    let text = self.tokenizer.decode(&[next_token]).unwrap_or_default();
                    yield Ok(GeneratedToken {
                        token_id: next_token,
                        text,
                        logprob: None,
                    });

                    // Allocate block if needed for new token
                    let cur_seq_len = paged_cache.seq_len();
                    paged_cache.allocate_blocks(1, &allocator)
                        .map_err(|e| anyhow!("Failed to allocate block for decode: {}", e))?;

                    // Single-token slot mapping
                    let slot_vec = paged_cache.compute_slot_mapping(cur_seq_len, 1)
                        .map_err(|e| anyhow!("Failed to compute decode slot mapping: {}", e))?;
                    let slot_mapping = Tensor::from_slice(&slot_vec, &[1], &self.device);

                    // Update block table tensor (may have grown)
                    let bt_vec = paged_cache.block_table_device_format(0);
                    let max_num_blocks = bt_vec.len();
                    let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, max_num_blocks], &self.device);

                    let new_seq_len_k = cur_seq_len + 1;
                    paged_cache.set_seq_len(new_seq_len_k);

                    let next_input = Tensor::from_slice(&[next_token as i64], &[1, 1], &self.device);
                    logits = self.model.forward_with_paged_kv_cache(
                        &next_input, &paged_cache, &slot_mapping, &block_table_tensor,
                        new_seq_len_k, cur_seq_len,
                    ).map_err(|e| anyhow!("Paged decode failed: {}", e))?;

                    tracing::info!("Paged token {}: {:?}", i + 1, t1.elapsed());
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

                if gen_config.is_greedy() {
                    // ── Pipelined greedy decode: keep token on GPU, overlap forward with CPU sync ──
                    //
                    // Pipeline:
                    //   1. argmax on GPU (async) → token_gpu
                    //   2. reshape token_gpu to [1,1] → next forward (async, ALL GPU)
                    //   3. to_vec token_gpu (SYNC) → CPU gets token_id
                    //   4. EOS check + yield on CPU (while GPU runs step 2)
                    //
                    // The GPU is never idle: the next forward launches BEFORE we sync.

                    for i in 0..max_tokens {
                        let t1 = std::time::Instant::now();

                        // Step 1: argmax stays on GPU (no sync)
                        let token_gpu = self.argmax_on_gpu(&logits)?;

                        // Step 2: record event on compute stream AFTER argmax
                        // This marks the point where token_gpu is ready to read.
                        let event = token_gpu.record_event()
                            .map_err(|e| anyhow!("Record event failed: {}", e))?;

                        // Step 3: launch next forward BEFORE syncing token to CPU
                        // Reshape [1] i64 → [1, 1] for embedding input
                        let next_input = token_gpu.reshape(&[1, 1])?;
                        let position = kv_cache.seq_len();
                        let next_logits = self.model.forward_with_kv_cache(&next_input, &mut kv_cache, position)
                            .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                        let fwd_time = t1.elapsed();

                        // Step 4: NOW sync to get token_id using copy stream + event
                        // Copy stream waits on event (fires after argmax, before next forward)
                        // Compute stream continues running next forward concurrently.
                        let t2 = std::time::Instant::now();
                        let next_token = Self::read_token_id(&token_gpu, event)?;
                        let sync_time = t2.elapsed();

                        tracing::info!("Token {}: fwd_launch={:?} sync={:?} total={:?}", i+1, fwd_time, sync_time, t1.elapsed());

                        if self.tokenizer.is_eos(next_token) {
                            tracing::debug!("Hit EOS token, stopping generation");
                            break;
                        }

                        let text = self.tokenizer.decode(&[next_token]).unwrap_or_default();
                        yield Ok(GeneratedToken {
                            token_id: next_token,
                            text,
                            logprob: None,
                        });

                        logits = next_logits;
                    }
                } else {
                    // ── Non-greedy decode: needs CPU for stochastic sampling ──
                    for i in 0..max_tokens {
                        let t1 = std::time::Instant::now();
                        let next_token = self.sample_token_stochastic(&logits, gen_config)?;
                        let sample_time = t1.elapsed();

                        if self.tokenizer.is_eos(next_token) {
                            tracing::debug!("Hit EOS token, stopping generation");
                            break;
                        }

                        let text = self.tokenizer.decode(&[next_token]).unwrap_or_default();
                        yield Ok(GeneratedToken {
                            token_id: next_token,
                            text,
                            logprob: None,
                        });

                        let t2 = std::time::Instant::now();
                        let next_input = Tensor::from_slice(&[next_token as i64], &[1, 1], &self.device);
                        let position = kv_cache.seq_len();
                        logits = self.model.forward_with_kv_cache(&next_input, &mut kv_cache, position)
                            .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                        tracing::info!("Token {}: fwd_launch={:?} sample={:?}", i+1, t2.elapsed(), sample_time);
                    }
                }
            }

            tracing::debug!("Generation loop complete");
        }
    }

    /// Generate text and return the complete result
    pub async fn generate_text(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<String> {
        use futures::StreamExt;

        let mut result = String::new();
        let mut stream = std::pin::pin!(self.generate(prompt, gen_config));

        while let Some(token_result) = stream.next().await {
            let token = token_result?;
            result.push_str(&token.text);

            // Check stop sequences
            for stop in &gen_config.stop_sequences {
                if result.ends_with(stop) {
                    result.truncate(result.len() - stop.len());
                    return Ok(result);
                }
            }
        }

        Ok(result)
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

    /// Cast tensor to F32 if needed (for sampling which requires f32 to_vec)
    fn ensure_f32(&self, t: &Tensor<R>) -> Result<Tensor<R>> {
        if t.dtype() != DType::F32 {
            let client = R::default_client(t.device());
            client
                .cast(t, DType::F32)
                .map_err(|e| anyhow!("Cast to F32 failed: {}", e))
        } else {
            Ok(t.clone())
        }
    }

    /// Greedy argmax on GPU — returns the token as a GPU tensor [1] i64.
    /// No CPU sync happens here; the result stays on device.
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

    /// Sample next token from logits (non-greedy path — requires CPU)
    fn sample_token_stochastic(
        &self,
        logits: &Tensor<R>,
        gen_config: &GenerationConfig,
    ) -> Result<u32> {
        let seq_len = logits.dim(1)?;
        let narrowed = logits.narrow(1, seq_len - 1, 1)?;
        let squeezed = narrowed.squeeze(Some(1)).squeeze(Some(0));
        let last_logits = self.ensure_f32(&squeezed.contiguous())?;

        // Temperature sampling
        let scaled = if gen_config.temperature != 1.0 {
            last_logits.scale(1.0 / gen_config.temperature as f64)?
        } else {
            last_logits
        };

        let probs = scaled.softmax(-1)?;

        let token = if gen_config.top_p < 1.0 {
            self.top_p_sample(&probs, gen_config.top_p)?
        } else if let Some(k) = gen_config.top_k {
            self.top_k_sample(&probs, k)?
        } else {
            self.multinomial_sample(&probs)?
        };

        Ok(token)
    }

    /// Multinomial sampling from probabilities
    fn multinomial_sample(&self, probs: &Tensor<R>) -> Result<u32> {
        use rand::Rng;

        let probs_vec: Vec<f32> = probs.to_vec();

        let mut rng = rand::thread_rng();
        let sample: f32 = rng.gen();

        let mut cumsum = 0.0;
        for (i, &p) in probs_vec.iter().enumerate() {
            cumsum += p;
            if cumsum > sample {
                return Ok(i as u32);
            }
        }

        Ok((probs_vec.len() - 1) as u32)
    }

    /// Top-k sampling
    fn top_k_sample(&self, probs: &Tensor<R>, k: usize) -> Result<u32> {
        use rand::Rng;

        let probs_vec: Vec<f32> = probs.to_vec();

        // Get top-k indices
        let mut indexed: Vec<(usize, f32)> =
            probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        // Renormalize
        let sum: f32 = indexed.iter().map(|(_, p)| p).sum();
        let mut rng = rand::thread_rng();
        let sample: f32 = rng.gen();

        let mut cumsum = 0.0;
        for (i, p) in &indexed {
            cumsum += p / sum;
            if cumsum > sample {
                return Ok(*i as u32);
            }
        }

        Ok(indexed.last().map(|(i, _)| *i as u32).unwrap_or(0))
    }

    /// Top-p (nucleus) sampling
    fn top_p_sample(&self, probs: &Tensor<R>, p: f32) -> Result<u32> {
        use rand::Rng;

        let probs_vec: Vec<f32> = probs.to_vec();

        // Sort by probability descending
        let mut indexed: Vec<(usize, f32)> = probs_vec
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find nucleus (cumsum > p)
        let mut cumsum = 0.0;
        let mut nucleus = Vec::new();
        for (i, prob) in indexed {
            cumsum += prob;
            nucleus.push((i, prob));
            if cumsum > p {
                break;
            }
        }

        // Renormalize and sample
        let sum: f32 = nucleus.iter().map(|(_, prob)| prob).sum();
        let mut rng = rand::thread_rng();
        let sample: f32 = rng.gen();

        let mut cumsum = 0.0;
        for (i, prob) in &nucleus {
            cumsum += prob / sum;
            if cumsum > sample {
                return Ok(*i as u32);
            }
        }

        Ok(nucleus.last().map(|(i, _)| *i as u32).unwrap_or(0))
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
}
