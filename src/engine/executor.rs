//! Inference executor
//!
//! Runs inference on loaded models using boostr's LoadedModel.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::inference::{LayeredKvCache, LayeredSsmState};
use boostr::model::LoadedModel;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, ScalarOps, Tensor,
    UnaryOps,
};

use crate::config::{BlazrConfig, GenerationConfig};
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
        + BinaryOps<R>,
{
    /// Create a new executor
    ///
    /// # Arguments
    /// * `model` - The loaded model
    /// * `config` - Model configuration
    /// * `tokenizer` - Tokenizer for encoding/decoding (boxed trait object)
    /// * `device` - Device to run on
    /// * `num_ctx` - Initial context size for KV cache (like Ollama's num_ctx).
    ///               The KV cache will grow dynamically if more tokens are needed,
    ///               up to the model's max_position_embeddings.
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
        let warmup_input = Tensor::from_slice(&[1u32], &[1, 1], &self.device);

        if self.model.needs_ssm_state() {
            // Mamba2: warmup with SSM state
            let mamba_config = self
                .model
                .mamba_config()
                .ok_or_else(|| anyhow!("Mamba2 model missing mamba config"))?;
            let num_layers = self.model.num_layers();

            let warmup_dtype = match self.config.dtype() {
                "f16" | "float16" => DType::F16,
                "bf16" | "bfloat16" => DType::BF16,
                "f32" | "float32" => DType::F32,
                _ => DType::BF16,
            };

            let mut ssm_state =
                LayeredSsmState::new(num_layers, 1, mamba_config, warmup_dtype, &self.device);

            let _ = self
                .model
                .forward_with_ssm_state(&warmup_input, &mut ssm_state)
                .map_err(|e| anyhow!("Warmup forward pass failed: {}", e))?;
        } else {
            // Llama: warmup with KV cache
            let num_layers = self.model.num_layers();
            let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
            let head_dim = self.model.head_dim().unwrap_or(64);

            let kv_dtype = match self.config.dtype() {
                "f16" | "float16" => DType::F16,
                "bf16" | "bfloat16" => DType::BF16,
                "f32" | "float32" => DType::F32,
                _ => DType::BF16,
            };

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
                .map_err(|e| anyhow!("Warmup forward pass failed: {}", e))?;
        }

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

            eprintln!("[EXEC_DEBUG] Prompt tokens: {:?} (len={})", prompt_tokens, prompt_tokens.len());

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

                let state_dtype = match self.config.dtype() {
                    "f16" | "float16" => DType::F16,
                    "bf16" | "bfloat16" => DType::BF16,
                    "f32" | "float32" => DType::F32,
                    _ => DType::BF16,
                };

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

                    let next_token = self.sample_token(&logits, gen_config)?;

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
                    let next_input = Tensor::from_slice(&[next_token], &[1, 1], &self.device);
                    logits = self.model.forward_with_ssm_state(&next_input, &mut ssm_state)
                        .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                }
            } else {
                // ── Llama path: KV cache ──
                let num_layers = self.model.num_layers();
                let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
                let head_dim = self.model.head_dim().unwrap_or(64);
                let initial_capacity = self.num_ctx;

                let kv_dtype = match self.config.dtype() {
                    "f16" | "float16" => DType::F16,
                    "bf16" | "bfloat16" => DType::BF16,
                    "f32" | "float32" => DType::F32,
                    _ => DType::BF16,
                };

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
                tracing::debug!("Starting prefill forward pass...");
                let mut logits = self.model.forward_with_kv_cache(&input, &mut kv_cache, 0)
                    .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                tracing::debug!("Prefill complete (kv_cache seq_len={})", kv_cache.seq_len());

                // Generate tokens
                for i in 0..max_tokens {
                    tracing::debug!("Generating token {} / {}", i + 1, max_tokens);

                    let next_token = self.sample_token(&logits, gen_config)?;

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

                    let next_input = Tensor::from_slice(&[next_token], &[1, 1], &self.device);
                    let position = kv_cache.seq_len();
                    logits = self.model.forward_with_kv_cache(&next_input, &mut kv_cache, position)
                        .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
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
        // Use U32 tensors for exact token indices (no f32 precision loss)
        Ok(Tensor::from_slice(tokens, &[1, tokens.len()], &self.device))
    }

    /// Sample next token from logits
    fn sample_token(&self, logits: &Tensor<R>, gen_config: &GenerationConfig) -> Result<u32> {
        // Get last position logits
        let seq_len = logits.dim(1)?;
        let narrowed = logits.narrow(1, seq_len - 1, 1)?;
        let squeezed = narrowed.squeeze(Some(1));
        let last_logits = squeezed.contiguous();

        if gen_config.is_greedy() {
            tracing::debug!("sample_token: doing argmax");

            // Debug: print top-5 logits
            if std::env::var("DEBUG_LOGITS").is_ok() {
                let logits_vec: Vec<f32> = last_logits.to_vec();
                let mut indexed: Vec<(usize, f32)> =
                    logits_vec.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!(
                    "[DEBUG_LOGITS] Top-5: {:?}",
                    &indexed[..5.min(indexed.len())]
                );
                eprintln!(
                    "[DEBUG_LOGITS] Logits range: min={:.4}, max={:.4}",
                    logits_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                    logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                );
                if logits_vec.len() > 13 {
                    let newline_logit = logits_vec[13];
                    let newline_rank = indexed.iter().position(|(id, _)| *id == 13);
                    eprintln!(
                        "[DEBUG_LOGITS] Token 13 (newline): logit={:.4}, rank={:?}",
                        newline_logit, newline_rank
                    );
                }
            }

            // Argmax for greedy decoding - dim 0 since we squeezed to 1D, keepdim=false
            let token = last_logits.argmax(0, false)?;

            let token_vec: Vec<i32> = token.to_vec();
            let token_id = token_vec[0];
            tracing::debug!("sample_token: argmax done, token = {}", token_id);
            Ok(token_id as u32)
        } else {
            // Temperature sampling
            let scaled = if gen_config.temperature != 1.0 {
                last_logits.scale(1.0 / gen_config.temperature as f64)?
            } else {
                last_logits
            };

            // Softmax
            let probs = scaled.softmax(-1)?;

            // Top-p (nucleus) sampling
            let token = if gen_config.top_p < 1.0 {
                self.top_p_sample(&probs, gen_config.top_p)?
            } else if let Some(k) = gen_config.top_k {
                self.top_k_sample(&probs, k)?
            } else {
                self.multinomial_sample(&probs)?
            };

            Ok(token)
        }
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
