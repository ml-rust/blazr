//! `generate()` method for the inference executor.
//!
//! This is a separate `impl` block for `Executor<R>` — the struct and all other
//! methods remain in `executor.rs`.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::inference::{LayeredKvCache, LayeredSsmState};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::tokenizer::token_piece;

use super::types::{FinishReason, GeneratedToken};

use super::executor::Executor;
use super::executor_cache::prefix_cache_allocate;
use super::executor_generate_qwen35::DecodeSetup;
use crate::engine::requests::kv_recompute::{block_ids, block_slots};

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
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    /// Generate from already-encoded prompt ids.
    ///
    /// The ids are the caller's — the handler that validated them against its
    /// special-token policy — so nothing here re-tokenizes text and no second
    /// encode can disagree with the one that was checked.
    pub fn generate<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        gen_config: &'a GenerationConfig,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            if gen_config.verbose_prompt {
                let prompt = self.tokenizer.decode(prompt_tokens).unwrap_or_default();
                eprintln!("\nprompt: '{}'", prompt);
                eprintln!("number of tokens in prompt = {}", prompt_tokens.len());
                for &tok in prompt_tokens {
                    let piece = token_piece(&self.tokenizer, tok);
                    eprintln!("{:>6} -> '{}'", tok, piece);
                }
                eprintln!();
            }

            if prompt_tokens.is_empty() {
                return;
            }

            let max_seq_len = self.config.max_seq_len();
            let max_tokens = crate::engine::token_budget(
                gen_config.max_tokens, max_seq_len, prompt_tokens.len(),
            );

            let input = self.create_input_tensor(prompt_tokens)?;

            let mut sampler = self.sampler_state(prompt_tokens, gen_config)?;

            if self.model.needs_gdn_state() {
                // ── qwen35 path: KV cache (attention layers) + GDN state ──
                // Checked first: needs_kv_cache() is also true here, and the
                // KV branches below size the cache by num_layers().
                let setup = DecodeSetup {
                    input,
                    max_tokens,
                    sampler,
                    plan: None,
                };
                let inner = self.generate_qwen35(prompt_tokens, gen_config, setup);
                for await item in inner {
                    yield item;
                }
            } else if self.model.needs_ssm_state() {
                // ── Mamba2 path: SSM state instead of KV cache ──
                let mamba_config = self.model.mamba_config()
                    .ok_or_else(|| anyhow!("Mamba2 model missing mamba config"))?;
                let num_layers = self.model.num_layers();

                let state_dtype = self.config.compute_dtype()?;

                let mut ssm_state = LayeredSsmState::new(
                    num_layers, 1, mamba_config, state_dtype, &self.device,
                )?;

                // Prefill
                tracing::info!(phase = "prefill_start", backend = "mamba2", prompt_tokens = prompt_tokens.len());
                let mut logits = self.model.forward_with_ssm_state(&input, &mut ssm_state)
                    .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                tracing::info!(phase = "prefill_end", backend = "mamba2");
                tracing::info!(phase = "decode_start", backend = "mamba2", max_tokens = max_tokens);
                for i in 0..max_tokens {
                    let cur_logits = &logits;
                    let pending = self.sample_begin(cur_logits, &mut sampler, gen_config, None)?;

                    let next_input = pending.tensor.reshape(&[1, 1])?;
                    let next_logits = self.model.forward_with_ssm_state(&next_input, &mut ssm_state)
                        .map_err(|e| anyhow!("Forward pass failed: {}", e))?;

                    let next_token = self.sample_finish(&pending, &mut sampler)?;

                    let finish = self.decode_finish_reason(next_token, i, max_tokens);

                    yield Ok(self.make_token(next_token, &mut sampler.text, cur_logits, gen_config, finish)?);
                    if finish == Some(FinishReason::Eos) { break; }
                    logits = next_logits;
                }
                tracing::info!(phase = "decode_end", backend = "mamba2");
            } else if self.config.inference.paged_attention {
                // ── Llama path: the executor's paged KV pool ──
                // Every block id of the shared allocator is a block of this pool.
                let paged_cache = self.paged_pool()?;
                let block_size = paged_cache.block_size();
                // The lease returns its blocks when the stream ends or drops.
                let mut lease = self.pool_lease()?;
                let cached_token_count = prefix_cache_allocate(&mut lease, prompt_tokens)?;

                #[cfg(feature = "cuda")]
                {
                    super::executor_cache::gpu_prefix_cache_insert(
                        &self.gpu_prefix_cache, prompt_tokens, block_size, lease.blocks(),
                    );
                    lease.hold_gpu_prefix(super::paged_pool::GpuPrefixRelease {
                        cache: &self.gpu_prefix_cache,
                        prompt_tokens,
                    });
                }

                // The prefill skips the cached prefix. The last prompt token
                // always runs, so the prefill yields the first logits.
                let prefill_start = cached_token_count.min(prompt_tokens.len() - 1);
                let prefill_tokens = &prompt_tokens[prefill_start..];
                let seq_len_k = prompt_tokens.len();

                let slot_mapping_vec = block_slots(lease.blocks(), block_size, prefill_start..seq_len_k);
                let slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[prefill_tokens.len()], &self.device)?;
                let bt_vec = block_ids(lease.blocks());
                let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], &self.device)?;

                let prefill_input = if prefill_start > 0 {
                    self.create_input_tensor(prefill_tokens)?
                } else {
                    input.clone()
                };

                tracing::info!(
                    phase = "prefill_start", backend = "paged",
                    prompt_tokens = prompt_tokens.len(),
                    cached_tokens = prefill_start,
                    prefill_tokens = prefill_tokens.len(),
                    num_blocks = bt_vec.len(),
                );
                let t0 = std::time::Instant::now();
                let mut logits = self.model.forward_with_paged_kv_cache(
                    &prefill_input, paged_cache, &slot_mapping, &block_table_tensor,
                    seq_len_k, prefill_start, None, None,
                ).map_err(|e| anyhow!("Paged prefill failed: {}", e))?;
                tracing::info!("Paged prefill: {:?} (seq_len={}, blocks={})", t0.elapsed(), seq_len_k, bt_vec.len());
                tracing::info!(phase = "prefill_end", backend = "paged");
                tracing::info!(phase = "decode_start", backend = "paged", max_tokens = max_tokens);
                let mut cur_seq_len = seq_len_k;
                for i in 0..max_tokens {
                    let t1 = std::time::Instant::now();

                    let cur_logits = &logits;
                    let pending = self.sample_begin(cur_logits, &mut sampler, gen_config, None)?;

                    let new_seq_len_k = cur_seq_len + 1;
                    lease.cover(new_seq_len_k)?;

                    let slot_vec = block_slots(lease.blocks(), block_size, cur_seq_len..new_seq_len_k);
                    let slot_mapping = Tensor::from_slice(&slot_vec, &[1], &self.device)?;

                    let bt_vec = block_ids(lease.blocks());
                    let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], &self.device)?;

                    let next_input = pending.tensor.reshape(&[1, 1])?;
                    let next_logits = self.model.forward_with_paged_kv_cache(
                        &next_input, paged_cache, &slot_mapping, &block_table_tensor,
                        new_seq_len_k, cur_seq_len, None, None,
                    ).map_err(|e| anyhow!("Paged decode failed: {}", e))?;
                    cur_seq_len = new_seq_len_k;
                    let fwd_time = t1.elapsed();

                    let t2 = std::time::Instant::now();
                    let next_token = self.sample_finish(&pending, &mut sampler)?;
                    tracing::info!("Paged token {}: fwd_launch={:?} sync={:?} total={:?}", i+1, fwd_time, t2.elapsed(), t1.elapsed());

                    let finish = self.decode_finish_reason(next_token, i, max_tokens);

                    yield Ok(self.make_token(next_token, &mut sampler.text, cur_logits, gen_config, finish)?);
                    if finish == Some(FinishReason::Eos) { break; }
                    logits = next_logits;
                }
                tracing::debug!("Freed {} blocks back to pool", lease.blocks().len());
                drop(lease);
                tracing::info!(phase = "decode_end", backend = "paged");
            } else {
                // ── Llama path: contiguous KV cache ──
                let num_layers = self.model.num_layers();
                let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
                let head_dim = self.model.head_dim().unwrap_or(64);
                let initial_capacity = (prompt_tokens.len() + max_tokens).min(max_seq_len);

                let kv_dtype = self.config.kv_cache_dtype()?;

                let mut kv_cache = LayeredKvCache::new_positional(
                    num_layers, 1, num_kv_heads, initial_capacity, max_seq_len,
                    head_dim, kv_dtype, &self.device,
                ).map_err(|e| anyhow!("Failed to create KV cache: {}", e))?;
                let fresh_cache = crate::engine::tensor_parallel::KvCacheSpec {
                    initial_capacity,
                    max_seq_len,
                    dtype: kv_dtype,
                };
                // A tensor-parallel group runs one sequence at a time.
                let _tp_guard = self.tp_generation_guard().await;

                tracing::info!(phase = "prefill_start", backend = "contiguous", prompt_tokens = prompt_tokens.len());
                let t0 = std::time::Instant::now();
                let mut logits = self.forward_contiguous(&input, &mut kv_cache, 0, Some(fresh_cache))?;
                tracing::info!("Prefill: {:?} (seq_len={}, kv={})", t0.elapsed(), prompt_tokens.len(), kv_cache.seq_len());
                tracing::info!(phase = "prefill_end", backend = "contiguous");
                tracing::info!(phase = "decode_start", backend = "contiguous", max_tokens = max_tokens);
                for i in 0..max_tokens {
                    let t1 = std::time::Instant::now();

                    let cur_logits = &logits;
                    let pending = self.sample_begin(cur_logits, &mut sampler, gen_config, None)?;

                    let next_input = pending.tensor.reshape(&[1, 1])?;
                    let position = kv_cache.seq_len();
                    let next_logits = self.forward_contiguous(&next_input, &mut kv_cache, position, None)?;
                    let fwd_time = t1.elapsed();

                    let t2 = std::time::Instant::now();
                    let next_token = self.sample_finish(&pending, &mut sampler)?;
                    tracing::trace!("Token {}: id={} fwd_launch={:?} sync={:?} total={:?}", i+1, next_token, fwd_time, t2.elapsed(), t1.elapsed());

                    let finish = self.decode_finish_reason(next_token, i, max_tokens);

                    yield Ok(self.make_token(next_token, &mut sampler.text, cur_logits, gen_config, finish)?);
                    if finish == Some(FinishReason::Eos) { break; }
                    logits = next_logits;
                }
                tracing::info!(phase = "decode_end", backend = "contiguous");
            }

            tracing::debug!("Generation loop complete");
        }
    }
}
