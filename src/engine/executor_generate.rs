//! `generate()` method for the inference executor.
//!
//! Extracted into its own file to keep executor.rs under the file size limit.
//! This is a separate `impl` block for `Executor<R>` — the struct and all other
//! methods remain in `executor.rs`.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::inference::kv_cache::LayeredPagedKvCache;
use boostr::inference::memory::{BlockTable, CpuBlockAllocator};
use boostr::inference::{LayeredKvCache, LayeredSsmState};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::{parse_dtype, GenerationConfig};
use crate::tokenizer::TokenizerTrait;

use super::sampling::MirostatState;
use super::types::{FinishReason, GeneratedToken};

use super::executor::Executor;

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
        + ModelClient<R>,
{
    pub fn generate<'a>(
        &'a self,
        prompt: &'a str,
        gen_config: &'a GenerationConfig,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            // Log active LoRA adapter if one is requested
            if let Some(ref adapter_name) = gen_config.lora_adapter {
                if self.lora_registry.get(adapter_name).is_some() {
                    tracing::debug!(adapter = %adapter_name, "LoRA adapter active for generation");
                } else {
                    tracing::warn!(
                        adapter = %adapter_name,
                        "Requested LoRA adapter not found in registry — generating without it"
                    );
                }
            }

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

            // Grammar DFA (if enabled) — compile to CPU DFA for state tracking,
            // and build a DeviceGrammarDfa for on-device logit masking
            let mut grammar_dfa: Option<super::grammar::GrammarDfa> = None;
            let mut device_grammar: Option<boostr::DeviceGrammarDfa<R>> = None;
            if let Some(ref grammar) = gen_config.grammar {
                match super::grammar::compile_grammar_to_dfa(grammar) {
                    Ok(dfa) => {
                        tracing::info!("Grammar DFA compiled: {} states", dfa.num_states());
                        // Build vocab byte representations for the device DFA
                        let vocab_size = self.vocab_size();
                        let vocab_bytes: Vec<Vec<u8>> = (0..vocab_size)
                            .map(|i| {
                                self.tokenizer()
                                    .decode(&[i as u32])
                                    .unwrap_or_default()
                                    .into_bytes()
                            })
                            .collect();
                        device_grammar = Some(dfa.to_device::<R>(&vocab_bytes, &self.device));
                        grammar_dfa = Some(dfa);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to compile grammar DFA: {}", e);
                    }
                }
            }

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
                    let (token_gpu, miro_id) = self.sample_token_dispatch_with_grammar(cur_logits, &token_history, gen_config, &mut mirostat, device_grammar.as_ref())?;
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
                    // Advance grammar DFA state
                    if let Some(ref mut dfa) = grammar_dfa {
                        let token_text = self.tokenizer().decode(&[next_token]).unwrap_or_default();
                        for byte in token_text.bytes() {
                            dfa.advance(byte);
                        }
                        // Sync device-side DFA state with CPU-side state
                        if let Some(ref mut dg) = device_grammar {
                            dg.current_state = dfa.current_state() as u32;
                        }
                    }

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

                // Use shared allocator if available, otherwise create per-request
                let total_tokens = prompt_tokens.len() + max_tokens;
                let num_blocks = if self.config.inference.num_blocks > 0 {
                    self.config.inference.num_blocks
                } else {
                    BlockTable::blocks_needed(total_tokens, block_size) + 4
                };

                let allocator = if let Some(ref shared) = self.shared_allocator {
                    let cloned = {
                        let guard = shared.lock().expect("block allocator lock poisoned");
                        guard.clone()
                    };
                    cloned
                } else {
                    CpuBlockAllocator::new(num_blocks, block_size)
                };

                let mut paged_cache = LayeredPagedKvCache::new(
                    num_layers, num_blocks, block_size, num_kv_heads, head_dim, kv_dtype, &self.device,
                );

                // Prefix cache: get block IDs with KV reuse for cached prefixes.
                // Helper returns result without holding MutexGuard across yield points.
                let (cached_token_count, prefix_cache_seq_id) =
                    Self::prefix_cache_allocate(
                        &self.prefix_cache, &prompt_tokens, block_size,
                        &mut paged_cache, &allocator,
                    )?;

                // Sync with GPU prefix cache (insert block mappings)
                #[cfg(feature = "cuda")]
                {
                    let bt = paged_cache.block_table(0);
                    Self::gpu_prefix_cache_insert(
                        &self.gpu_prefix_cache, &prompt_tokens, block_size, &bt.blocks,
                    );
                }

                // If prefix cache provided cached tokens, only prefill the uncached suffix
                let prefill_start = cached_token_count.min(prompt_tokens.len());
                let prefill_tokens = &prompt_tokens[prefill_start..];

                let slot_mapping_vec = paged_cache.compute_slot_mapping(prefill_start, prefill_tokens.len())
                    .map_err(|e| anyhow!("Failed to compute slot mapping: {}", e))?;
                let slot_mapping = Tensor::from_slice(&slot_mapping_vec, &[prefill_tokens.len()], &self.device);

                let bt_vec = paged_cache.block_table_device_format(0);
                let max_num_blocks = bt_vec.len();
                let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, max_num_blocks], &self.device);

                let seq_len_k = prompt_tokens.len();
                paged_cache.set_seq_len(seq_len_k);

                // Build input for only the uncached portion
                let prefill_input = if prefill_start > 0 && !prefill_tokens.is_empty() {
                    self.create_input_tensor(prefill_tokens)?
                } else {
                    input.clone()
                };

                tracing::info!(
                    phase = "prefill_start", backend = "paged",
                    prompt_tokens = prompt_tokens.len(),
                    cached_tokens = prefill_start,
                    prefill_tokens = prefill_tokens.len(),
                    num_blocks = num_blocks,
                );
                let t0 = std::time::Instant::now();
                let mut logits = self.model.forward_with_paged_kv_cache(
                    &prefill_input, &paged_cache, &slot_mapping, &block_table_tensor,
                    seq_len_k, prefill_start,
                ).map_err(|e| anyhow!("Paged prefill failed: {}", e))?;
                tracing::info!("Paged prefill: {:?} (seq_len={}, blocks={})", t0.elapsed(), prompt_tokens.len(), num_blocks);
                tracing::info!(phase = "prefill_end", backend = "paged");
                tracing::info!(phase = "decode_start", backend = "paged", max_tokens = max_tokens);
                for i in 0..max_tokens {
                    let t1 = std::time::Instant::now();

                    let cur_logits = &logits;
                    let (token_gpu, miro_id) = self.sample_token_dispatch_with_grammar(cur_logits, &token_history, gen_config, &mut mirostat, device_grammar.as_ref())?;
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
                    Self::moe_offload_step(&self.moe_offload, &self.expert_placements, &self.model, &self.device);
                    let fwd_time = t1.elapsed();

                    let t2 = std::time::Instant::now();
                    let next_token = miro_id.map_or_else(
                        || Self::read_token_id(&token_gpu, event),
                        Ok,
                    )?;
                    token_history.push(next_token);
                    // Advance grammar DFA state
                    if let Some(ref mut dfa) = grammar_dfa {
                        let token_text = self.tokenizer().decode(&[next_token]).unwrap_or_default();
                        for byte in token_text.bytes() {
                            dfa.advance(byte);
                        }
                        // Sync device-side DFA state with CPU-side state
                        if let Some(ref mut dg) = device_grammar {
                            dg.current_state = dfa.current_state() as u32;
                        }
                    }
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
                // Release prefix cache references and free blocks
                let bt = paged_cache.block_table(0);
                let blocks_to_free: Vec<_> = bt.blocks.clone();
                Self::prefix_cache_release(
                    &self.prefix_cache, prefix_cache_seq_id,
                    &blocks_to_free, &allocator,
                );
                #[cfg(feature = "cuda")]
                Self::gpu_prefix_cache_release(
                    &self.gpu_prefix_cache, &prompt_tokens, block_size,
                );
                tracing::debug!("Freed {} blocks back to pool", blocks_to_free.len());
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
                    let (token_gpu, miro_id) = self.sample_token_dispatch_with_grammar(cur_logits, &token_history, gen_config, &mut mirostat, device_grammar.as_ref())?;
                    let event = token_gpu.record_event()
                        .map_err(|e| anyhow!("Record event failed: {}", e))?;

                    let next_input = token_gpu.reshape(&[1, 1])?;
                    let position = kv_cache.seq_len();
                    let next_logits = self.model.forward_with_kv_cache(&next_input, &mut kv_cache, position)
                        .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                    Self::moe_offload_step(&self.moe_offload, &self.expert_placements, &self.model, &self.device);
                    let fwd_time = t1.elapsed();

                    let t2 = std::time::Instant::now();
                    let next_token = miro_id.map_or_else(
                        || Self::read_token_id(&token_gpu, event),
                        Ok,
                    )?;
                    token_history.push(next_token);
                    // Advance grammar DFA state
                    if let Some(ref mut dfa) = grammar_dfa {
                        let token_text = self.tokenizer().decode(&[next_token]).unwrap_or_default();
                        for byte in token_text.bytes() {
                            dfa.advance(byte);
                        }
                        // Sync device-side DFA state with CPU-side state
                        if let Some(ref mut dg) = device_grammar {
                            dg.current_state = dfa.current_state() as u32;
                        }
                    }
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
}
