//! `generate()` method for the inference executor.
//!
//! This is a separate `impl` block for `Executor<R>` — the struct and all other
//! methods remain in `executor.rs`.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::inference::speculative::SpeculativeClient;
use boostr::inference::LayeredSsmState;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::tokenizer::token_piece;

use super::types::{FinishReason, GeneratedToken, PrefillCancel};

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
        + boostr::quant::traits::DequantOps<R>
        + SpeculativeClient<R>,
{
    /// Generate from already-encoded prompt ids.
    ///
    /// The ids are the caller's — the handler that validated them against its
    /// special-token policy — so nothing here re-tokenizes text and no second
    /// encode can disagree with the one that was checked.
    ///
    /// With `inference.speculative.enabled`, an eligible request decodes
    /// speculatively. Every other request takes the plain loops below.
    pub fn generate<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        gen_config: &'a GenerationConfig,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        self.generate_cancellable(prompt_tokens, gen_config, PrefillCancel::never())
    }

    /// [`Self::generate`], with a prefill that stops once `cancel` trips.
    ///
    /// Every path prefills in chunks of `inference.prefill_chunk_size`
    /// tokens and reads `cancel` before each chunk. A cancelled prefill ends
    /// the stream with no item and drops its KV cache. At a chunk size of 0
    /// `cancel` is read once, before the one prefill forward.
    pub fn generate_cancellable<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        gen_config: &'a GenerationConfig,
        cancel: PrefillCancel<'a>,
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

            // ── Speculative path: one sequence, drafts checked per forward ──
            if let Some(plan) = self.speculative_route(gen_config, prompt_tokens, max_tokens) {
                let inner = self.generate_speculative(prompt_tokens, gen_config, plan, cancel);
                for await item in inner {
                    yield item;
                }
                return;
            }

            let input = self.create_input_tensor(prompt_tokens)?;
            let chunk = self.config.inference.prefill_chunk_size;

            let mut sampler = self.sampler_state(prompt_tokens, gen_config)?;

            let model = self.model()?;

            if model.needs_gdn_state() {
                // ── qwen35 path: KV cache (attention layers) + GDN state ──
                // Checked first: needs_kv_cache() is also true here, and the
                // KV branches below size the cache by num_layers().
                let setup = DecodeSetup {
                    input,
                    max_tokens,
                    sampler,
                    plan: None,
                };
                let inner = self.generate_qwen35(prompt_tokens, gen_config, setup, cancel);
                for await item in inner {
                    yield item;
                }
            } else if model.needs_ssm_state() {
                // ── Mamba2 path: SSM state instead of KV cache ──
                let mamba_config = model.mamba_config()
                    .ok_or_else(|| anyhow!("Mamba2 model missing mamba config"))?;
                let num_layers = model.num_layers();

                let state_dtype = self.config.compute_dtype()?;

                let mut ssm_state = LayeredSsmState::new(
                    num_layers, 1, mamba_config, state_dtype, &self.device,
                )?;

                // Prefill
                tracing::info!(phase = "prefill_start", backend = "mamba2", prompt_tokens = prompt_tokens.len());
                let prefilled = cancel.run(0..prompt_tokens.len(), chunk, |r| {
                    let piece = self.prompt_piece(&input, prompt_tokens, r)?;
                    model.forward_with_ssm_state(&piece, &mut ssm_state)
                        .map_err(|e| anyhow!("Forward pass failed: {}", e))
                })?;
                let Some(mut logits) = prefilled else { return; };
                tracing::info!(phase = "prefill_end", backend = "mamba2");
                tracing::info!(phase = "decode_start", backend = "mamba2", max_tokens = max_tokens);
                for i in 0..max_tokens {
                    let cur_logits = &logits;
                    let pending = self.sample_begin(cur_logits, &mut sampler, gen_config, None)?;

                    let next_input = pending.tensor.reshape(&[1, 1])?;
                    let next_logits = model.forward_with_ssm_state(&next_input, &mut ssm_state)
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
                        prompt_tokens: prompt_tokens.to_vec(),
                    });
                }

                // The prefill skips the cached prefix. The last prompt token
                // always runs, so the prefill yields the first logits.
                let prefill_start = cached_token_count.min(prompt_tokens.len() - 1);
                let prefill_tokens = &prompt_tokens[prefill_start..];
                let seq_len_k = prompt_tokens.len();

                let bt_vec = block_ids(lease.blocks());
                let block_table_tensor = Tensor::from_slice(&bt_vec, &[1, bt_vec.len()], &self.device)?;

                tracing::info!(
                    phase = "prefill_start", backend = "paged",
                    prompt_tokens = prompt_tokens.len(),
                    cached_tokens = prefill_start,
                    prefill_tokens = prefill_tokens.len(),
                    num_blocks = bt_vec.len(),
                );
                let t0 = std::time::Instant::now();
                let prefilled = cancel.run(prefill_start..seq_len_k, chunk, |r| {
                    let slots = block_slots(lease.blocks(), block_size, r.clone());
                    let slot_mapping = Tensor::from_slice(&slots, &[r.len()], &self.device)?;
                    let piece = self.prompt_piece(&input, prompt_tokens, r.clone())?;
                    model.forward_with_paged_kv_cache(
                        &piece, paged_cache, &slot_mapping, &block_table_tensor,
                        r.end, r.start, None, None,
                    ).map_err(|e| anyhow!("Paged prefill of positions {}..{} failed: {}", r.start, r.end, e))
                })?;
                let Some(mut logits) = prefilled else { return; };
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
                    let next_logits = model.forward_with_paged_kv_cache(
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
                let (mut kv_cache, fresh_cache) =
                    self.contiguous_kv_cache(prompt_tokens.len(), max_tokens)?;
                // A tensor-parallel group runs one sequence at a time.
                let _tp_guard = self.tp_generation_guard().await;

                tracing::info!(phase = "prefill_start", backend = "contiguous", prompt_tokens = prompt_tokens.len());
                let t0 = std::time::Instant::now();
                let mut fresh = Some(fresh_cache);
                let prefilled = cancel.run(0..prompt_tokens.len(), chunk, |r| {
                    let piece = self.prompt_piece(&input, prompt_tokens, r.clone())?;
                    self.forward_contiguous(&piece, &mut kv_cache, r.start, fresh.take())
                })?;
                let Some(mut logits) = prefilled else { return; };
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

    /// The `[1, n]` input of prompt positions `range`: `input` itself when
    /// the range covers the whole prompt.
    ///
    /// # Errors
    ///
    /// An error when the input tensor cannot be built.
    pub(crate) fn prompt_piece(
        &self,
        input: &Tensor<R>,
        prompt_tokens: &[u32],
        range: std::ops::Range<usize>,
    ) -> Result<Tensor<R>> {
        if range == (0..prompt_tokens.len()) {
            return Ok(input.clone());
        }
        let piece = prompt_tokens.get(range.clone()).ok_or_else(|| {
            anyhow!(
                "prompt positions {range:?} lie outside the {}-token prompt",
                prompt_tokens.len()
            )
        })?;
        self.create_input_tensor(piece)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use boostr::CpuRuntime;
    use futures::StreamExt;

    use super::*;
    use crate::engine::requests::batch_test_rig::{greedy, POOL, SEED};
    use crate::engine::test_support::{free_pool_blocks, seeded_executor};

    /// Nine prompt tokens: five chunks of two.
    const PROMPT: [u32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

    /// Generate from `prompt` with a probe that trips on its read `trip`.
    /// Returns the items the stream yielded and the reads the probe took.
    pub(in crate::engine) async fn stopped_at(
        exec: &Executor<CpuRuntime>,
        prompt: &[u32],
        trip: usize,
    ) -> (usize, usize) {
        let reads = AtomicUsize::new(0);
        let probe = || reads.fetch_add(1, Ordering::SeqCst) + 1 >= trip;
        let cfg = greedy(4);
        let cancel = PrefillCancel::when(&probe);
        let items: Vec<_> = exec
            .generate_cancellable(prompt, &cfg, cancel)
            .collect()
            .await;
        (items.len(), reads.load(Ordering::SeqCst))
    }

    /// The greedy ids of `prompt`, with no cancel.
    async fn ids(exec: &Executor<CpuRuntime>, prompt: &[u32]) -> Result<Vec<u32>> {
        let cfg = greedy(4);
        let items: Vec<_> = exec.generate(prompt, &cfg).collect().await;
        items.into_iter().map(|t| t.map(|t| t.token_id)).collect()
    }

    #[tokio::test]
    async fn a_contiguous_prefill_stops_at_the_chunk_after_the_cancel() -> Result<()> {
        let whole = seeded_executor(SEED, None)?;
        let mut exec = seeded_executor(SEED, None)?;
        exec.config.inference.prefill_chunk_size = 2;
        assert_eq!(ids(&exec, &PROMPT).await?, ids(&whole, &PROMPT).await?);

        assert_eq!(
            stopped_at(&exec, &PROMPT, 2).await,
            (0, 2),
            "one of five chunks ran"
        );
        assert_eq!(
            stopped_at(&exec, &PROMPT, 4).await,
            (0, 4),
            "three chunks ran"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_paged_prefill_stops_and_returns_its_blocks() -> Result<()> {
        let whole = seeded_executor(SEED, Some(POOL))?;
        let mut exec = seeded_executor(SEED, Some(POOL))?;
        exec.config.inference.prefill_chunk_size = 2;
        assert_eq!(ids(&exec, &PROMPT).await?, ids(&whole, &PROMPT).await?);

        assert_eq!(
            stopped_at(&exec, &PROMPT, 2).await,
            (0, 2),
            "one of five chunks ran"
        );
        assert_eq!(
            free_pool_blocks(&exec)?,
            POOL.blocks,
            "the lease returned its blocks"
        );
        Ok(())
    }

    /// At chunk size 0 the probe is read once, before the one forward.
    #[tokio::test]
    async fn an_unchunked_prefill_reads_the_cancel_once_before_its_forward() -> Result<()> {
        let mut exec = seeded_executor(SEED, Some(POOL))?;
        exec.config.inference.prefill_chunk_size = 0;
        assert_eq!(stopped_at(&exec, &PROMPT, 1).await, (0, 1));
        assert_eq!(free_pool_blocks(&exec)?, POOL.blocks);
        let (items, reads) = stopped_at(&exec, &PROMPT, 2).await;
        assert_eq!(
            (items, reads),
            (4, 1),
            "an untripped probe lets all four tokens out"
        );
        Ok(())
    }
}
