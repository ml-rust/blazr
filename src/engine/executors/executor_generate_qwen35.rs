//! `qwen35` arm of [`Executor::generate`]: KV cache for the attention
//! layers plus `LayeredGdnState` for the Gated DeltaNet layers.
//!
//! The sampling, grammar, stop and streaming steps mirror the contiguous
//! KV branch in `executor_generate.rs`. A prompt with images arrives as a
//! [`Qwen35PromptPlan`] and prefills through the embeddings forward; the
//! decode loop is the same either way.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
use boostr::model::qwen35::Qwen35PromptPlan;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;

use super::executor::Executor;
use super::types::{FinishReason, GeneratedToken};
use crate::engine::grammar::GrammarDfa;
use crate::engine::sampling::MirostatState;
use crate::tokenizer::TokenTextStream;

/// Sampling state one request carries through its decode loop.
pub(in crate::engine) struct SamplerState<R: Runtime<DType = DType>> {
    /// Prompt ids, extended with each sampled id for repetition penalty.
    pub token_history: Vec<u32>,
    pub mirostat: Option<MirostatState>,
    pub grammar_dfa: Option<GrammarDfa>,
    pub device_grammar: Option<boostr::DeviceGrammarDfa<R>>,
    /// The request's text decoder, primed with the prompt.
    pub text: TokenTextStream,
}

/// Per-request state `generate` builds before it picks a decode arm.
pub(in crate::engine) struct DecodeSetup<R: Runtime<DType = DType>> {
    /// Prompt ids as a `[1, prompt_len]` tensor on the executor device.
    pub input: Tensor<R>,
    /// Decode steps, already capped to the context window.
    pub max_tokens: usize,
    pub sampler: SamplerState<R>,
    /// The prompt expanded around image rows; `None` prefills `input`.
    pub plan: Option<Qwen35PromptPlan<R>>,
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
        + boostr::GrammarDfaOps<R>
        + ModelClient<R>
        + RoPEPackedOps<R>
        + boostr::quant::traits::DequantOps<R>,
{
    /// KV cache sized for the attention layers only, and zeroed GDN state
    /// for the GDN layers. `num_layers()` counts both, so sizing the KV
    /// cache by it over-allocates on any model where GDN layers outnumber
    /// attention layers.
    pub(in crate::engine) fn qwen35_caches(
        &self,
        initial_capacity: usize,
        max_seq_len: usize,
    ) -> Result<(LayeredKvCache<R>, LayeredGdnState<R>)> {
        self.qwen35_caches_at(1, initial_capacity, max_seq_len)
    }

    /// The same caches at `batch` lanes: one KV lane and one GDN row per
    /// sequence a batched decode step serves.
    pub(in crate::engine) fn qwen35_caches_at(
        &self,
        batch: usize,
        initial_capacity: usize,
        max_seq_len: usize,
    ) -> Result<(LayeredKvCache<R>, LayeredGdnState<R>)> {
        let num_attention = self
            .model
            .num_attention_layers()
            .ok_or_else(|| anyhow!("qwen35 model missing attention layer count"))?;
        let num_gdn = self
            .model
            .num_gdn_layers()
            .ok_or_else(|| anyhow!("qwen35 model missing GDN layer count"))?;
        let gdn_config = self
            .model
            .gdn_config()
            .ok_or_else(|| anyhow!("qwen35 model missing GDN config"))?;
        let num_kv_heads = self
            .model
            .num_kv_heads()
            .ok_or_else(|| anyhow!("qwen35 model missing KV head count"))?;
        let head_dim = self
            .model
            .head_dim()
            .ok_or_else(|| anyhow!("qwen35 model missing head dim"))?;
        let dtype = self.config.kv_cache_dtype()?;

        let kv_config = LayeredKvCacheConfig {
            batch_size: batch,
            num_kv_heads,
            initial_capacity,
            max_seq_len,
            head_dim,
            dtype,
        };
        let kv_cache = LayeredKvCache::new(num_attention, &kv_config, &self.device)
            .map_err(|e| anyhow!("Failed to create KV cache: {}", e))?;
        let gdn_state = LayeredGdnState::zeros(num_gdn, gdn_config, batch, dtype, &self.device)
            .map_err(|e| anyhow!("Failed to create GDN state: {}", e))?;
        Ok((kv_cache, gdn_state))
    }

    /// The `qwen35` model itself, whether it is loaded bare or as the LLM
    /// backbone of a multimodal model.
    ///
    /// # Errors
    ///
    /// An error naming the loaded model when it is not `qwen35`.
    #[cfg(feature = "cuda")]
    pub(in crate::engine) fn qwen35_backbone(
        &self,
    ) -> Result<&boostr::model::qwen35::Qwen35Model<R>> {
        match self.model() {
            boostr::model::LoadedModel::Qwen35(m) => Ok(m),
            boostr::model::LoadedModel::Multimodal(m) => match m.llm() {
                boostr::model::LoadedModel::Qwen35(inner) => Ok(inner),
                other => Err(anyhow!(
                    "multimodal backbone {other:?} is not qwen35; load a qwen35 model to run \
                     the qwen35 decode path"
                )),
            },
            other => Err(anyhow!(
                "model {other:?} is not qwen35; load a qwen35 model to run the qwen35 decode \
                 path"
            )),
        }
    }

    /// Prefill at rope position 0: `plan` through the embeddings forward,
    /// else `input` through the token forward. Returns the prefill logits.
    pub(in crate::engine) fn qwen35_prefill(
        &self,
        input: &Tensor<R>,
        plan: Option<&Qwen35PromptPlan<R>>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
    ) -> Result<Tensor<R>> {
        match plan {
            Some(plan) => self
                .model
                .forward_qwen35_embeds(&plan.embeds, &plan.positions, kv_cache, gdn_state)
                .map_err(|e| anyhow!("qwen35 image prefill failed: {}", e)),
            None => self.qwen35_prefill_at(input, 0, kv_cache, gdn_state),
        }
    }

    /// Prefill `input` at rope position `rope_pos`, continuing whatever
    /// `kv_cache` and `gdn_state` already hold.
    ///
    /// `rope_pos` is the position of `input`'s first token, which for a
    /// text-only prompt is the number of tokens already cached. The
    /// attention layers offset their causal mask by the keys in the cache,
    /// and each GDN layer carries its conv window and delta-rule state
    /// forward, so feeding a prompt as successive slices leaves the same
    /// state as feeding all of it in one call.
    ///
    /// Returns the logits of the tokens in `input`.
    pub(in crate::engine) fn qwen35_prefill_at(
        &self,
        input: &Tensor<R>,
        rope_pos: usize,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
    ) -> Result<Tensor<R>> {
        self.model
            .forward_qwen35(input, kv_cache, gdn_state, rope_pos)
            .map_err(|e| {
                anyhow!(
                    "qwen35 prefill of {} tokens at rope position {rope_pos} failed: {e}",
                    input.shape().last().copied().unwrap_or(0)
                )
            })
    }

    /// Rope position of the first decode token after a prefill: the plan's
    /// next position, or the KV slot for a text-only prompt.
    pub(in crate::engine) fn qwen35_first_rope_pos(
        plan: Option<&Qwen35PromptPlan<R>>,
        kv_cache: &LayeredKvCache<R>,
    ) -> usize {
        plan.map_or(kv_cache.seq_len(), Qwen35PromptPlan::next_rope_pos)
    }

    /// Prefill then greedy/sampled decode through `forward_qwen35`.
    ///
    /// The rope position of a decode step is the KV slot minus the lag the
    /// prefill left: zero for text, `sum(n_i - max(nx_i, ny_i))` after
    /// images.
    pub(in crate::engine) fn generate_qwen35<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        gen_config: &'a GenerationConfig,
        setup: DecodeSetup<R>,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            let DecodeSetup {
                input,
                max_tokens,
                sampler,
                plan,
            } = setup;
            let SamplerState {
                mut token_history,
                mut mirostat,
                mut grammar_dfa,
                mut device_grammar,
                mut text,
            } = sampler;

            let prompt_rows = plan.as_ref().map_or(prompt_tokens.len(), Qwen35PromptPlan::seq_len);
            let max_seq_len = self.config.max_seq_len();
            let initial_capacity = (prompt_rows + max_tokens).min(max_seq_len);
            let (mut kv_cache, mut gdn_state) = self.qwen35_caches(initial_capacity, max_seq_len)?;

            tracing::info!(phase = "prefill_start", backend = "qwen35", prompt_tokens = prompt_tokens.len(), prompt_rows = prompt_rows);
            let t0 = std::time::Instant::now();
            let mut logits = self.qwen35_prefill(&input, plan.as_ref(), &mut kv_cache, &mut gdn_state)?;
            let rope_lag = kv_cache.seq_len() - Self::qwen35_first_rope_pos(plan.as_ref(), &kv_cache);
            drop(plan);
            tracing::info!("Prefill: {:?} (seq_len={}, kv={})", t0.elapsed(), prompt_rows, kv_cache.seq_len());
            tracing::info!(phase = "prefill_end", backend = "qwen35");
            tracing::info!(phase = "decode_start", backend = "qwen35", max_tokens = max_tokens);
            for i in 0..max_tokens {
                let t1 = std::time::Instant::now();

                let cur_logits = &logits;
                let (token_gpu, miro_id) = self.sample_token_dispatch_with_grammar(cur_logits, &token_history, gen_config, &mut mirostat, device_grammar.as_ref())?;
                let event = token_gpu.record_event()
                    .map_err(|e| anyhow!("Record event failed: {}", e))?;

                let next_input = token_gpu.reshape(&[1, 1])?;
                let rope_pos = kv_cache.seq_len() - rope_lag;
                let next_logits = self.model.forward_qwen35(&next_input, &mut kv_cache, &mut gdn_state, rope_pos)
                    .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
                let fwd_time = t1.elapsed();

                let t2 = std::time::Instant::now();
                let next_token = miro_id.map_or_else(
                    || Self::read_token_id(&token_gpu, event),
                    Ok,
                )?;
                token_history.push(next_token);
                self.advance_grammar_state(&mut grammar_dfa, &mut device_grammar, next_token);
                tracing::trace!("Token {}: id={} fwd_launch={:?} sync={:?} total={:?}", i+1, next_token, fwd_time, t2.elapsed(), t1.elapsed());

                let finish = self.decode_finish_reason(next_token, i, max_tokens);

                yield Ok(self.make_token(next_token, &mut text, cur_logits, gen_config, finish)?);
                if finish == Some(FinishReason::Eos) { break; }
                logits = next_logits;
            }
            tracing::info!(phase = "decode_end", backend = "qwen35");
        }
    }
}
