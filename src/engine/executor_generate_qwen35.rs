//! `qwen35` arm of [`Executor::generate`]: KV cache for the attention
//! layers plus `LayeredGdnState` for the Gated DeltaNet layers.
//!
//! Split from `executor_generate.rs` for the file size cap. The sampling,
//! grammar, stop and streaming steps mirror the contiguous KV branch there.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::{parse_dtype, GenerationConfig};

use super::executor::Executor;
use super::grammar::GrammarDfa;
use super::sampling::MirostatState;
use super::types::{FinishReason, GeneratedToken};

/// Per-request state `generate` builds before it picks a decode arm.
pub(super) struct DecodeSetup<R: Runtime<DType = DType>> {
    /// Prompt ids as a `[1, prompt_len]` tensor on the executor device.
    pub input: Tensor<R>,
    /// Decode steps, already capped to the context window.
    pub max_tokens: usize,
    /// Prompt ids, extended with each sampled id for repetition penalty.
    pub token_history: Vec<u32>,
    pub mirostat: Option<MirostatState>,
    pub grammar_dfa: Option<GrammarDfa>,
    pub device_grammar: Option<boostr::DeviceGrammarDfa<R>>,
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
        + boostr::quant::traits::DequantOps<R>,
{
    /// KV cache sized for the attention layers only, and zeroed GDN state
    /// for the GDN layers. `num_layers()` counts both, so sizing the KV cache
    /// by it allocates 4x too much on Bonsai 2.
    pub(super) fn qwen35_caches(
        &self,
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
        let dtype = parse_dtype(self.config.dtype())?;

        let kv_config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads,
            initial_capacity,
            max_seq_len,
            head_dim,
            dtype,
        };
        let kv_cache = LayeredKvCache::new(num_attention, &kv_config, &self.device)
            .map_err(|e| anyhow!("Failed to create KV cache: {}", e))?;
        let gdn_state = LayeredGdnState::zeros(num_gdn, gdn_config, 1, dtype, &self.device)
            .map_err(|e| anyhow!("Failed to create GDN state: {}", e))?;
        Ok((kv_cache, gdn_state))
    }

    /// Prefill then greedy/sampled decode through `forward_qwen35`.
    /// `position` is `kv_cache.seq_len()` on every decode step.
    pub(super) fn generate_qwen35<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        gen_config: &'a GenerationConfig,
        setup: DecodeSetup<R>,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            let DecodeSetup {
                input,
                max_tokens,
                mut token_history,
                mut mirostat,
                mut grammar_dfa,
                mut device_grammar,
            } = setup;

            let max_seq_len = self.config.max_seq_len();
            let initial_capacity = (prompt_tokens.len() + max_tokens).min(max_seq_len);
            let (mut kv_cache, mut gdn_state) = self.qwen35_caches(initial_capacity, max_seq_len)?;

            tracing::info!(phase = "prefill_start", backend = "qwen35", prompt_tokens = prompt_tokens.len());
            let t0 = std::time::Instant::now();
            let mut logits = self.model.forward_qwen35(&input, &mut kv_cache, &mut gdn_state, 0)
                .map_err(|e| anyhow!("Forward pass failed: {}", e))?;
            tracing::info!("Prefill: {:?} (seq_len={}, kv={})", t0.elapsed(), prompt_tokens.len(), kv_cache.seq_len());
            tracing::info!(phase = "prefill_end", backend = "qwen35");
            tracing::info!(phase = "decode_start", backend = "qwen35", max_tokens = max_tokens);
            for i in 0..max_tokens {
                let t1 = std::time::Instant::now();

                let cur_logits = &logits;
                let (token_gpu, miro_id) = self.sample_token_dispatch_with_grammar(cur_logits, &token_history, gen_config, &mut mirostat, device_grammar.as_ref())?;
                let event = token_gpu.record_event()
                    .map_err(|e| anyhow!("Record event failed: {}", e))?;

                let next_input = token_gpu.reshape(&[1, 1])?;
                let position = kv_cache.seq_len();
                let next_logits = self.model.forward_qwen35(&next_input, &mut kv_cache, &mut gdn_state, position)
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

                yield Ok(self.make_token(next_token, cur_logits, gen_config, finish)?);
                if finish == Some(FinishReason::Eos) { break; }
                logits = next_logits;
            }
            tracing::info!(phase = "decode_end", backend = "qwen35");
        }
    }
}
