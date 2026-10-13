//! Fused logits-to-token sampling: the CPU-side penalties, grammar mask and
//! logit bias applied before the `SamplingOps` kernel call.

use anyhow::{anyhow, Result};

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
#[cfg(feature = "cuda")]
use boostr::SamplingScratch;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, DeviceGrammarDfa, GrammarDfaOps, NormalizationOps,
    Runtime, SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;

use crate::engine::executor::Executor;

/// Logits ready to sample, and what `logits_to_token[_into]` samples them
/// with: the penalty-window tensors, how many unique tokens they carry, and
/// the temperature to sample at.
struct PreparedLogitsToToken<R: Runtime> {
    logits: Tensor<R>,
    ids_tensor: Tensor<R>,
    cnts_tensor: Tensor<R>,
    num_unique: usize,
    temperature: f32,
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
        + GrammarDfaOps<R>
        + ModelClient<R>
        + boostr::quant::traits::DequantOps<R>,
{
    /// Fused logits-to-token: narrow + cast + penalties + argmax/sample in a single kernel launch.
    ///
    /// Returns `[1]` I64 tensor on device. All decode paths (greedy, greedy+penalties,
    /// non-greedy) return the same type, enabling uniform pipelined decode.
    pub(crate) fn logits_to_token_on_device(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<Tensor<R>> {
        self.logits_to_token_on_device_with_grammar(logits, recent_tokens, gen_config, None)
    }

    /// Like `logits_to_token_on_device` but with optional device-side grammar DFA masking.
    pub(crate) fn logits_to_token_on_device_with_grammar(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
        gpu_grammar: Option<&DeviceGrammarDfa<R>>,
    ) -> Result<Tensor<R>> {
        let ready = self.prepare_logits_to_token(logits, recent_tokens, gen_config, gpu_grammar)?;
        let client = R::default_client(self.device());
        client
            .logits_to_token(
                &ready.logits,
                &ready.ids_tensor,
                &ready.cnts_tensor,
                ready.num_unique,
                gen_config.repeat_penalty,
                gen_config.frequency_penalty,
                gen_config.presence_penalty,
                ready.temperature,
                gen_config.top_k,
                gen_config.top_p,
                gen_config.min_p,
                gen_config.seed,
            )
            .map_err(|e| anyhow!("logits_to_token failed: {}", e))
    }

    /// Like [`Self::logits_to_token_on_device_with_grammar`], but samples
    /// through [`SamplingOps::logits_to_token_into`], writing the token into
    /// `scratch.output` instead of allocating.
    #[cfg(feature = "cuda")]
    pub(crate) fn logits_to_token_on_device_with_grammar_into(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
        gpu_grammar: Option<&DeviceGrammarDfa<R>>,
        scratch: &SamplingScratch<R>,
    ) -> Result<Tensor<R>> {
        let ready = self.prepare_logits_to_token(logits, recent_tokens, gen_config, gpu_grammar)?;
        let client = R::default_client(self.device());
        client
            .logits_to_token_into(
                &ready.logits,
                &ready.ids_tensor,
                &ready.cnts_tensor,
                ready.num_unique,
                gen_config.repeat_penalty,
                gen_config.frequency_penalty,
                gen_config.presence_penalty,
                ready.temperature,
                gen_config.top_k,
                gen_config.top_p,
                gen_config.min_p,
                gen_config.seed,
                scratch,
            )
            .map_err(|e| anyhow!("logits_to_token_into failed: {}", e))?;
        Ok(scratch.output.clone())
    }

    /// Everything [`Self::logits_to_token_on_device_with_grammar`] and its
    /// `_into` sibling need before they call into [`SamplingOps`]: logits
    /// with the CPU-side penalties, the grammar mask and the logit bias
    /// already applied, the penalty-window tensors, and the temperature to
    /// sample at.
    ///
    /// Every step in this chain (the DRY/typical rewrite, the grammar mask,
    /// the logit bias, and the penalty-window tensors) still allocates when
    /// it runs — this only shares the logic between the allocating and the
    /// `_into` callers, it does not remove those allocations.
    fn prepare_logits_to_token(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
        gpu_grammar: Option<&DeviceGrammarDfa<R>>,
    ) -> Result<PreparedLogitsToToken<R>> {
        // Apply CPU-side penalties (DRY, typical) if enabled — these still need CPU
        let needs_cpu = gen_config.dry_multiplier > 0.0 || gen_config.typical_p > 0.0;
        let logits = if needs_cpu {
            let shape = logits.shape();
            let seq_len = shape[1];
            let vocab_size = shape[2];
            let mut all_logits: Vec<f32> = logits.to_vec();
            let offset = (seq_len - 1) * vocab_size;
            let last_logits = &mut all_logits[offset..offset + vocab_size];

            if gen_config.dry_multiplier > 0.0 {
                Self::apply_dry_penalty(last_logits, recent_tokens, gen_config);
            }
            if gen_config.typical_p > 0.0 {
                Self::apply_typical_filter(last_logits, gen_config.typical_p);
            }

            Tensor::from_slice(&all_logits, shape, logits.device())?
        } else {
            logits.clone()
        };

        // Apply grammar DFA mask on device (no CPU<->GPU roundtrip)
        let logits = if let Some(grammar) = gpu_grammar {
            let client = R::default_client(self.device());
            client
                .grammar_dfa_mask_logits(&logits, grammar)
                .map_err(|e| anyhow!("grammar_dfa_mask_logits failed: {}", e))?
        } else {
            logits
        };

        // Apply logit_bias before sampling if any biases are specified
        let logits = if !gen_config.logit_bias.is_empty() {
            self.apply_logit_bias(&logits, &gen_config.logit_bias)?
        } else {
            logits
        };

        let (ids, cnts) = Self::penalty_window(recent_tokens, gen_config.repeat_last_n);
        let num_unique = ids.len();
        let ids_tensor = Tensor::from_slice(&ids, &[ids.len()], self.device())?;
        let cnts_tensor = Tensor::from_slice(&cnts, &[cnts.len()], self.device())?;

        let temperature = if gen_config.is_greedy() {
            0.0
        } else if gen_config.dynatemp_range > 0.0 {
            self.compute_dynamic_temperature(&logits, gen_config)
        } else {
            gen_config.temperature
        };

        Ok(PreparedLogitsToToken {
            logits,
            ids_tensor,
            cnts_tensor,
            num_unique,
            temperature,
        })
    }

    /// Apply per-token logit biases by creating a sparse bias tensor and adding to logits.
    pub(crate) fn apply_logit_bias(
        &self,
        logits: &Tensor<R>,
        bias_map: &std::collections::HashMap<u32, f32>,
    ) -> Result<Tensor<R>> {
        let vocab_size = self.vocab_size();
        let mut bias_vec = vec![0.0f32; vocab_size];
        for (&token_id, &bias) in bias_map {
            if (token_id as usize) < vocab_size {
                bias_vec[token_id as usize] = bias;
            }
        }
        let bias_tensor = Tensor::from_slice(&bias_vec, &[1, 1, vocab_size], self.device())?;
        logits
            .add(&bias_tensor)
            .map_err(|e| anyhow!("logit_bias add failed: {}", e))
    }
}
