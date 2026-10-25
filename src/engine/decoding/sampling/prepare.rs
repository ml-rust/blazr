//! The host-side half of fused sampling: the penalties, grammar mask, logit
//! bias and penalty-window tensors applied before the `SamplingOps` call.

use anyhow::{anyhow, Result};

use boostr::inference::sampling::{
    apply_dry_penalty, apply_typical_filter, penalty_window, DryParams, DynaTemp, LogitBias,
};
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, DeviceGrammarDfa, GrammarDfaOps, NormalizationOps,
    Runtime, SamplingOps, ScalarOps, Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;

use crate::engine::executor::Executor;
use crate::engine::host_read::last_logits_row;

/// Logits ready to sample, and what `logits_to_token[_into]` samples them
/// with: the penalty-window tensors, how many unique tokens they carry, and
/// the temperature to sample at.
pub(super) struct PreparedLogitsToToken<R: Runtime> {
    pub(super) logits: Tensor<R>,
    pub(super) ids_tensor: Tensor<R>,
    pub(super) cnts_tensor: Tensor<R>,
    pub(super) num_unique: usize,
    pub(super) temperature: f32,
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
    /// Everything [`Self::logits_to_token_on_device_with_grammar`] and its
    /// `_into` sibling need before they call into [`SamplingOps`]: logits
    /// with the CPU-side penalties, the grammar mask and the logit bias
    /// already applied, the penalty-window tensors, and the temperature to
    /// sample at.
    ///
    /// The DRY/typical rewrite, the grammar mask, the logit bias and the
    /// penalty-window tensors allocate. This function shares those steps
    /// between the allocating and the `_into` callers.
    pub(super) fn prepare_logits_to_token(
        &self,
        logits: &Tensor<R>,
        recent_tokens: &[u32],
        gen_config: &GenerationConfig,
        dry: Option<&DryParams>,
        gpu_grammar: Option<&DeviceGrammarDfa<R>>,
    ) -> Result<PreparedLogitsToToken<R>> {
        // Apply CPU-side penalties (DRY, typical) if enabled — these still need CPU
        let needs_cpu = dry.is_some() || gen_config.typical_p > 0.0;
        let logits = if needs_cpu {
            // Only the sampled position matters, so only its row copies.
            let mut last_logits = last_logits_row(logits)?;
            let vocab_size = last_logits.len();

            if let Some(dry) = dry {
                apply_dry_penalty(&mut last_logits, recent_tokens, dry);
            }
            if gen_config.typical_p > 0.0 {
                apply_typical_filter(&mut last_logits, gen_config.typical_p);
            }

            Tensor::from_slice(&last_logits, &[1, 1, vocab_size], logits.device())?
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
        let bias = LogitBias::new(&gen_config.logit_bias);
        let logits = if !bias.is_empty() {
            bias.apply(&logits, self.vocab_size())
                .map_err(|e| anyhow!("logit_bias add failed: {e}"))?
        } else {
            logits
        };

        let (ids, cnts) = penalty_window(recent_tokens, gen_config.repeat_last_n);
        let num_unique = ids.len();
        let ids_tensor = Tensor::from_slice(&ids, &[ids.len()], self.device())?;
        let cnts_tensor = Tensor::from_slice(&cnts, &[cnts.len()], self.device())?;

        let temperature = if gen_config.is_greedy() {
            0.0
        } else if gen_config.dynatemp_range > 0.0 {
            DynaTemp::from(gen_config).temperature(&last_logits_row(&logits)?)
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
}
