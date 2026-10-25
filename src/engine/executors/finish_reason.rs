//! The end-of-generation policy shared by every decode loop.

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, GrammarDfaOps, NormalizationOps, Runtime,
    SamplingOps, ScalarOps, TypeConversionOps, UnaryOps,
};

use super::executor::Executor;
use super::types::FinishReason;

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
    /// Whether `next_token` ends generation, and why.
    ///
    /// Every decode loop and every lane uses it, so the eos and length
    /// policy cannot drift between them.
    pub(in crate::engine) fn decode_finish_reason(
        &self,
        next_token: u32,
        step: usize,
        max_tokens: usize,
    ) -> Option<FinishReason> {
        if self.tokenizer().is_eos(next_token) {
            Some(FinishReason::Eos)
        } else if step + 1 == max_tokens {
            Some(FinishReason::Length)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{anyhow, Result};

    use crate::engine::test_support::{full_vocab_executor, tiny_executor};

    #[test]
    fn eos_ids_are_the_finish_ids_inside_the_model_vocab() -> Result<()> {
        let tiny = tiny_executor()?;
        assert!(tiny.eos_ids().is_empty(), "the EOS id is past 32 ids");
        let full = full_vocab_executor(None)?;
        let eos = full
            .tokenizer()
            .eos_token_id()
            .ok_or_else(|| anyhow!("llama3 states no end-of-sequence token"))?;
        assert_eq!(full.eos_ids(), vec![eos]);
        for id in full.eos_ids() {
            assert_eq!(full.decode_finish_reason(id, 0, 8), Some(FinishReason::Eos));
        }
        Ok(())
    }

    #[test]
    fn finish_reason_is_eos_then_length_then_none() -> Result<()> {
        let exec = tiny_executor()?;
        let eos = exec
            .tokenizer()
            .eos_token_id()
            .ok_or_else(|| anyhow!("llama3 states no end-of-sequence token"))?;
        assert_eq!(
            exec.decode_finish_reason(eos, 0, 4),
            Some(FinishReason::Eos)
        );
        assert_eq!(
            exec.decode_finish_reason(5, 3, 4),
            Some(FinishReason::Length)
        );
        assert_eq!(exec.decode_finish_reason(5, 1, 4), None);
        Ok(())
    }
}
