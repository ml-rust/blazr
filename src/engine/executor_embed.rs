//! Embedding extraction from loaded models.
//!
//! Uses the model's embedding layer to produce token embeddings,
//! then extracts them as a flat f32 vector.

use anyhow::Result;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

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
        + ModelClient<R>
        + boostr::quant::DequantOps<R>
        + boostr::quant::QuantMatmulOps<R>,
{
    /// Contextualized token-level hidden states for the given token IDs.
    ///
    /// Returns a flat f32 vector of shape `[num_tokens * hidden_size]` where each row
    /// is the hidden state after the full transformer stack + final norm (before
    /// `lm_head`). The caller is responsible for pooling (mean, cls, last, none) and
    /// optional L2 normalization.
    pub async fn get_embeddings(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let device = &self.device;

        let ids_i64: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();
        let seq_len = ids_i64.len();
        let input = Tensor::<R>::from_slice(&ids_i64, &[1, seq_len], device)?;

        let hidden_var = self.model.forward_hidden(&input)?;
        let hidden_tensor = hidden_var.tensor();

        let data: Vec<f32> = hidden_tensor.to_vec();

        Ok(data)
    }
}
