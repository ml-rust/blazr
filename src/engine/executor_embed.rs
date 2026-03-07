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
    /// Get token-level embeddings for the given token IDs.
    ///
    /// Returns a flat f32 vector of shape `[num_tokens * hidden_size]`.
    /// The caller is responsible for pooling (mean, cls, last, none).
    pub async fn get_embeddings(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let device = &self.device;

        // Create input tensor [1, seq_len]
        let ids_i64: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();
        let seq_len = ids_i64.len();
        let input = Tensor::<R>::from_slice(&ids_i64, &[1, seq_len], device);

        // Run embedding layer to get token embeddings [1, seq_len, hidden_size]
        // TODO: For full contextualized embeddings, add a `forward_hidden` method
        // to LoadedModel that runs all layers + final norm (before lm_head).
        let hidden_var = self.model.forward_embed(&input)?;
        let hidden_tensor = hidden_var.tensor();

        // Extract to CPU as f32
        let data: Vec<f32> = hidden_tensor.to_vec();

        Ok(data)
    }
}
