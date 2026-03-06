//! Speculative decoding adapter
//!
//! Implements boostr's `SpeculativeModel` trait for blazr's model wrapper,
//! enabling speculative decoding with draft + target model pairs.

use std::sync::Arc;

use boostr::error::Result as BoostrResult;
use boostr::inference::speculative::{SpeculativeModel, TokenId};
use boostr::inference::LayeredKvCache;
use boostr::model::{LoadedModel, ModelClient};
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::parse_dtype;

/// Adapter wrapping a LoadedModel to implement SpeculativeModel
pub struct BlazrSpeculativeModel<R: Runtime<DType = DType>> {
    model: Arc<LoadedModel<R>>,
    kv_cache: Option<LayeredKvCache<R>>,
    device: R::Device,
    dtype: DType,
    position: usize,
    model_name: String,
}

impl<R: Runtime<DType = DType>> BlazrSpeculativeModel<R>
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
    pub fn new(
        model: Arc<LoadedModel<R>>,
        device: R::Device,
        dtype_str: &str,
        name: &str,
    ) -> anyhow::Result<Self> {
        let dtype = parse_dtype(dtype_str)?;
        Ok(Self {
            model,
            kv_cache: None,
            device,
            dtype,
            position: 0,
            model_name: name.to_string(),
        })
    }

    fn ensure_kv_cache(&mut self) -> anyhow::Result<()> {
        if self.kv_cache.is_none() {
            let num_layers = self.model.num_layers();
            let num_kv_heads = self.model.num_kv_heads().unwrap_or(8);
            let head_dim = self.model.head_dim().unwrap_or(64);
            let max_seq_len = 8192; // reasonable default

            let kv = LayeredKvCache::new_positional(
                num_layers,
                1,
                num_kv_heads,
                512, // initial capacity
                max_seq_len,
                head_dim,
                self.dtype,
                &self.device,
            )
            .map_err(|e| anyhow::anyhow!("Failed to create KV cache: {}", e))?;
            self.kv_cache = Some(kv);
        }
        Ok(())
    }
}

impl<R: Runtime<DType = DType>> SpeculativeModel<R> for BlazrSpeculativeModel<R>
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
    fn forward(&mut self, input_tokens: &[TokenId], position: usize) -> BoostrResult<Vec<f32>> {
        self.ensure_kv_cache()
            .map_err(|e| boostr::error::Error::InferenceError {
                reason: format!("KV cache init failed: {}", e),
            })?;

        let tokens_i64: Vec<i64> = input_tokens.iter().map(|&t| t as i64).collect();
        let input = Tensor::from_slice(&tokens_i64, &[1, input_tokens.len()], &self.device);

        let kv = self
            .kv_cache
            .as_mut()
            .ok_or_else(|| boostr::error::Error::InferenceError {
                reason: "KV cache not initialized after ensure_kv_cache".into(),
            })?;
        let logits = self.model.forward_with_kv_cache(&input, kv, position)?;

        self.position = position + input_tokens.len();

        // Extract last position logits
        let shape = logits.shape();
        let seq_len = shape[1];
        let vocab_size = shape[2];
        let all: Vec<f32> = logits.to_vec();
        let offset = (seq_len - 1) * vocab_size;
        Ok(all[offset..offset + vocab_size].to_vec())
    }

    fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    fn reset(&mut self) -> BoostrResult<()> {
        if let Some(ref mut kv) = self.kv_cache {
            kv.reset();
        }
        self.position = 0;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.model_name
    }
}
