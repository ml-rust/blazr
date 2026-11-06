//! Read access to an executor's parts: model, config, tokenizer, caches.

use std::fmt;
use std::sync::Arc;

use boostr::inference::memory::CpuBlockAllocator;
use boostr::model::{LoadedModel, ModelClient};
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use splintr::AnyTokenizer;

use super::executor::Executor;
use super::vision_tower::VisionTower;
use crate::config::BlazrConfig;
use crate::model::chat_template::ChatTemplate;

/// A model path ran on an executor built by [`Executor::tokenizer_only`].
///
/// Such an executor holds a tokenizer and a chat template and no weights.
/// The server answers this error with 501, since the front end cannot run
/// the request, whatever its content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoWeights;

impl fmt::Display for NoWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "this server holds no model weights: it runs the tokenizer and chat template \
             only, and remote workers run the model. Send generation requests to the \
             completion or chat routes; embeddings, images and LoRA variants are not served",
        )
    }
}

impl std::error::Error for NoWeights {}

impl<R: Runtime<DType = DType>> Executor<R> {
    /// The vision tower, when `mmproj` was configured.
    pub fn vision_tower(&self) -> Option<&Arc<VisionTower<R>>> {
        self.vision_tower.as_ref()
    }

    /// Get the chat template for this model
    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// Get model configuration
    pub fn config(&self) -> &BlazrConfig {
        &self.config
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &AnyTokenizer {
        &self.tokenizer
    }

    /// Get prefix cache statistics (if enabled)
    pub fn prefix_cache_stats(&self) -> Option<boostr::inference::PrefixCacheStats> {
        self.prefix_cache
            .as_ref()
            .and_then(|pc| pc.lock().ok().map(|c| c.stats()))
    }

    /// Get GPU prefix cache statistics (CUDA only).
    #[cfg(feature = "cuda")]
    pub fn gpu_prefix_cache_stats(&self) -> Option<boostr::inference::prefix_cache::GpuRadixStats> {
        self.gpu_prefix_cache
            .as_ref()
            .and_then(|pc| pc.lock().ok().map(|c| c.stats()))
    }

    /// Access the GPU prefix cache (CUDA only, for batch engine integration).
    #[cfg(feature = "cuda")]
    pub fn gpu_prefix_cache(
        &self,
    ) -> Option<&std::sync::Mutex<boostr::inference::prefix_cache::GpuPrefixCache>> {
        self.gpu_prefix_cache.as_ref()
    }

    /// Get the device
    pub(crate) fn device(&self) -> &R::Device {
        &self.device
    }

    /// The loaded model as an Arc, for sharing with batch engines.
    ///
    /// # Errors
    ///
    /// [`NoWeights`] on an executor built by [`Self::tokenizer_only`].
    pub(crate) fn model_arc(&self) -> Result<&Arc<LoadedModel<R>>, NoWeights> {
        self.model.as_ref().ok_or(NoWeights)
    }

    /// The loaded model: the base model, or for a LoRA variant the model
    /// with its adapter attached.
    ///
    /// # Errors
    ///
    /// [`NoWeights`] on an executor built by [`Self::tokenizer_only`].
    pub fn model(&self) -> Result<&LoadedModel<R>, NoWeights> {
        self.model.as_deref().ok_or(NoWeights)
    }

    /// Whether this executor holds model weights.
    pub fn has_weights(&self) -> bool {
        self.model.is_some()
    }

    /// Whether the model takes image or audio inputs beside text. `false`
    /// on an executor with no weights.
    pub fn is_multimodal(&self) -> bool {
        matches!(self.model.as_deref(), Some(LoadedModel::Multimodal(_)))
    }

    /// Get the shared block allocator (if paged attention enabled)
    pub(crate) fn shared_allocator(&self) -> Option<&Arc<std::sync::Mutex<CpuBlockAllocator>>> {
        self.shared_allocator.as_ref()
    }

    /// Get num_ctx
    #[cfg(feature = "cuda")]
    pub(crate) fn num_ctx(&self) -> usize {
        self.num_ctx
    }
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
        + ModelClient<R>,
{
    /// The vocabulary size: the model's, else the config's on an executor
    /// with no weights.
    pub fn vocab_size(&self) -> usize {
        self.model
            .as_ref()
            .map_or_else(|| self.config.vocab_size(), |m| m.vocab_size())
    }

    /// The ids that end generation, inside the model vocabulary.
    ///
    /// They come from the tokenizer, never from a hard-coded list.
    /// [`Self::decode_finish_reason`] ends on the same ids. The model can
    /// never emit an id past its vocabulary.
    pub(crate) fn eos_ids(&self) -> Vec<u32> {
        self.tokenizer()
            .eos_token_ids()
            .iter()
            .copied()
            .filter(|&id| usize::try_from(id).is_ok_and(|at| at < self.vocab_size()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::test_support::executor_with_eos;

    /// Every id the tokenizer stops on inside the vocabulary ends generation, so
    /// a stream, a batch and a disaggregated worker stop on the same ids.
    #[test]
    fn eos_ids_holds_every_stop_inside_the_vocabulary() {
        let executor = executor_with_eos(&[7, 9, u32::MAX]).expect("executor starts");
        assert_eq!(executor.eos_ids(), [7, 9]);
    }
}
