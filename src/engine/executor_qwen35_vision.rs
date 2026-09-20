//! Image inputs for a `qwen35` executor: encode the images through the
//! vision tower, expand the prompt ids around their rows, then run the
//! eager `qwen35` decode arm from that plan.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::model::qwen35::{plan_positions_only, ImageGrid, Qwen35PromptPlan};
use boostr::model::{LoadedModel, ModelClient};
use boostr::ops::TensorOps;
use boostr::quant::{DequantOps, QuantMatmulOps};
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;

use super::executor::Executor;
use super::executor_generate_qwen35::DecodeSetup;
use super::generate_text::collect_generation;
use super::types::{GeneratedToken, GenerationResult};
use super::vision_tower::{grid_of, NO_TOWER_ERROR};

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
        + DequantOps<R>
        + QuantMatmulOps<R>,
{
    /// Encode `images` and expand `prompt_tokens` (one `<|image_pad|>` per
    /// image, in order) into a prefill plan at rope position 0.
    ///
    /// The layout is checked before the tower runs: a marker mismatch or a
    /// plan longer than the context window is refused without encoding.
    ///
    /// # Errors
    ///
    /// No vision tower is loaded, the model is not `qwen35`, an image does
    /// not decode, the placeholders and images differ in count or order,
    /// or the expanded prompt exceeds `max_seq_len`.
    pub(crate) fn qwen35_image_plan(
        &self,
        prompt_tokens: &[u32],
        images: &[Vec<u8>],
    ) -> Result<Qwen35PromptPlan<R>> {
        let tower = self
            .vision_tower
            .as_ref()
            .ok_or_else(|| anyhow!("{NO_TOWER_ERROR}"))?;
        let LoadedModel::Qwen35(model) = &*self.model else {
            return Err(anyhow!(
                "image input needs a qwen35 model; the loaded model is {}",
                self.model.model_type()
            ));
        };
        let markers = tower.markers();
        let prepared = tower.preprocess(images)?;
        let grids: Vec<ImageGrid> = prepared.iter().map(grid_of).collect();
        let layout = plan_positions_only(prompt_tokens, &grids, &markers, 0)
            .map_err(|e| anyhow!("image prompt layout: {e}"))?;
        let max_seq_len = self.config.max_seq_len();
        if layout.seq_len > max_seq_len {
            return Err(anyhow!(
                "image prompt of {} rows ({} ids, {} image rows) exceeds the context of {max_seq_len}",
                layout.seq_len,
                prompt_tokens.len(),
                layout.seq_len + images.len() - prompt_tokens.len(),
            ));
        }
        let client = R::default_client(&self.device);
        let embeds = tower.encode(&client, &prepared)?;
        Qwen35PromptPlan::build(&client, model, prompt_tokens, &embeds, &markers, 0)
            .map_err(|e| anyhow!("image prompt plan: {e}"))
    }

    /// Generate from prompt ids that hold one `<|image_pad|>` per entry of
    /// `images`, encoded bytes in prompt order. Empty `images` runs
    /// [`Executor::generate`] unchanged.
    pub fn generate_with_images<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        images: &'a [Vec<u8>],
        gen_config: &'a GenerationConfig,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            if images.is_empty() {
                let inner = self.generate(prompt_tokens, gen_config);
                for await item in inner {
                    yield item;
                }
                return;
            }
            if prompt_tokens.is_empty() {
                yield Err(anyhow!("image prompt has no ids: the markers alone are missing"));
                return;
            }
            let plan = self.qwen35_image_plan(prompt_tokens, images)?;
            let max_seq_len = self.config.max_seq_len();
            let max_tokens = gen_config
                .max_tokens
                .min(max_seq_len.saturating_sub(plan.seq_len()));
            let input = self.create_input_tensor(prompt_tokens)?;
            let sampler = self.sampler_state(prompt_tokens, gen_config)?;
            let setup = DecodeSetup {
                input,
                max_tokens,
                sampler,
                plan: Some(plan),
            };
            let inner = self.generate_qwen35(prompt_tokens, gen_config, setup);
            for await item in inner {
                yield item;
            }
        }
    }

    /// Non-streaming form of [`Executor::generate_with_images`], with the
    /// stop-sequence and logprob handling of `generate_text`.
    pub async fn generate_images_text(
        &self,
        prompt_tokens: &[u32],
        images: &[Vec<u8>],
        gen_config: &GenerationConfig,
    ) -> Result<GenerationResult> {
        let stream = self.generate_with_images(prompt_tokens, images, gen_config);
        collect_generation(stream, prompt_tokens.len(), gen_config).await
    }
}
