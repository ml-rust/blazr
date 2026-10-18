//! `generate_multimodal()` method for the inference executor.
//!
//! Handles multimodal (vision + audio) inference by encoding images/audio
//! through boostr's vision/audio encoders, then running the LLM backbone
//! with merged embeddings for prefill and standard autoregressive decode.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::Stream;

use boostr::autograd::Var;
use boostr::inference::LayeredKvCache;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::tokenizer::TokenTextStream;

use super::types::{FinishReason, GeneratedToken};

use super::executor::Executor;
use super::sampler_step::SamplerState;

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
    /// Generate tokens from a multimodal prompt (text + images + audio).
    ///
    /// Images are preprocessed via `boostr::model::vision::preprocess::preprocess_image`,
    /// audio segments are converted to mel spectrograms via
    /// `boostr::model::audio::mel::compute_mel_spectrogram`, then encoded through
    /// the model's vision/audio encoders. The resulting embeddings are merged with
    /// text token embeddings for the prefill step. Subsequent decode steps are
    /// text-only autoregressive generation using the KV cache.
    pub fn generate_multimodal<'a>(
        &'a self,
        prompt_tokens: &'a [u32],
        images: &'a [Vec<u8>],
        audio_segments: &'a [Vec<f32>],
        gen_config: &'a GenerationConfig,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            // Check that the model is multimodal
            let multimodal_model = match &*self.model {
                boostr::model::LoadedModel::Multimodal(m) => m,
                _ => {
                    yield Err(anyhow!(
                        "Model does not support multimodal input. \
                         Load a model with vision/audio encoders (e.g., LLaVA, Qwen-VL)."
                    ));
                    return;
                }
            };

            if multimodal_model.llm().needs_gdn_state() {
                yield Err(super::qwen35_unsupported(
                    "multimodal",
                    None,
                    Some("it needs GDN state beside the KV cache"),
                ));
                return;
            }

            let vision_config = multimodal_model.config().vision.as_ref();
            let audio_config = multimodal_model.config().audio.as_ref();

            // Check inputs: images need a vision encoder, audio needs an audio encoder
            if !images.is_empty() && vision_config.is_none() {
                yield Err(anyhow!(
                    "Request contains images but model has no vision encoder configured"
                ));
                return;
            }
            if !audio_segments.is_empty() && audio_config.is_none() {
                yield Err(anyhow!(
                    "Request contains audio but model has no audio encoder configured"
                ));
                return;
            }

            if prompt_tokens.is_empty() {
                return;
            }

            let max_seq_len = self.config.max_seq_len();
            let max_tokens = gen_config.max_tokens.min(
                max_seq_len.saturating_sub(prompt_tokens.len())
            );

            let client: R::Client = R::default_client(&self.device);

            // ── Encode images ──
            let image_embeds: Option<Tensor<R>> = if !images.is_empty() {
                let vc = vision_config.ok_or_else(|| {
                    anyhow!("Vision encoder config missing despite the presence check above")
                })?;
                let image_size = vc.image_size;

                // Preprocess all images into a batched pixel tensor [N, 3, H, W]
                let mut all_pixels: Vec<f32> = Vec::new();
                for img_bytes in images {
                    let pixels = boostr::model::vision::preprocess::preprocess_image(
                        img_bytes, image_size,
                    ).map_err(|e| anyhow!("Image preprocessing failed: {}", e))?;
                    all_pixels.extend_from_slice(&pixels);
                }
                let num_images = images.len();
                let pixel_tensor: Tensor<R> = Tensor::from_slice(
                    &all_pixels,
                    &[num_images, 3, image_size, image_size],
                    &self.device,
                )?;

                // Encode through vision encoder + projector → [N, num_patches, llm_hidden]
                let embeds: Tensor<R> = multimodal_model.encode_images(&client, &pixel_tensor)
                    .map_err(|e| anyhow!("Vision encoding failed: {}", e))?;

                tracing::info!(
                    num_images = num_images,
                    embed_shape = ?embeds.shape(),
                    "Encoded image embeddings"
                );
                Some(embeds)
            } else {
                None
            };

            // ── Encode audio ──
            // Vision and audio share this path, so the module is not gated as a
            // whole; only the audio branch is. Without the feature a request
            // carrying audio is refused rather than answered from its text
            // alone, which would look like a working reply to a half-read prompt.
            #[cfg(not(feature = "audio"))]
            if !audio_segments.is_empty() {
                yield Err(anyhow!(
                    "Request carries audio but this build lacks the `audio` feature. \
                     Rebuild blazr with --features audio."
                ));
                return;
            }
            #[cfg(not(feature = "audio"))]
            let audio_embeds: Option<Tensor<R>> = None;
            #[cfg(feature = "audio")]
            let audio_embeds: Option<Tensor<R>> = if !audio_segments.is_empty() {
                let ac = audio_config.ok_or_else(|| {
                    anyhow!("Audio encoder config missing despite the presence check above")
                })?;
                let num_mel_bins = ac.num_mel_bins;
                let sample_rate = 16000usize; // Standard for Whisper-family encoders

                // Process each audio segment into mel spectrogram, then batch
                let mut all_mels = Vec::new();
                let mut num_frames = 0usize;
                for (i, samples) in audio_segments.iter().enumerate() {
                    let mel = boostr::model::audio::mel::compute_mel_spectrogram(
                        samples, num_mel_bins, sample_rate,
                    )
                    .map_err(|e| anyhow!("Mel spectrogram failed: {}", e))?;
                    let frames = mel.len() / num_mel_bins;
                    if i == 0 {
                        num_frames = frames;
                    } else if frames != num_frames {
                        // Pad or truncate to match first segment's frame count
                        // (batch dimension requires uniform size)
                        let target_len = num_mel_bins * num_frames;
                        let mut padded = mel;
                        padded.resize(target_len, 0.0);
                        all_mels.extend_from_slice(&padded[..target_len]);
                        continue;
                    }
                    all_mels.extend_from_slice(&mel);
                }
                let num_audio = audio_segments.len();
                let mel_tensor: Tensor<R> = Tensor::from_slice(
                    &all_mels,
                    &[num_audio, num_mel_bins, num_frames],
                    &self.device,
                )?;

                // Encode through audio encoder + projector → [N, num_audio_tokens, llm_hidden]
                let embeds: Tensor<R> = multimodal_model.encode_audio(&client, &mel_tensor)
                    .map_err(|e| anyhow!("Audio encoding failed: {}", e))?;

                tracing::info!(
                    num_audio = num_audio,
                    embed_shape = ?embeds.shape(),
                    "Encoded audio embeddings"
                );
                Some(embeds)
            } else {
                None
            };

            // ── Build multimodal input and run prefill ──
            //
            // Strategy: use the LLM backbone's embedding layer to get text embeddings,
            // merge with image/audio embeddings at placeholder positions, then run
            // through the transformer layers.
            //
            // For the prefill, we use forward_embed + forward_layers_range + forward_head
            // on the LLM backbone. For decode, we use forward_with_kv_cache.

            let llm = multimodal_model.llm();
            let input = self.create_input_tensor(prompt_tokens)?;

            // Get text embeddings [1, seq_len, hidden]
            let text_embeds = llm.forward_embed(&input)
                .map_err(|e| anyhow!("Embedding forward failed: {}", e))?;

            // Merge image/audio embeddings into the text embedding sequence.
            // Convention: <image> placeholder tokens are replaced by image embeddings,
            // <audio> placeholder tokens are replaced by audio embeddings.
            // For simplicity, image embeddings are prepended before the text sequence
            // and audio embeddings are appended, creating a concatenated embedding.
            let merged_embeds = {
                let mut parts: Vec<Tensor<R>> = Vec::new();

                if let Some(ref img_emb) = image_embeds {
                    // Reshape from [N, patches, hidden] to [1, N*patches, hidden]
                    let shape = img_emb.shape();
                    let total_img_tokens = shape[0] * shape[1];
                    let hidden = shape[2];
                    let flat = img_emb.reshape(&[1, total_img_tokens, hidden])
                        .map_err(|e| anyhow!("Image embed reshape failed: {}", e))?;
                    parts.push(flat);
                }

                parts.push(text_embeds.tensor().clone());

                if let Some(ref aud_emb) = audio_embeds {
                    let shape = aud_emb.shape();
                    let total_aud_tokens = shape[0] * shape[1];
                    let hidden = shape[2];
                    let flat = aud_emb.reshape(&[1, total_aud_tokens, hidden])
                        .map_err(|e| anyhow!("Audio embed reshape failed: {}", e))?;
                    parts.push(flat);
                }

                // Concatenate along sequence dimension
                if parts.len() == 1 {
                    Var::new(parts.remove(0), false)
                } else {
                    let refs: Vec<&Tensor<R>> = parts.iter().collect();
                    let concatenated = Tensor::cat(&refs, 1)
                        .map_err(|e| anyhow!("Embedding concatenation failed: {}", e))?;
                    Var::new(concatenated, false)
                }
            };

            // Set up KV cache for the full merged sequence
            let merged_seq_len = merged_embeds.tensor().shape()[1];
            let num_layers = llm.num_layers();
            let num_kv_heads = llm.num_kv_heads().unwrap_or(8);
            let head_dim = llm.head_dim().unwrap_or(64);
            let initial_capacity = (merged_seq_len + max_tokens).min(max_seq_len);
            let kv_dtype = self.config.kv_cache_dtype()?;

            let mut kv_cache = LayeredKvCache::new_positional(
                num_layers, 1, num_kv_heads, initial_capacity, max_seq_len,
                head_dim, kv_dtype, &self.device,
            ).map_err(|e| anyhow!("Failed to create KV cache: {}", e))?;

            // Prefill: run merged embeddings through transformer layers + LM head
            tracing::info!(
                phase = "prefill_start",
                backend = "multimodal",
                text_tokens = prompt_tokens.len(),
                merged_seq_len = merged_seq_len,
                num_images = images.len(),
                num_audio = audio_segments.len(),
            );

            let mut logits = llm.forward_layers_range(
                merged_embeds, None, &mut kv_cache, 0, num_layers, 0,
            ).map_err(|e| anyhow!("Multimodal prefill layers failed: {}", e))
            .and_then(|(hidden, prev_mlp)| {
                llm.forward_head(hidden, prev_mlp)
                    .map_err(|e| anyhow!("Multimodal prefill head failed: {}", e))
            })?;

            tracing::info!(phase = "prefill_end", backend = "multimodal");

            // ── Autoregressive decode (text-only, same as regular generate) ──
            // This path applies no grammar, so the sampler carries no DFA.
            let mut sampler = SamplerState {
                token_history: prompt_tokens.to_vec(),
                mirostat: Self::mirostat_state(gen_config),
                grammar_dfa: None,
                device_grammar: None,
                text: TokenTextStream::new(self.tokenizer(), prompt_tokens),
            };

            tracing::info!(phase = "decode_start", backend = "multimodal", max_tokens = max_tokens);
            for i in 0..max_tokens {
                let cur_logits = &logits;
                let pending = self.sample_begin(cur_logits, &mut sampler, gen_config, None)?;

                let next_input = pending.tensor.reshape(&[1, 1])?;
                let position = kv_cache.seq_len();
                let next_logits = llm.forward_with_kv_cache(&next_input, &mut kv_cache, position)
                    .map_err(|e| anyhow!("Decode forward failed: {}", e))?;

                let next_token = self.sample_finish(&pending, &mut sampler)?;

                let finish = self.decode_finish_reason(next_token, i, max_tokens);

                yield Ok(self.make_token(next_token, &mut sampler.text, cur_logits, gen_config, finish)?);
                if finish == Some(FinishReason::Eos) { break; }
                logits = next_logits;
            }
            tracing::info!(phase = "decode_end", backend = "multimodal");
        }
    }

    /// Non-streaming multimodal text generation.
    ///
    /// Collects all tokens from `generate_multimodal` into a `GenerationResult`,
    /// handling stop sequences and logprobs the same way as `generate_text`.
    pub async fn generate_multimodal_text(
        &self,
        prompt_tokens: &[u32],
        images: &[Vec<u8>],
        audio_segments: &[Vec<f32>],
        gen_config: &GenerationConfig,
    ) -> Result<super::types::GenerationResult> {
        let stream = self.generate_multimodal(prompt_tokens, images, audio_segments, gen_config);
        super::generate_text::collect_generation(stream, prompt_tokens.len(), gen_config).await
    }
}
