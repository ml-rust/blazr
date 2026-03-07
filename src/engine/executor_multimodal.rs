//! `generate_multimodal()` method for the inference executor.
//!
//! Handles multimodal (vision + audio) inference by encoding images/audio
//! through boostr's vision/audio encoders, then running the LLM backbone
//! with merged embeddings for prefill and standard autoregressive decode.

use anyhow::{anyhow, Result};
use async_stream::stream;
use futures::{Stream, StreamExt};

use boostr::autograd::Var;
use boostr::inference::LayeredKvCache;
use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    Tensor, TypeConversionOps, UnaryOps,
};

use crate::config::{parse_dtype, GenerationConfig};
use crate::tokenizer::TokenizerTrait;

use super::sampling::MirostatState;
use super::types::{FinishReason, GeneratedToken};

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
        + ModelClient<R>,
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
        prompt: &'a str,
        images: &'a [Vec<u8>],
        audio_segments: &'a [Vec<f32>],
        gen_config: &'a GenerationConfig,
    ) -> impl Stream<Item = Result<GeneratedToken>> + 'a {
        stream! {
            // Verify model is multimodal
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

            let vision_config = multimodal_model.config().vision.as_ref();
            let audio_config = multimodal_model.config().audio.as_ref();

            // Validate: images require vision encoder, audio requires audio encoder
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

            // Encode prompt tokens
            let prompt_tokens = self.tokenizer.encode(prompt)
                .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

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
                let vc = vision_config.unwrap();
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
                );

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
            let audio_embeds: Option<Tensor<R>> = if !audio_segments.is_empty() {
                let ac = audio_config.unwrap();
                let num_mel_bins = ac.num_mel_bins;
                let sample_rate = 16000usize; // Standard for Whisper-family encoders

                // Process each audio segment into mel spectrogram, then batch
                let mut all_mels = Vec::new();
                let mut num_frames = 0usize;
                for (i, samples) in audio_segments.iter().enumerate() {
                    let mel = boostr::model::audio::mel::compute_mel_spectrogram(
                        samples, num_mel_bins, sample_rate,
                    );
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
                );

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
            let input = self.create_input_tensor(&prompt_tokens)?;

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
                    Var::new(parts.into_iter().next().unwrap(), false)
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
            let kv_dtype = parse_dtype(self.config.dtype())?;

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
            let mut token_history: Vec<u32> = prompt_tokens.clone();
            let mut mirostat: Option<MirostatState> = if gen_config.mirostat_mode >= 2 {
                Some(MirostatState::new(gen_config.mirostat_tau, gen_config.mirostat_eta, gen_config.seed))
            } else {
                None
            };

            tracing::info!(phase = "decode_start", backend = "multimodal", max_tokens = max_tokens);
            for i in 0..max_tokens {
                let cur_logits = &logits;
                let (token_gpu, miro_id) = self.sample_token_dispatch_with_grammar(
                    cur_logits, &token_history, gen_config, &mut mirostat, None,
                )?;
                let event = token_gpu.record_event()
                    .map_err(|e| anyhow!("Record event failed: {}", e))?;

                let next_input = token_gpu.reshape(&[1, 1])?;
                let position = kv_cache.seq_len();
                let next_logits = llm.forward_with_kv_cache(&next_input, &mut kv_cache, position)
                    .map_err(|e| anyhow!("Decode forward failed: {}", e))?;

                let next_token = miro_id.map_or_else(
                    || Self::read_token_id(&token_gpu, event),
                    Ok,
                )?;
                token_history.push(next_token);

                let is_last = i + 1 == max_tokens;
                let finish = if self.tokenizer.is_eos(next_token) {
                    Some(FinishReason::Eos)
                } else if is_last {
                    Some(FinishReason::Length)
                } else {
                    None
                };

                yield Ok(self.make_token(next_token, cur_logits, gen_config, finish)?);
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
        prompt: &str,
        images: &[Vec<u8>],
        audio_segments: &[Vec<f32>],
        gen_config: &GenerationConfig,
    ) -> Result<super::types::GenerationResult> {
        let prompt_tokens = self
            .tokenizer()
            .encode(prompt)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?
            .len();

        let mut result = String::new();
        let mut completion_tokens = 0usize;
        let mut finish_reason = FinishReason::Length;
        let mut stream =
            std::pin::pin!(self.generate_multimodal(prompt, images, audio_segments, gen_config));
        let prefill_start = std::time::Instant::now();
        let mut prompt_eval_duration_ms = 0u64;
        let mut token_logprobs: Vec<GeneratedToken> = Vec::new();

        while let Some(token_result) = stream.next().await {
            let token = token_result?;
            completion_tokens += 1;

            if completion_tokens == 1 {
                prompt_eval_duration_ms = prefill_start.elapsed().as_millis() as u64;
            }

            result.push_str(&token.text);

            if let Some(reason) = token.finish_reason {
                finish_reason = reason;
            }

            if gen_config.logprobs {
                token_logprobs.push(token);
            }

            // Check stop sequences
            for stop in &gen_config.stop_sequences {
                if result.ends_with(stop) {
                    result.truncate(result.len() - stop.len());
                    return Ok(super::types::GenerationResult {
                        text: result,
                        prompt_tokens,
                        completion_tokens,
                        finish_reason: FinishReason::Stop,
                        prompt_eval_duration_ms,
                        token_logprobs: if gen_config.logprobs {
                            Some(token_logprobs)
                        } else {
                            None
                        },
                    });
                }
            }
        }

        Ok(super::types::GenerationResult {
            text: result,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            prompt_eval_duration_ms,
            token_logprobs: if gen_config.logprobs {
                Some(token_logprobs)
            } else {
                None
            },
        })
    }
}
