//! Non-streaming text generation with JSON retry support.

use anyhow::{anyhow, Result};
use futures::StreamExt;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::quant::{DequantOps, QuantMatmulOps};
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, Runtime, SamplingOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;

use super::executor::Executor;
use super::types::{is_valid_json, FinishReason, GeneratedToken, GenerationResult};

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
    /// Generate text and return the complete result with metadata
    pub async fn generate_text(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<GenerationResult> {
        // Use speculative decoding if configured
        if self.config().inference.speculative.is_some() {
            return self.generate_text_speculative(prompt, gen_config).await;
        }

        let max_attempts = if gen_config.json_mode { 3 } else { 1 };
        for attempt in 0..max_attempts {
            let result = self.generate_text_once(prompt, gen_config).await?;
            if !gen_config.json_mode || is_valid_json(&result.text) {
                return Ok(result);
            }
            tracing::warn!(
                "JSON mode: attempt {}/{} produced invalid JSON, retrying",
                attempt + 1,
                max_attempts
            );
        }
        self.generate_text_once(prompt, gen_config).await
    }

    /// Speculative decoding path: uses draft model for fast speculation, target for verification.
    async fn generate_text_speculative(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<GenerationResult> {
        let spec_config = self
            .config()
            .inference
            .speculative
            .as_ref()
            .ok_or_else(|| anyhow!("speculative config missing"))?;

        let prompt_tokens = self
            .tokenizer()
            .encode(prompt)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let prompt_len = prompt_tokens.len();
        let prefill_start = std::time::Instant::now();

        // Load separate draft model (smaller, faster model for speculation)
        let draft_model_arc = self.get_or_load_draft_model()?;
        let draft = super::speculative::BlazrSpeculativeModel::new(
            draft_model_arc,
            self.device().clone(),
            self.config().dtype(),
            &spec_config.draft_model,
        )?;

        // Target model is the main (larger) model
        let target = super::speculative::BlazrSpeculativeModel::new(
            std::sync::Arc::clone(self.model_arc()),
            self.device().clone(),
            self.config().dtype(),
            "target",
        )?;

        let boostr_config = boostr::inference::speculative::SpeculativeConfig {
            num_speculative_tokens: spec_config.num_speculative_tokens,
            adaptive_depth: spec_config.adaptive_depth,
            seed: gen_config.seed,
            ..Default::default()
        };

        let mut executor =
            boostr::inference::speculative::SpeculativeExecutor::new(draft, target, boostr_config);

        let generated_ids = executor
            .generate(&prompt_tokens, gen_config.max_tokens)
            .map_err(|e| anyhow!("speculative generation failed: {}", e))?;

        let prompt_eval_duration_ms = prefill_start.elapsed().as_millis() as u64;

        let text = self.tokenizer().decode(&generated_ids).unwrap_or_default();

        // Check for EOS in generated tokens
        let eos_pos = generated_ids
            .iter()
            .position(|&t| self.tokenizer().is_eos(t));
        let (final_text, finish_reason) = if let Some(pos) = eos_pos {
            let text = self
                .tokenizer()
                .decode(&generated_ids[..pos])
                .unwrap_or_default();
            (text, FinishReason::Eos)
        } else {
            (text, FinishReason::Length)
        };

        let stats = executor.stats;
        tracing::info!(
            speculative_iterations = stats.iterations,
            speculative_accepted = stats.accepted_tokens,
            speculative_rejected = stats.rejected_tokens,
            "speculative decoding complete"
        );

        Ok(GenerationResult {
            text: final_text,
            prompt_tokens: prompt_len,
            completion_tokens: generated_ids.len(),
            finish_reason,
            prompt_eval_duration_ms,
            token_logprobs: None,
        })
    }

    /// Single generation attempt (used by generate_text for JSON retry)
    async fn generate_text_once(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<GenerationResult> {
        let prompt_tokens = self
            .tokenizer()
            .encode(prompt)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?
            .len();

        let mut result = String::new();
        let mut completion_tokens = 0usize;
        let mut finish_reason = FinishReason::Length;
        let mut stream = std::pin::pin!(self.generate(prompt, gen_config));
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
                    return Ok(GenerationResult {
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

        Ok(GenerationResult {
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
