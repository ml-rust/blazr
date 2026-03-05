//! Non-streaming text generation with JSON retry support.

use anyhow::{anyhow, Result};
use futures::StreamExt;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
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
        + ModelClient<R>,
{
    /// Generate text and return the complete result with metadata
    pub async fn generate_text(
        &self,
        prompt: &str,
        gen_config: &GenerationConfig,
    ) -> Result<GenerationResult> {
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
