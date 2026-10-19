//! Non-streaming text generation.

use anyhow::Result;
use futures::StreamExt;

use boostr::model::ModelClient;
use boostr::ops::TensorOps;
use boostr::quant::{DequantOps, QuantMatmulOps};
use boostr::{
    ActivationOps, BinaryOps, ConvOps, DType, NormalizationOps, RoPEPackedOps, Runtime,
    SamplingOps, ScalarOps, TypeConversionOps, UnaryOps,
};

use crate::config::GenerationConfig;
use crate::engine::StopMatcher;

use super::executor::Executor;
use super::types::{FinishReason, GeneratedToken, GenerationResult};

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
        + RoPEPackedOps<R>
        + DequantOps<R>
        + QuantMatmulOps<R>,
{
    /// Generate text from already-encoded prompt ids and return the complete
    /// result with metadata.
    ///
    /// The ids come from the handler's validated encode; this path never
    /// re-tokenizes.
    pub async fn generate_text(
        &self,
        prompt_tokens: &[u32],
        gen_config: &GenerationConfig,
    ) -> Result<GenerationResult> {
        let stream = self.generate(prompt_tokens, gen_config);
        collect_generation(stream, prompt_tokens.len(), gen_config).await
    }
}

/// Drain a token stream into one `GenerationResult`: stop sequences cut the
/// text and finish with `Stop`, logprobs are kept when asked for, and the
/// prompt eval time is the time to the first token.
pub(in crate::engine) async fn collect_generation<S>(
    stream: S,
    prompt_token_count: usize,
    gen_config: &GenerationConfig,
) -> Result<GenerationResult>
where
    S: futures::Stream<Item = Result<GeneratedToken>>,
{
    let mut result = String::new();
    let mut matcher = StopMatcher::new(&gen_config.stop_sequences);
    let mut completion_tokens = 0usize;
    let mut finish_reason = FinishReason::Length;
    let mut stream = std::pin::pin!(stream);
    let prefill_start = std::time::Instant::now();
    let mut prompt_eval_duration_ms = 0u64;
    let mut token_logprobs: Vec<GeneratedToken> = Vec::new();
    let mut last_token = prefill_start;

    while let Some(token_result) = stream.next().await {
        let token = token_result?;
        completion_tokens += 1;

        // The first token closes the prompt eval. Each later one is an ITL
        // sample for the metrics and the SLO checks.
        let now = std::time::Instant::now();
        if completion_tokens == 1 {
            prompt_eval_duration_ms = now.duration_since(prefill_start).as_millis() as u64;
        } else {
            crate::telemetry::record::record_itl(now.duration_since(last_token).as_secs_f64());
        }
        last_token = now;

        let step = matcher.push(&token.text);
        result.push_str(&step.emit);

        if let Some(reason) = token.finish_reason {
            finish_reason = reason;
        }

        if gen_config.logprobs {
            token_logprobs.push(token);
        }

        // The text ends at the first stop, even when it sits inside a token.
        if step.stopped {
            return Ok(GenerationResult {
                text: result,
                prompt_tokens: prompt_token_count,
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

    // No stop matched: held-back text is part of the result.
    result.push_str(&matcher.finish());

    Ok(GenerationResult {
        text: result,
        prompt_tokens: prompt_token_count,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SpeculativeDecodingConfig;
    use crate::engine::test_support::tiny_executor;

    /// Token ids generated from a fixed prompt, greedy.
    async fn greedy_count(exec: &Executor<boostr::CpuRuntime>) -> Result<(String, usize)> {
        let cfg = GenerationConfig {
            max_tokens: 6,
            temperature: 0.0,
            ..GenerationConfig::default()
        };
        let result = exec.generate_text(&[1, 2, 3], &cfg).await?;
        Ok((result.text, result.completion_tokens))
    }

    #[tokio::test]
    async fn speculative_config_leaves_generation_unchanged() -> Result<()> {
        let plain = tiny_executor()?;
        let mut with_spec = tiny_executor()?;
        with_spec.config.inference.speculative = Some(SpeculativeDecodingConfig {
            draft_model: "missing-draft".to_string(),
            ..SpeculativeDecodingConfig::default()
        });
        assert_eq!(greedy_count(&plain).await?, greedy_count(&with_spec).await?);
        Ok(())
    }
}
