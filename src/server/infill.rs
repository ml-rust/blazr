//! Fill-in-the-middle (FIM) endpoint handler
//!
//! Implements `POST /v1/infill` for code completion models that support
//! FIM tokens (e.g., StarCoder, CodeLlama, GPT-4 with cl100k_base).
//!
//! The endpoint takes a `prefix` and `suffix`, wraps them with the model's
//! FIM special tokens, and generates the middle portion.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::generation::{
    apply_keep_alive, error_response, overloaded_response, record_generation_metrics,
    stream_with_stop_sequences, validate_generation_params, HasSamplingFields, SamplingParams,
    Usage,
};
use super::handlers::AppState;
use super::metrics;
use super::streaming::{create_completion_stream, StreamToken};

/// FIM token names used across different model families
const FIM_PREFIX_TOKEN: &str = "<|fim_prefix|>";
const FIM_MIDDLE_TOKEN: &str = "<|fim_middle|>";
const FIM_SUFFIX_TOKEN: &str = "<|fim_suffix|>";

/// Build a FIM prompt from prefix and suffix using special token IDs.
///
/// PSM order (prefix-suffix-middle): `<fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>`
/// This is the standard order used by StarCoder, CodeLlama, and OpenAI models.
fn build_fim_prompt(
    tokenizer: &dyn crate::tokenizer::TokenizerTrait,
    prefix: &str,
    suffix: &str,
) -> Result<String, String> {
    // Check if tokenizer supports FIM tokens
    let fim_prefix = tokenizer
        .special_token_id(FIM_PREFIX_TOKEN)
        .ok_or_else(|| {
            "Model tokenizer does not support FIM tokens (<|fim_prefix|>)".to_string()
        })?;
    let fim_suffix = tokenizer
        .special_token_id(FIM_SUFFIX_TOKEN)
        .ok_or_else(|| {
            "Model tokenizer does not support FIM tokens (<|fim_suffix|>)".to_string()
        })?;
    let fim_middle = tokenizer
        .special_token_id(FIM_MIDDLE_TOKEN)
        .ok_or_else(|| {
            "Model tokenizer does not support FIM tokens (<|fim_middle|>)".to_string()
        })?;

    // Decode token IDs to their string representations
    let prefix_tok = tokenizer
        .decode(&[fim_prefix])
        .map_err(|e| format!("Failed to decode FIM prefix token: {}", e))?;
    let suffix_tok = tokenizer
        .decode(&[fim_suffix])
        .map_err(|e| format!("Failed to decode FIM suffix token: {}", e))?;
    let middle_tok = tokenizer
        .decode(&[fim_middle])
        .map_err(|e| format!("Failed to decode FIM middle token: {}", e))?;

    // PSM order: <fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>
    Ok(format!(
        "{}{}{}{}{}",
        prefix_tok, prefix, suffix_tok, suffix, middle_tok
    ))
}

/// FIM/Infill endpoint
pub async fn infill(
    State(state): State<Arc<AppState>>,
    Json(request): Json<InfillRequest>,
) -> Response {
    if let Err(e) =
        validate_generation_params(request.temperature, request.top_p, request.max_tokens)
    {
        return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
    }

    let load_start = Instant::now();
    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                StatusCode::NOT_FOUND,
                &format!("Model not found: {}", e),
                "invalid_request_error",
            );
        }
    };
    let load_duration_ms = load_start.elapsed().as_millis() as u64;

    // Build FIM prompt
    let prompt = match build_fim_prompt(
        executor.tokenizer(),
        &request.prefix,
        request.suffix.as_deref().unwrap_or(""),
    ) {
        Ok(p) => p,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, &e, "invalid_request_error");
        }
    };

    let gen_config = request.sampling_params().into_gen_config();

    // Token budget admission control
    let prompt_token_count = executor
        .tokenizer()
        .encode(&prompt)
        .map(|t| t.len())
        .unwrap_or(prompt.len() / 4);
    let estimated_tokens = prompt_token_count + gen_config.max_tokens;
    if !state.try_admit(estimated_tokens) {
        return overloaded_response();
    }

    if request.stream.unwrap_or(false) {
        let id = format!("fim-{}", uuid::Uuid::new_v4());
        let model_name = request.model.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamToken>(32);
        let state_clone = Arc::clone(&state);
        let budget = estimated_tokens;

        metrics::adjust_decode_slots(1.0);
        tokio::spawn(async move {
            stream_with_stop_sequences(executor, prompt, gen_config, tx).await;
            state_clone.release(budget);
            metrics::adjust_decode_slots(-1.0);
        });

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        create_completion_stream(id, model_name, Box::pin(rx_stream)).into_response()
    } else {
        metrics::adjust_decode_slots(1.0);
        let model_name = request.model.clone();
        let start = Instant::now();

        match executor.generate_text(&prompt, &gen_config).await {
            Ok(result) => {
                let elapsed = start.elapsed();
                record_generation_metrics(
                    &model_name,
                    &request.user,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.prompt_eval_duration_ms,
                    elapsed,
                );
                apply_keep_alive(&state, &model_name, &request.keep_alive).await;
                state.release(estimated_tokens);
                metrics::adjust_decode_slots(-1.0);

                let response = InfillResponse {
                    id: format!("fim-{}", uuid::Uuid::new_v4()),
                    object: "text_completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: request.model,
                    text: result.text,
                    usage: Usage {
                        prompt_tokens: result.prompt_tokens,
                        completion_tokens: result.completion_tokens,
                        total_tokens: result.prompt_tokens + result.completion_tokens,
                        total_duration_ms: elapsed.as_millis() as u64,
                        prompt_eval_duration_ms: result.prompt_eval_duration_ms,
                        load_duration_ms,
                        tokens_per_second: if elapsed.as_secs_f64() > 0.0 {
                            result.completion_tokens as f64 / elapsed.as_secs_f64()
                        } else {
                            0.0
                        },
                    },
                    finish_reason: result.finish_reason.as_str().to_string(),
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(e) => {
                state.release(estimated_tokens);
                metrics::adjust_decode_slots(-1.0);
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e.to_string(),
                    "server_error",
                )
            }
        }
    }
}

// ── Request/Response types ──

#[derive(Deserialize)]
pub struct InfillRequest {
    pub model: String,
    /// Code before the cursor
    pub prefix: String,
    /// Code after the cursor (optional)
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub keep_alive: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
}

impl HasSamplingFields for InfillRequest {
    fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: self.stop.clone(),
            seed: self.seed,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            json_mode: false,
            mirostat_mode: None,
            mirostat_tau: None,
            mirostat_eta: None,
            dynatemp_range: None,
            dynatemp_exponent: None,
            dry_multiplier: None,
            dry_base: None,
            dry_allowed_length: None,
            dry_sequence_breakers: None,
            typical_p: None,
            grammar: None,
            lora_adapter: None,
        }
    }
}

#[derive(Serialize)]
pub struct InfillResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub text: String,
    pub usage: Usage,
    pub finish_reason: String,
}
