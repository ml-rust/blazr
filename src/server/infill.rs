//! Fill-in-the-middle (FIM) endpoint handler
//!
//! Implements `POST /v1/infill` for code completion models that support
//! FIM tokens (e.g., StarCoder, CodeLlama, GPT-4 with cl100k_base).
//!
//! The endpoint takes a `prefix` and `suffix`, wraps them with the model's
//! FIM special tokens, and generates the middle portion.

use std::sync::{Arc, LazyLock};
use std::time::Instant;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use super::generation::{
    apply_keep_alive, error_response, overloaded_response, policy_error_response,
    record_generation_metrics, stream_with_stop_sequences, validate_generation_params,
    HasSamplingFields, SamplingParams, Usage,
};
use super::handlers::AppState;
use super::metrics;
use super::streaming::{create_completion_stream, StreamToken};

/// FIM token names used across different model families
const FIM_PREFIX_TOKEN: &str = "<|fim_prefix|>";
const FIM_MIDDLE_TOKEN: &str = "<|fim_middle|>";
const FIM_SUFFIX_TOKEN: &str = "<|fim_suffix|>";

/// The only special tokens this endpoint inserts, and therefore the only ones
/// the assembled prompt may contain — a `prefix` spelling out `<|im_start|>` is
/// refused rather than promoted to that control token's id.
///
/// Built once: every infill request borrows it.
static FIM_MARKERS: LazyLock<splintr::FxHashSet<String>> = LazyLock::new(|| {
    [FIM_PREFIX_TOKEN, FIM_MIDDLE_TOKEN, FIM_SUFFIX_TOKEN]
        .iter()
        .map(|t| (*t).to_string())
        .collect()
});

/// Build a FIM prompt from prefix and suffix using the model's FIM markers.
///
/// PSM order (prefix-suffix-middle): `<fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>`
/// This is the standard order used by StarCoder, CodeLlama, and OpenAI models.
///
/// The markers go in as their literal spellings, which is exactly what the
/// allow-list names, so the encode below turns them into the same ids
/// `special_token_id` reports here.
fn build_fim_prompt(
    tokenizer: &splintr::AnyTokenizer,
    prefix: &str,
    suffix: &str,
) -> Result<String, String> {
    // Check if tokenizer supports FIM tokens
    for token in [FIM_PREFIX_TOKEN, FIM_SUFFIX_TOKEN, FIM_MIDDLE_TOKEN] {
        if tokenizer.special_token_id(token).is_none() {
            return Err(format!(
                "Model tokenizer does not support FIM tokens ({})",
                token
            ));
        }
    }

    // PSM order: <fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>
    Ok(format!(
        "{}{}{}{}{}",
        FIM_PREFIX_TOKEN, prefix, FIM_SUFFIX_TOKEN, suffix, FIM_MIDDLE_TOKEN
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

    // Token budget admission control. The single encode of the assembled prompt
    // doubles as the gate: only the FIM markers this endpoint inserted are
    // allowed to be control tokens.
    let prompt_tokens = match executor
        .tokenizer()
        .encode_with(&prompt, &splintr::SpecialMode::Allow(&FIM_MARKERS))
    {
        Ok(tokens) => tokens,
        Err(e) => return policy_error_response(&e),
    };
    let estimated_tokens = prompt_tokens.len() + gen_config.max_tokens;
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
            stream_with_stop_sequences(executor, prompt_tokens, gen_config, tx).await;
            state_clone.release(budget);
            metrics::adjust_decode_slots(-1.0);
        });

        let rx_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        create_completion_stream(id, model_name, Box::pin(rx_stream)).into_response()
    } else {
        metrics::adjust_decode_slots(1.0);
        let model_name = request.model.clone();
        let start = Instant::now();

        match executor.generate_text(&prompt_tokens, &gen_config).await {
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
