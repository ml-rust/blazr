//! The `POST /v1/embeddings` handler.

use std::sync::Arc;

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::options::{check_dimensions, check_token_ids, finish_vector, EncodingFormat, Pooling};
use super::request::{EmbedItem, EmbeddingRequest};
use super::response::{EmbeddingData, EmbeddingResponse, EmbeddingUsage, EmbeddingValue};
use crate::engine::pooling::{pool_cls, pool_last, pool_mean};
use crate::engine::Executor;
use crate::server::encoding::encode_f32_base64;
use crate::server::generation::{error_response, policy_error_response};
use crate::server::handlers::{lookup_error_response, AppState, ServerRuntime, VisionEmbedder};
use crate::server::multimodal::decode_image;

/// The last hidden states of `ids`, computed on a blocking thread so the
/// forward pass never holds an async worker.
///
/// # Errors
///
/// The forward pass's error, or the blocking task's when it panicked.
pub(crate) async fn hidden_states(
    executor: &Arc<Executor<ServerRuntime>>,
    ids: Vec<u32>,
) -> anyhow::Result<Vec<f32>> {
    let executor = Arc::clone(executor);
    tokio::task::spawn_blocking(move || executor.get_embeddings(&ids))
        .await
        .map_err(|e| anyhow::anyhow!("the embedding task stopped: {e}"))?
}

/// The embedding of one encoded image, computed on a blocking thread so the
/// encoder forward never holds an async worker.
///
/// # Errors
///
/// The encoder's error, or the blocking task's when it panicked.
async fn image_embedding(
    embedder: Arc<VisionEmbedder>,
    image: Vec<u8>,
) -> anyhow::Result<Vec<f32>> {
    tokio::task::spawn_blocking(move || {
        let device = <ServerRuntime as boostr::Runtime>::default_device();
        embedder.embed_bytes_default(&image, &device)
    })
    .await
    .map_err(|e| anyhow::anyhow!("the image embedding task stopped: {e}"))?
    .map_err(anyhow::Error::from)
}

fn bad_request(message: &str) -> Response {
    error_response(StatusCode::BAD_REQUEST, message, "invalid_request_error")
}

/// Pool the hidden states of `token_ids` into one vector.
///
/// # Errors
///
/// The `500` response of a failed forward pass.
pub(crate) async fn embed_tokens(
    executor: &Arc<Executor<ServerRuntime>>,
    token_ids: Vec<u32>,
    pooling: Pooling,
) -> Result<Vec<f32>, Box<Response>> {
    let num_tokens = token_ids.len();
    let hidden = hidden_states(executor, token_ids).await.map_err(|e| {
        Box::new(error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("Embedding generation failed: {e}"),
            "server_error",
        ))
    })?;
    let hidden_size = hidden.len().checked_div(num_tokens).unwrap_or(0);
    Ok(match pooling {
        Pooling::Mean => pool_mean(&hidden, num_tokens, hidden_size),
        Pooling::Cls => pool_cls(&hidden, hidden_size),
        Pooling::Last => pool_last(&hidden, num_tokens, hidden_size),
        Pooling::None => hidden,
    })
}

/// The finished vector of caller `text` and its token count.
///
/// The text has no template around it, so it encodes under
/// `SpecialMode::Ordinary`: nothing in it can become a control token.
/// The pooled vector then takes `dimensions` and `normalize`.
///
/// # Errors
///
/// `400` for text that spells a control token or a vector that does not
/// take `dimensions`, `500` for a failed forward pass.
pub(crate) async fn embed_text(
    executor: &Arc<Executor<ServerRuntime>>,
    text: &str,
    pooling: Pooling,
    dimensions: Option<usize>,
    normalize: bool,
) -> Result<(Vec<f32>, usize), Box<Response>> {
    let ids = executor
        .tokenizer()
        .encode_with(text, &splintr::SpecialMode::Ordinary)
        .map_err(|e| Box::new(policy_error_response(&e)))?;
    let num_tokens = ids.len();
    let pooled = embed_tokens(executor, ids, pooling).await?;
    Ok((finished(pooled, dimensions, normalize)?, num_tokens))
}

/// `vector` cut to `dimensions` and normalized when `normalize` is set.
///
/// # Errors
///
/// The `400` response of a vector that does not take `dimensions`.
fn finished(
    mut vector: Vec<f32>,
    dimensions: Option<usize>,
    normalize: bool,
) -> Result<Vec<f32>, Box<Response>> {
    finish_vector(&mut vector, dimensions, normalize).map_err(|msg| Box::new(bad_request(&msg)))?;
    Ok(vector)
}

/// Embeddings endpoint
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
) -> Response {
    let encoding = match EncodingFormat::parse(request.encoding_format.as_deref()) {
        Ok(e) => e,
        Err(msg) => return bad_request(&msg),
    };
    let pooling = match Pooling::parse(&request.pooling) {
        Ok(p) => p,
        Err(msg) => return bad_request(&msg),
    };
    if let Err(msg) = check_dimensions(request.dimensions, pooling) {
        return bad_request(&msg);
    }

    let executor = match state.scheduler.get_executor(&request.model).await {
        Ok(e) => e,
        Err(e) => return lookup_error_response(&e),
    };

    // Route images to the standalone vision embedder registered for this model.
    let vision_embedder = if request.input.has_images() {
        match state.vision_embedder(&request.model).await {
            Some(v) => Some(v),
            None => {
                return bad_request(&format!(
                    "Model '{}' is not registered as a vision embedder; image inputs require a loaded SigLIP/CLIP encoder",
                    request.model
                ));
            }
        }
    } else {
        None
    };

    let items = request.input.items();
    if items.is_empty() {
        return bad_request("Input must not be empty");
    }
    let vocab_size = executor.vocab_size();
    for item in &items {
        if let EmbedItem::Tokens(ids) = item {
            if ids.is_empty() {
                return bad_request("Input must not be empty");
            }
            if let Err(msg) = check_token_ids(ids, vocab_size) {
                return bad_request(&msg);
            }
        }
    }

    let mut data = Vec::with_capacity(items.len());
    let mut prompt_tokens = 0usize;

    for (index, item) in items.into_iter().enumerate() {
        let embedding = match item {
            EmbedItem::Text(text) => {
                match embed_text(
                    &executor,
                    text,
                    pooling,
                    request.dimensions,
                    request.normalize,
                )
                .await
                {
                    Ok((embedding, num_tokens)) => {
                        prompt_tokens += num_tokens;
                        embedding
                    }
                    Err(resp) => return *resp,
                }
            }
            EmbedItem::Tokens(ids) => {
                prompt_tokens += ids.len();
                let pooled = match embed_tokens(&executor, ids.to_vec(), pooling).await {
                    Ok(v) => v,
                    Err(resp) => return *resp,
                };
                match finished(pooled, request.dimensions, request.normalize) {
                    Ok(v) => v,
                    Err(resp) => return *resp,
                }
            }
            EmbedItem::Image(image_url) => {
                let Some(embedder) = vision_embedder.as_ref() else {
                    return bad_request("Image inputs require a registered vision embedder");
                };
                let decoded = match decode_image(&image_url.url).await {
                    Ok(d) => d,
                    Err(e) => return bad_request(&format!("Failed to decode image: {e}")),
                };
                let pooled = match image_embedding(Arc::clone(embedder), decoded.data).await {
                    Ok(v) => v,
                    Err(e) => {
                        return error_response(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            &format!("Image embedding failed: {e}"),
                            "server_error",
                        );
                    }
                };
                match finished(pooled, request.dimensions, request.normalize) {
                    Ok(v) => v,
                    Err(resp) => return *resp,
                }
            }
        };

        let value = match encoding {
            EncodingFormat::Base64 => EmbeddingValue::Base64(encode_f32_base64(&embedding)),
            EncodingFormat::Float => EmbeddingValue::Float(embedding),
        };
        data.push(EmbeddingData {
            object: "embedding",
            index,
            embedding: value,
        });
    }

    let response = EmbeddingResponse {
        object: "list",
        data,
        model: request.model,
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    };
    (StatusCode::OK, Json(response)).into_response()
}
