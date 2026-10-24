//! `POST /api/embed` and the legacy `POST /api/embeddings`.
//!
//! Each vector is the mean of the token states, L2-normalized. The legacy
//! route normalizes too, where Ollama returns the raw vector.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{rejection::JsonRejection, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Map, Value};

use super::model_name::resolve;
use super::ollama_error::{bad_request, from_openai, rejection};
use super::ollama_render::nanos;
use super::ollama_types::{
    OllamaEmbedRequest, OllamaEmbedResponse, OllamaEmbeddingsRequest, OllamaEmbeddingsResponse,
    OllamaKeepAlive,
};
use crate::engine::Executor;
use crate::server::embeddings::handler::{embed_text, TokenLimit};
use crate::server::embeddings::options::{check_dimensions, Pooling};
use crate::server::generation::apply_keep_alive;
use crate::server::handlers::{select_executor, AppState, ServerRuntime};

/// Log each `options` key: embeddings take none.
fn log_ignored(options: Option<&Map<String, Value>>) {
    for key in options.into_iter().flat_map(Map::keys) {
        tracing::debug!(option = key.as_str(), "Ollama embed option ignored");
    }
}

/// The token limit of `executor` when it reports a context length.
fn limit(executor: &Executor<ServerRuntime>, truncate: bool) -> Option<TokenLimit> {
    let max = executor.config().inference.max_context_len?;
    Some(TokenLimit { max, truncate })
}

/// The keep-alive argument of `keep_alive`.
fn keep_alive_arg(keep_alive: Option<&OllamaKeepAlive>) -> Option<String> {
    keep_alive.map(OllamaKeepAlive::to_arg)
}

/// POST /api/embed. An empty input list loads the model and returns no vectors.
pub async fn ollama_embed(
    State(state): State<Arc<AppState>>,
    body: Result<Json<OllamaEmbedRequest>, JsonRejection>,
) -> Response {
    let start = Instant::now();
    let request = match body {
        Ok(Json(request)) => request,
        Err(e) => return rejection(e),
    };
    log_ignored(request.options.as_ref());
    if let Err(message) = check_dimensions(request.dimensions, Pooling::Mean) {
        return bad_request(&message);
    }
    let model = resolve(&state, &request.model).await;
    let executor = match select_executor(&state, &model, None).await {
        Ok(executor) => executor,
        Err(response) => return from_openai(*response, &request.model).await,
    };
    let load = start.elapsed();
    let keep_alive = keep_alive_arg(request.keep_alive.as_ref());
    let inputs = request.input.map(|i| i.into_vec()).unwrap_or_default();
    if inputs.is_empty() {
        apply_keep_alive(&state, &model, &keep_alive).await;
        let body = json!({"model": request.model, "embeddings": []});
        return (StatusCode::OK, Json(body)).into_response();
    }
    let limit = limit(&executor, request.truncate.unwrap_or(true));
    let mut embeddings = Vec::with_capacity(inputs.len());
    let mut prompt_eval_count = 0;
    for (i, text) in inputs.iter().enumerate() {
        if text.is_empty() {
            return bad_request(&format!("input {i} is empty"));
        }
        match embed_text(
            &executor,
            text,
            Pooling::Mean,
            request.dimensions,
            true,
            limit,
        )
        .await
        {
            Ok((vector, tokens)) => {
                embeddings.push(vector);
                prompt_eval_count += tokens;
            }
            Err(response) => return from_openai(*response, &request.model).await,
        }
    }
    apply_keep_alive(&state, &model, &keep_alive).await;
    let response = OllamaEmbedResponse {
        model: request.model,
        embeddings,
        total_duration: nanos(start.elapsed()),
        load_duration: nanos(load),
        prompt_eval_count,
    };
    (StatusCode::OK, Json(response)).into_response()
}

/// POST /api/embeddings, the legacy form: one `prompt`, one normalized
/// `embedding`. An empty prompt returns an empty vector.
pub async fn ollama_embeddings(
    State(state): State<Arc<AppState>>,
    body: Result<Json<OllamaEmbeddingsRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(e) => return rejection(e),
    };
    log_ignored(request.options.as_ref());
    let model = resolve(&state, &request.model).await;
    let executor = match select_executor(&state, &model, None).await {
        Ok(executor) => executor,
        Err(response) => return from_openai(*response, &request.model).await,
    };
    let keep_alive = keep_alive_arg(request.keep_alive.as_ref());
    let embedding = if request.prompt.is_empty() {
        Vec::new()
    } else {
        let limit = limit(&executor, true);
        match embed_text(&executor, &request.prompt, Pooling::Mean, None, true, limit).await {
            Ok((vector, _)) => vector,
            Err(response) => return from_openai(*response, &request.model).await,
        }
    };
    apply_keep_alive(&state, &model, &keep_alive).await;
    (StatusCode::OK, Json(OllamaEmbeddingsResponse { embedding })).into_response()
}

/// End-to-end runs through the router on a loaded seeded model.
#[cfg(all(test, not(any(feature = "cuda", feature = "wgpu"))))]
mod tests {
    use anyhow::{anyhow, Result};
    use serde_json::{json, Value};

    use super::super::e2e_rig::{Rig, REQUESTED};

    /// How far an L2-normalized vector's norm can differ from 1.
    const NORM_TOL: f64 = 1e-5;

    /// The vectors of `/api/embed`'s `embeddings`.
    fn vectors(got: &Value) -> Result<Vec<Vec<f64>>> {
        Ok(serde_json::from_value(got["embeddings"].clone())?)
    }

    fn assert_unit(vector: &[f64]) -> Result<()> {
        let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
        if (norm - 1.0).abs() > NORM_TOL {
            return Err(anyhow!("norm {norm} of {vector:?}"));
        }
        Ok(())
    }

    #[tokio::test]
    async fn embed_returns_one_unit_vector_per_input() -> Result<()> {
        let rig = Rig::new().await?;
        let one = rig
            .ok("/api/embed", &json!({"model": REQUESTED, "input": "red"}))
            .await?;
        assert_eq!(one["model"], REQUESTED, "{one}");
        let single = vectors(&one)?;
        assert_eq!(single.len(), 1, "{one}");
        assert_unit(&single[0])?;
        assert!(one["prompt_eval_count"].as_u64() > Some(0), "{one}");

        let body = json!({"model": REQUESTED, "input": ["red", "a blue sky"]});
        let many = vectors(&rig.ok("/api/embed", &body).await?)?;
        assert_eq!(many.len(), 2);
        for vector in &many {
            assert_unit(vector)?;
        }
        assert_eq!(
            many[0], single[0],
            "one text embeds alike alone or in a list"
        );
        assert_ne!(many[0], many[1]);
        Ok(())
    }

    #[tokio::test]
    async fn dimensions_truncate_then_normalize() -> Result<()> {
        let rig = Rig::new().await?;
        let full = vectors(
            &rig.ok("/api/embed", &json!({"model": REQUESTED, "input": "red"}))
                .await?,
        )?;
        let body = json!({"model": REQUESTED, "input": "red", "dimensions": 4});
        let cut = vectors(&rig.ok("/api/embed", &body).await?)?;
        assert_eq!(cut[0].len(), 4);
        assert!(full[0].len() > 4);
        assert_unit(&cut[0])?;
        // The cut vector points along the full vector's first 4 components.
        let head = &full[0][..4];
        let scale = head.iter().map(|x| x * x).sum::<f64>().sqrt();
        for (c, h) in cut[0].iter().zip(head) {
            assert!((c - h / scale).abs() < NORM_TOL, "{cut:?} vs {head:?}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn legacy_embeddings_returns_one_unit_embedding() -> Result<()> {
        let rig = Rig::new().await?;
        let body = json!({"model": REQUESTED, "prompt": "red"});
        let got = rig.ok("/api/embeddings", &body).await?;
        let legacy: Vec<f64> = serde_json::from_value(got["embedding"].clone())?;
        assert_unit(&legacy)?;
        let embed = vectors(
            &rig.ok("/api/embed", &json!({"model": REQUESTED, "input": "red"}))
                .await?,
        )?;
        assert_eq!(legacy, embed[0]);
        Ok(())
    }
}
