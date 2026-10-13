//! `POST /v1/systemone`: typed questions about one state, answered by a
//! System One decision model.
//!
//! The body is the Jev wire shape, `{model?, state, questions: {id: {type,
//! instructions, criteria}}, calibration?: "raw", ...}`, each family's own
//! fields included; question order is kept and unknown fields are ignored.
//! The prompt renders on a blocking thread and the model runs on the
//! model's decision worker, never on an async worker.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value};

use super::answer::answers_json;
use super::error::failure_response;
use super::response::{response_body, ResponseFacts};
use crate::engine::decision::{unique_tokens, DecisionFailure};
use crate::model::decision_template::{Calibration, DecisionError, JevRequest};
use crate::server::handlers::AppState;

/// `POST /v1/systemone`.
pub async fn systemone(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let started = Instant::now();
    match decide(&state, &body, started).await {
        Ok(value) => (StatusCode::OK, axum::Json(value)).into_response(),
        Err(failure) => failure_response(&failure),
    }
}

/// The body as a JSON object; an empty body reads as `{}`.
fn parse_body(body: &[u8]) -> Result<Value, DecisionFailure> {
    let value: Value = if body.iter().all(u8::is_ascii_whitespace) {
        Value::Object(Map::new())
    } else {
        serde_json::from_slice(body)
            .map_err(|e| DecisionFailure::InvalidRequest(format!("body is not valid JSON: {e}")))?
    };
    if value.is_object() {
        Ok(value)
    } else {
        Err(DecisionFailure::InvalidRequest(
            "body must be a JSON object".into(),
        ))
    }
}

/// The model the request names, else the start model.
fn model_name(body: &Value, start: Option<&str>) -> Result<String, DecisionFailure> {
    match body.get("model") {
        Some(Value::String(s)) if !s.trim().is_empty() => Ok(s.trim().to_string()),
        None | Some(Value::Null) | Some(Value::String(_)) => {
            start.map(str::to_string).ok_or_else(|| {
                DecisionFailure::InvalidRequest(
                    "'model' is required: this server has no start model (--model) to answer with"
                        .into(),
                )
            })
        }
        Some(_) => Err(DecisionFailure::InvalidRequest(
            "'model' must be a string when given".into(),
        )),
    }
}

/// A renderer error as a request failure: the family's own message for a
/// request it refuses, a server error for a model that cannot render.
fn render_failure(e: DecisionError) -> DecisionFailure {
    match e {
        DecisionError::Request(m) => DecisionFailure::InvalidRequest(m),
        DecisionError::Model(m) => DecisionFailure::Internal(m),
    }
}

async fn decide(state: &AppState, body: &[u8], started: Instant) -> Result<Value, DecisionFailure> {
    let body = parse_body(body)?;
    let name = model_name(&body, state.scheduler.decision_start_model())?;
    let service = state.scheduler.decision_service(&name).await?;
    let request = JevRequest::parse(&body, service.family()).map_err(render_failure)?;
    let raw = request.calibration == Calibration::Raw;
    let renderer = Arc::clone(&service);
    let rendered = tokio::task::spawn_blocking(move || renderer.renderer().render(&request))
        .await
        .map_err(|e| DecisionFailure::Internal(format!("the prompt render stopped: {e}")))?
        .map_err(render_failure)?;
    let answers = if rendered.questions.is_empty() {
        Map::new()
    } else {
        let prepared = service.prepare(&rendered, raw)?;
        let result = service.submit(prepared)?.wait().await?;
        answers_json(service.family(), &rendered.questions, &result)
            .map_err(DecisionFailure::Internal)?
    };
    let mut warnings = service.warnings().to_vec();
    warnings.extend(rendered.warnings.iter().cloned());
    let facts = ResponseFacts {
        model: &name,
        input_tokens: unique_tokens(&rendered.rows),
        latency_ms: started.elapsed().as_secs_f64() * 1000.0,
        calibration: service.calibration(),
        raw,
        warnings,
    };
    Ok(response_body(facts, answers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn body_must_be_a_json_object() {
        assert_eq!(parse_body(b"  ").expect("empty is {}"), json!({}));
        assert!(matches!(
            parse_body(b"{bad"),
            Err(DecisionFailure::InvalidRequest(m)) if m.starts_with("body is not valid JSON")
        ));
        assert!(matches!(
            parse_body(b"[1]"),
            Err(DecisionFailure::InvalidRequest(m)) if m == "body must be a JSON object"
        ));
    }

    #[test]
    fn model_falls_back_to_the_start_model() {
        assert_eq!(
            model_name(&json!({"model": " d "}), None).expect("named"),
            "d"
        );
        assert_eq!(
            model_name(&json!({"model": ""}), Some("start")).expect("start"),
            "start"
        );
        assert_eq!(
            model_name(&json!({}), Some("start")).expect("start"),
            "start"
        );
        assert!(matches!(
            model_name(&json!({}), None),
            Err(DecisionFailure::InvalidRequest(_))
        ));
        assert!(matches!(
            model_name(&json!({"model": 3}), Some("start")),
            Err(DecisionFailure::InvalidRequest(_))
        ));
    }
}
