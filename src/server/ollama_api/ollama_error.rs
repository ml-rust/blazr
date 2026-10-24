//! Ollama error responses: `{"error": message}` under the HTTP status.

use axum::{
    extract::rejection::JsonRejection,
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE},
        StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

use super::model_name::not_found;

/// The most bytes of an OpenAI error body read back for conversion.
const ERROR_BODY_LIMIT: usize = 64 * 1024;

/// The OpenAI `code` of an unknown model.
const MODEL_NOT_FOUND: &str = "model_not_found";

/// The Ollama error response of `message` under `status`.
pub(in crate::server) fn ollama_error(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({ "error": message }))).into_response()
}

/// The `400` Ollama error response of `message`.
pub(in crate::server) fn bad_request(message: &str) -> Response {
    ollama_error(StatusCode::BAD_REQUEST, message)
}

/// The Ollama error response of a request body that did not parse.
pub(in crate::server) fn rejection(rejection: JsonRejection) -> Response {
    ollama_error(rejection.status(), &rejection.body_text())
}

/// The Ollama form of `response`, an OpenAI-shaped error response.
///
/// The status and every header but the body's type and length stay, so a
/// `503` keeps its `retry-after`. An unknown model reads
/// `model '<requested>' not found`. Any other error keeps its message.
pub(in crate::server) async fn from_openai(response: Response, requested: &str) -> Response {
    let (parts, body) = response.into_parts();
    let bytes = axum::body::to_bytes(body, ERROR_BODY_LIMIT)
        .await
        .unwrap_or_default();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let error = &parsed["error"];
    let message = if parts.status == StatusCode::NOT_FOUND && error["code"] == MODEL_NOT_FOUND {
        not_found(requested)
    } else {
        error["message"]
            .as_str()
            .or_else(|| error.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned())
    };
    let mut out = ollama_error(parts.status, &message);
    for (name, value) in &parts.headers {
        if name != CONTENT_TYPE && name != CONTENT_LENGTH {
            out.headers_mut().append(name.clone(), value.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelNotFound;
    use crate::server::generation::{error_response, overloaded_response};
    use crate::server::handlers::lookup_error_response;

    async fn body_of(response: Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body reads");
        serde_json::from_slice(&bytes).expect("body is JSON")
    }

    #[tokio::test]
    async fn an_unknown_model_names_the_requested_model() {
        let openai = lookup_error_response(&ModelNotFound::new("x").into());
        let out = from_openai(openai, "x:latest").await;
        assert_eq!(out.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body_of(out).await,
            json!({"error": "model 'x:latest' not found"})
        );
    }

    #[tokio::test]
    async fn the_status_message_and_retry_after_survive() {
        let out = from_openai(overloaded_response(), "m").await;
        assert_eq!(out.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(out.headers()["retry-after"], "5");
        assert_eq!(out.headers()[CONTENT_TYPE], "application/json");
        let message = body_of(out).await["error"].as_str().unwrap().to_string();
        assert!(message.starts_with("Server overloaded"), "{message}");

        let bad = error_response(StatusCode::BAD_REQUEST, "no", "invalid_request_error");
        let out = from_openai(bad, "m").await;
        assert_eq!(out.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_of(out).await, json!({"error": "no"}));
    }
}
