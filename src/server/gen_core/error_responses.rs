//! OpenAI-style HTTP error responses and their wire types.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// Return a 503 Service Unavailable response with Retry-After header
pub fn overloaded_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [("retry-after", "5")],
        axum::Json(serde_json::json!({
            "error": {
                "message": "Server overloaded: in-flight token budget exceeded. Retry after a short delay.",
                "type": "server_overloaded"
            }
        })),
    )
        .into_response()
}

/// Build a standard OpenAI error response
pub fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    error_response_with_code(status, message, error_type, None)
}

/// Build an OpenAI error response that also carries the machine-readable
/// `code`, such as `model_not_found`. A `None` code is left out.
pub fn error_response_with_code(
    status: StatusCode,
    message: &str,
    error_type: &str,
    code: Option<&str>,
) -> Response {
    (
        status,
        axum::Json(ErrorResponse {
            error: ErrorDetail {
                message: message.to_string(),
                r#type: error_type.to_string(),
                code: code.map(str::to_string),
            },
        }),
    )
        .into_response()
}

/// Build the error response for a tokenizer special-token policy refusal.
///
/// A refusal means the request text spelled out a control token the server did
/// not itself insert there — a malformed (or hostile) request, so 400 rather
/// than 500. The message carries splintr's token and byte offset, which is what
/// a legitimate caller who tripped the check needs to locate the offending text.
pub fn policy_error_response(err: &splintr::PolicyError) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        &format!("Rejected input: {}", err),
        "invalid_request_error",
    )
}

/// Build the error response for a chat prompt that did not assemble.
///
/// A template that refuses the conversation (`raise_exception`) or text that
/// spells a control token is a request error: 400, with the reason.
pub fn chat_prompt_error_response(err: &crate::model::chat_template::ChatPromptError) -> Response {
    error_response(
        StatusCode::BAD_REQUEST,
        &format!("Rejected input: {}", err),
        "invalid_request_error",
    )
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_response_omits_absent_code() {
        let body = serde_json::to_value(ErrorResponse {
            error: ErrorDetail {
                message: "m".to_string(),
                r#type: "t".to_string(),
                code: None,
            },
        })
        .expect("serializes");
        assert!(body["error"].get("code").is_none());
    }

    #[test]
    fn error_response_carries_status() {
        let resp = error_response_with_code(
            StatusCode::NOT_FOUND,
            "missing",
            "invalid_request_error",
            Some("model_not_found"),
        );
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            overloaded_response().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
