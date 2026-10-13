//! `/v1/systemone` errors: `{"error": {"message", "type", "code"}}` with
//! the status each failure kind takes.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::engine::decision::DecisionFailure;

/// The status, `type` and `code` of a failure.
fn classify(failure: &DecisionFailure) -> (StatusCode, &'static str, &'static str) {
    match failure {
        DecisionFailure::ModelNotFound(_) => (
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "model_not_found",
        ),
        DecisionFailure::NotDecisionModel(_) => (
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "not_a_decision_model",
        ),
        DecisionFailure::InvalidRequest(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request_error",
            "invalid_request",
        ),
        DecisionFailure::TooLarge(_) => (
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            "request_too_large",
        ),
        DecisionFailure::Busy(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "server_overloaded",
            "queue_full",
        ),
        DecisionFailure::Unavailable(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "service_unavailable",
            "model_unavailable",
        ),
        DecisionFailure::Internal(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "internal_error",
        ),
    }
}

/// The response of `failure`. A full queue also says when to retry.
pub fn failure_response(failure: &DecisionFailure) -> Response {
    let (status, kind, code) = classify(failure);
    let body = axum::Json(json!({
        "error": {"message": failure.message(), "type": kind, "code": code}
    }));
    match failure {
        DecisionFailure::Busy(_) => (status, [("retry-after", "1")], body).into_response(),
        _ => (status, body).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_kind_has_its_status_and_code() {
        let cases = [
            (
                DecisionFailure::ModelNotFound("m".into()),
                404,
                "model_not_found",
            ),
            (
                DecisionFailure::NotDecisionModel("m".into()),
                400,
                "not_a_decision_model",
            ),
            (
                DecisionFailure::InvalidRequest("m".into()),
                422,
                "invalid_request",
            ),
            (
                DecisionFailure::TooLarge("m".into()),
                413,
                "request_too_large",
            ),
            (DecisionFailure::Busy("m".into()), 503, "queue_full"),
            (
                DecisionFailure::Unavailable("m".into()),
                503,
                "model_unavailable",
            ),
            (DecisionFailure::Internal("m".into()), 500, "internal_error"),
        ];
        for (failure, status, code) in cases {
            let (got_status, _, got_code) = classify(&failure);
            assert_eq!((got_status.as_u16(), got_code), (status, code));
            let response = failure_response(&failure);
            assert_eq!(response.status().as_u16(), status);
        }
        let busy = failure_response(&DecisionFailure::Busy("full".into()));
        assert_eq!(
            busy.headers().get("retry-after").map(|v| v.as_bytes()),
            Some(&b"1"[..])
        );
    }
}
