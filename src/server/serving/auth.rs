//! API key auth: the `Authorization: Bearer <key>` check on protected routes.

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use subtle::{Choice, ConstantTimeEq};

/// The valid API keys, in request extensions. Absent when auth is off.
#[derive(Clone)]
struct ApiKeys(Vec<String>);

/// Whether `token` equals one of `keys`. Every key is compared in full, in
/// constant time per byte, so the time taken does not tell how much of a
/// key a guess matched.
fn key_matches(keys: &[String], token: &str) -> bool {
    let token = token.as_bytes();
    let found = keys.iter().fold(Choice::from(0), |found, key| {
        found | key.as_bytes().ct_eq(token)
    });
    found.into()
}

/// The `401` reply with `message`.
fn unauthorized(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        axum::Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_api_key"
            }
        })),
    )
        .into_response()
}

/// API key auth middleware
pub(super) async fn auth_middleware(request: Request, next: Next) -> Response {
    if let Some(ApiKeys(valid_keys)) = request.extensions().get::<ApiKeys>() {
        if !valid_keys.is_empty() {
            let auth_header = request
                .headers()
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok());
            match auth_header.and_then(|h| h.strip_prefix("Bearer ")) {
                Some(token) if key_matches(valid_keys, token) => {}
                Some(_) => return unauthorized("Invalid API key"),
                None => {
                    return unauthorized(
                        "Missing Authorization header. Use: Authorization: Bearer <api-key>",
                    )
                }
            }
        }
    }
    next.run(request).await
}

/// Middleware layer that injects API keys into request extensions
pub(super) async fn inject_api_keys(
    axum::extract::State(keys): axum::extract::State<Vec<String>>,
    mut request: Request,
    next: Next,
) -> Response {
    if !keys.is_empty() {
        request.extensions_mut().insert(ApiKeys(keys));
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn only_a_whole_key_matches() {
        let valid = keys(&["alpha-key", "beta-key"]);
        assert!(key_matches(&valid, "alpha-key"));
        assert!(key_matches(&valid, "beta-key"));
        for wrong in ["", "alpha", "alpha-key2", "ALPHA-KEY", "beta-kez"] {
            assert!(!key_matches(&valid, wrong), "{wrong}");
        }
        assert!(!key_matches(&[], "alpha-key"));
    }
}
