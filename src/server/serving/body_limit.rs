//! `server.max_body_size`: the one request body limit every route and every
//! extractor reads.
//!
//! axum's extractors (`Json`, `Bytes`, `Multipart`, ...) cap a body at 2 MB
//! unless a [`DefaultBodyLimit`] says otherwise. [`with_body_limit`] sets that
//! cap to the configured limit, so the configured limit is the real one. A
//! body over the limit answers `413` in the OpenAI error shape, naming the
//! limit and the key that sets it, whether the request declared its length
//! or streamed it.

use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{
    header::{CONTENT_LENGTH, CONTENT_TYPE},
    StatusCode,
};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::Router;
use tower_http::limit::RequestBodyLimitLayer;

use crate::server::gen_types::error_response_with_code;

/// The configured body limit in bytes, in each request's extensions, so a
/// handler that reads the body itself can name it in a `413`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BodyLimit(pub usize);

/// The `code` of every body-limit `413`.
pub const BODY_TOO_LARGE_CODE: &str = "request_too_large";

/// The `413` for a body over `limit` bytes. `len` is the declared length,
/// when the request sent one.
pub fn body_too_large(len: Option<usize>, limit: Option<usize>) -> Response {
    let size = match len {
        Some(n) => format!("Request body is {n} bytes, over"),
        None => "Request body is over".to_string(),
    };
    let max = match limit {
        Some(n) => format!("the server limit of {n} bytes"),
        None => "the server limit".to_string(),
    };
    error_response_with_code(
        StatusCode::PAYLOAD_TOO_LARGE,
        &format!(
            "{size} {max} (`server.max_body_size`). Send a smaller body, or raise \
             `server.max_body_size` in the server config."
        ),
        "invalid_request_error",
        Some(BODY_TOO_LARGE_CODE),
    )
}

/// Applies a `limit`-byte body limit to `router`:
/// - a declared `Content-Length` over `limit` answers [`body_too_large`]
///   before any handler runs,
/// - a streamed body is cut at `limit` bytes, and the plain-text `413` an
///   extractor answers for it becomes [`body_too_large`],
/// - every axum extractor reads up to `limit` bytes, not its 2 MB default.
pub fn with_body_limit<S>(router: Router<S>, limit: usize) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router
        .layer(DefaultBodyLimit::max(limit))
        .layer(RequestBodyLimitLayer::new(limit))
        .layer(middleware::from_fn_with_state(
            BodyLimit(limit),
            reject_declared_oversize,
        ))
}

/// Answers [`body_too_large`] for a declared length over the limit, and
/// stores the limit in the request's extensions otherwise.
async fn reject_declared_oversize(
    State(limit): State<BodyLimit>,
    mut request: Request,
    next: Next,
) -> Response {
    let declared = request
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok()?.parse::<usize>().ok());
    if let Some(len) = declared.filter(|&len| len > limit.0) {
        return body_too_large(Some(len), Some(limit.0));
    }
    request.extensions_mut().insert(limit);
    let response = next.run(request).await;
    if is_plain_too_large(&response) {
        return body_too_large(None, Some(limit.0));
    }
    response
}

/// A `413` without a JSON body: the one an extractor (`Json`, `Bytes`,
/// `Form`, ...) or the limit layer answers when a streamed body runs over
/// the limit. A handler's own JSON `413` passes through.
fn is_plain_too_large(response: &Response) -> bool {
    response.status() == StatusCode::PAYLOAD_TOO_LARGE
        && !response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("application/json"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, Bytes};
    use axum::routing::post;
    use axum::Json;
    use tower::ServiceExt;

    const LIMIT: usize = 4 * 1024 * 1024;

    fn app() -> Router {
        let router = Router::new()
            .route("/echo", post(|b: Bytes| async move { b.len().to_string() }))
            .route(
                "/json",
                post(|Json(v): Json<serde_json::Value>| async move { v.to_string() }),
            )
            .route(
                "/own413",
                post(|| async {
                    error_response_with_code(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        "handler's own",
                        "invalid_request_error",
                        None,
                    )
                }),
            );
        with_body_limit(router, LIMIT)
    }

    /// `body` sent in 1 MB chunks, with no declared length.
    fn streamed(body: Vec<u8>) -> Body {
        let chunks: Vec<Result<Vec<u8>, std::io::Error>> =
            body.chunks(1 << 20).map(|c| Ok(c.to_vec())).collect();
        Body::from_stream(futures::stream::iter(chunks))
    }

    fn assert_names_limit(text: &str) {
        assert!(text.contains(&format!("limit of {LIMIT} bytes")), "{text}");
        assert!(text.contains("server.max_body_size"), "{text}");
        assert!(text.contains(BODY_TOO_LARGE_CODE), "{text}");
    }

    async fn body_text(res: Response) -> String {
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap_or_default();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    #[tokio::test]
    async fn body_over_axum_default_under_limit_is_read() {
        let n = 3 * 1024 * 1024;
        let req = Request::post("/echo")
            .header(CONTENT_LENGTH, n)
            .body(Body::from(vec![0u8; n]))
            .expect("request");
        let res = app().oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(body_text(res).await, n.to_string());
    }

    #[tokio::test]
    async fn declared_length_over_limit_names_the_limit() {
        let n = LIMIT + 1;
        let req = Request::post("/echo")
            .header(CONTENT_LENGTH, n)
            .body(Body::from(vec![0u8; n]))
            .expect("request");
        let res = app().oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let text = body_text(res).await;
        assert!(text.contains(&format!("{n} bytes")), "{text}");
        assert!(text.contains(&format!("limit of {LIMIT} bytes")), "{text}");
        assert!(text.contains("server.max_body_size"), "{text}");
        assert!(text.contains(BODY_TOO_LARGE_CODE), "{text}");
    }

    #[tokio::test]
    async fn streamed_body_over_limit_names_the_limit() {
        let req = Request::post("/echo")
            .body(streamed(vec![0u8; LIMIT + 1]))
            .expect("request");
        let res = app().oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_names_limit(&body_text(res).await);
    }

    #[tokio::test]
    async fn streamed_json_over_limit_names_the_limit() {
        let req = Request::post("/json")
            .header(CONTENT_TYPE, "application/json")
            .body(streamed(vec![b' '; LIMIT + 1]))
            .expect("request");
        let res = app().oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            res.headers()
                .get(CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        assert_names_limit(&body_text(res).await);
    }

    #[tokio::test]
    async fn streamed_json_over_two_megabytes_under_limit_is_read() {
        let mut body = vec![b' '; 3 * 1024 * 1024];
        body[0] = b'7';
        let req = Request::post("/json")
            .header(CONTENT_TYPE, "application/json")
            .body(streamed(body))
            .expect("request");
        let res = app().oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(body_text(res).await, "7");
    }

    #[tokio::test]
    async fn handler_json_413_passes_through() {
        let req = Request::post("/own413")
            .body(Body::empty())
            .expect("request");
        let res = app().oneshot(req).await.expect("response");
        assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(body_text(res).await.contains("handler's own"));
    }
}
