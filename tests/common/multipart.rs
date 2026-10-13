//! A hand-rolled multipart body encoder, and a POST of one through a router.

#[cfg(feature = "server")]
use axum::body::Body;
#[cfg(feature = "server")]
use axum::http::{Request, StatusCode};
#[cfg(feature = "server")]
use axum::Router;
#[cfg(feature = "server")]
use tower::ServiceExt;

/// One multipart/form-data body: `fields` in call order (repeat a name for a
/// repeated field, e.g. two `("timestamp_granularities[]", "word")` /
/// `("timestamp_granularities[]", "segment")` entries), then one `file` part
/// as `(field_name, filename, content_type, bytes)`.
pub fn multipart_body(
    boundary: &str,
    fields: &[(&str, &str)],
    file: (&str, &str, &str, &[u8]),
) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    let (field_name, filename, content_type, bytes) = file;
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(bytes);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

/// POST an already-encoded multipart body to `uri` on `router` and return the
/// status plus the response parsed as JSON. Panics if the body is not JSON —
/// callers that expect `text`/`srt`/`vtt` read the router response directly
/// instead of going through this helper.
#[cfg(feature = "server")]
pub async fn post_multipart_json(
    router: Router,
    uri: &str,
    boundary: &str,
    body: Vec<u8>,
) -> (StatusCode, serde_json::Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .expect("building multipart request");
    let response = router.oneshot(request).await.expect("router call failed");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("reading response body");
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "response body is not JSON: {e}; body = {:?}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, json)
}
