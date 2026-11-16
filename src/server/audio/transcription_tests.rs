//! `/v1/audio/transcriptions` form reading against the server body limit.
//! No model loads: a form read in full reaches the model lookup, which
//! answers `404` for the unregistered model.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::routing::post;
use axum::Router;
use tower::ServiceExt;

use super::transcription::transcriptions;
use crate::server::body_limit::with_body_limit;
use crate::server::handlers::test_state;

const LIMIT: usize = 8 * 1024 * 1024;
const BOUNDARY: &str = "blazr-test-boundary";

fn app(dir: &std::path::Path) -> Router {
    let router = Router::new()
        .route("/v1/audio/transcriptions", post(transcriptions))
        .with_state(Arc::new(test_state(dir)));
    with_body_limit(router, LIMIT)
}

/// A form with a `file` part of `file_len` bytes before its `model` part,
/// as curl sends `-F file=@... -F model=...`.
fn form(file_len: usize) -> Vec<u8> {
    let mut body = Vec::with_capacity(file_len + 512);
    body.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; \
             filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.resize(body.len() + file_len, 0);
    body.extend_from_slice(
        format!(
            "\r\n--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n\
             absent-whisper\r\n--{BOUNDARY}--\r\n"
        )
        .as_bytes(),
    );
    body
}

fn request(body: Body, len: Option<usize>) -> Request<Body> {
    let mut req = Request::post("/v1/audio/transcriptions").header(
        header::CONTENT_TYPE,
        format!("multipart/form-data; boundary={BOUNDARY}"),
    );
    if let Some(n) = len {
        req = req.header(header::CONTENT_LENGTH, n);
    }
    req.body(body).expect("request")
}

async fn send(dir: &std::path::Path, req: Request<Body>) -> (StatusCode, String) {
    let res = app(dir).oneshot(req).await.expect("response");
    let status = res.status();
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
        .await
        .unwrap_or_default();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn upload_over_two_megabytes_under_the_limit_is_read() -> std::io::Result<()> {
    let dir = tempfile::tempdir()?;
    let body = form(3 * 1024 * 1024);
    let len = body.len();
    let (status, text) = send(dir.path(), request(Body::from(body), Some(len))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{text}");
    assert!(
        text.contains("ASR model 'absent-whisper' is not loaded"),
        "{text}"
    );
    Ok(())
}

#[tokio::test]
async fn declared_upload_over_the_limit_names_the_limit() -> std::io::Result<()> {
    let dir = tempfile::tempdir()?;
    let body = form(LIMIT);
    let len = body.len();
    let (status, text) = send(dir.path(), request(Body::from(body), Some(len))).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{text}");
    assert!(text.contains(&format!("limit of {LIMIT} bytes")), "{text}");
    assert!(text.contains("server.max_body_size"), "{text}");
    assert!(!text.contains("'model' is required"), "{text}");
    Ok(())
}

#[tokio::test]
async fn streamed_upload_over_the_limit_names_the_limit() -> std::io::Result<()> {
    let dir = tempfile::tempdir()?;
    let body = form(LIMIT);
    let chunks: Vec<Result<Vec<u8>, std::io::Error>> =
        body.chunks(1024 * 1024).map(|c| Ok(c.to_vec())).collect();
    let stream = Body::from_stream(futures::stream::iter(chunks));
    let (status, text) = send(dir.path(), request(stream, None)).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{text}");
    assert!(text.contains(&format!("limit of {LIMIT} bytes")), "{text}");
    assert!(!text.contains("'model' is required"), "{text}");
    Ok(())
}

#[tokio::test]
async fn truncated_form_names_the_read_error() -> std::io::Result<()> {
    let dir = tempfile::tempdir()?;
    let mut body = form(1024);
    body.truncate(600);
    let len = body.len();
    let (status, text) = send(dir.path(), request(Body::from(body), Some(len))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("Reading the multipart form failed"), "{text}");
    Ok(())
}

#[tokio::test]
async fn temperature_that_is_not_a_number_is_refused() -> std::io::Result<()> {
    let dir = tempfile::tempdir()?;
    let body = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"temperature\"\r\n\r\nwarm\r\n\
         --{BOUNDARY}--\r\n"
    )
    .into_bytes();
    let len = body.len();
    let (status, text) = send(dir.path(), request(Body::from(body), Some(len))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("'temperature' is 'warm'"), "{text}");
    Ok(())
}
