//! Newline-delimited JSON responses, the Ollama streaming format.

use std::convert::Infallible;

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, StatusCode},
    response::{IntoResponse, Response},
};
use futures::stream::{Stream, StreamExt};
use serde_json::{json, Value};

/// The media type of an NDJSON body.
pub(in crate::server) const NDJSON: &str = "application/x-ndjson";

/// `value` as one NDJSON line.
pub(in crate::server) fn ndjson_line(value: &Value) -> String {
    format!("{value}\n")
}

/// A `200` NDJSON response with one line per item of `lines`.
///
/// An `Err(message)` item sends the line `{"error": message}` and ends
/// the body.
pub(in crate::server) fn ndjson_response<S>(lines: S) -> Response
where
    S: Stream<Item = Result<Value, String>> + Send + 'static,
{
    let body = lines.scan(false, |ended, item| {
        let line = match item {
            _ if *ended => None,
            Ok(value) => Some(ndjson_line(&value)),
            Err(message) => {
                *ended = true;
                Some(ndjson_line(&json!({ "error": message })))
            }
        };
        futures::future::ready(line.map(Ok::<_, Infallible>))
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, NDJSON)
        .body(Body::from_stream(body))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

/// Helpers that read an NDJSON response back, for the stream tests.
#[cfg(test)]
pub(in crate::server) mod ndjson_test {
    use axum::response::Response;

    /// The raw body of `response`.
    pub(in crate::server) async fn body(response: Response) -> anyhow::Result<String> {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
        Ok(String::from_utf8(bytes.to_vec())?)
    }
}

#[cfg(test)]
mod tests {
    use super::ndjson_test::body;
    use super::*;

    #[tokio::test]
    async fn each_item_is_one_line_and_an_error_ends_the_body() -> anyhow::Result<()> {
        let items = vec![
            Ok(json!({"a": 1})),
            Err("boom".to_string()),
            Ok(json!({"never": true})),
        ];
        let response = ndjson_response(futures::stream::iter(items));
        assert_eq!(response.headers()[CONTENT_TYPE], NDJSON);
        assert_eq!(body(response).await?, "{\"a\":1}\n{\"error\":\"boom\"}\n");
        Ok(())
    }
}
