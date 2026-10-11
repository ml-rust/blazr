//! `server.max_concurrent_requests`: the HTTP request limit.
//!
//! A request over the limit waits for a free slot. Each waiting request
//! counts in `blazr_queue_depth`. `/health` and `/metrics` skip the limit,
//! so probes and scrapes answer while the server is saturated.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;
use tokio::sync::Semaphore;

use crate::engine::QueueDepth;

/// Paths that never wait for a slot.
const EXEMPT_PATHS: [&str; 2] = ["/health", "/metrics"];

/// Request slots and the count of requests that wait for one.
pub struct ConcurrencyLimit {
    slots: Arc<Semaphore>,
    waiting: QueueDepth,
}

impl ConcurrencyLimit {
    /// A limit of `max` requests in flight. 0 is treated as 1.
    pub fn new(max: usize) -> Arc<Self> {
        Arc::new(Self {
            slots: Arc::new(Semaphore::new(max.max(1))),
            waiting: QueueDepth::new(),
        })
    }

    /// Requests waiting for a slot.
    #[cfg(test)]
    fn waiting(&self) -> usize {
        self.waiting.get()
    }
}

/// Middleware: hold one slot for the whole request.
pub async fn limit_concurrency(
    State(limit): State<Arc<ConcurrencyLimit>>,
    request: Request,
    next: Next,
) -> Response {
    if EXEMPT_PATHS.contains(&request.uri().path()) {
        return next.run(request).await;
    }
    // The semaphore is never closed, so the acquire always yields a permit.
    let _permit = {
        // The guard leaves the queue when the wait ends or the client leaves.
        let _waiting = limit.waiting.enter();
        Arc::clone(&limit.slots).acquire_owned().await.ok()
    };
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::routing::get;
    use axum::Router;
    use tokio::sync::Notify;
    use tower::ServiceExt;

    use crate::server::metrics::names::QUEUE_DEPTH;
    use crate::server::test_recorder::TestRecorder;

    #[tokio::test]
    async fn a_request_over_the_limit_waits_and_counts_in_the_queue_depth() {
        let recorder = TestRecorder::install();
        let limit = ConcurrencyLimit::new(1);
        let release = Arc::new(Notify::new());
        let gate = Arc::clone(&release);
        let app = Router::new()
            .route(
                "/slow",
                get(move || {
                    let gate = Arc::clone(&gate);
                    async move {
                        gate.notified().await;
                        "done"
                    }
                }),
            )
            .route("/health", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&limit),
                limit_concurrency,
            ));

        let call = |path: &'static str| {
            let app = app.clone();
            async move {
                let request = Request::builder().uri(path).body(Body::empty());
                app.oneshot(request.expect("request")).await
            }
        };
        let first = tokio::spawn(call("/slow"));
        let second = tokio::spawn(call("/slow"));
        for _ in 0..200 {
            if limit.waiting() == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(limit.waiting(), 1);
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(1.0));

        // A probe skips the queue while both slots are taken.
        let health = call("/health").await.expect("health");
        assert_eq!(health.status(), axum::http::StatusCode::OK);

        release.notify_one();
        let _ = first.await;
        release.notify_one();
        let _ = second.await;
        assert_eq!(limit.waiting(), 0);
        assert_eq!(recorder.value(QUEUE_DEPTH, &[]), Some(0.0));
    }
}
