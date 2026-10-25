//! `server.max_concurrent_requests`: the HTTP request limit.
//!
//! A request over the limit waits for a free slot. Each waiting request
//! counts in `blazr_queue_depth`. `/health` and `/metrics` skip the limit,
//! so probes and scrapes answer while the server is saturated.
//!
//! A request's slot is a [`WorkHold`]. A streamed body holds it until the
//! body drops. The work the request starts holds it too: a decode thread
//! until it exits, and a batched request until its scheduler frees it. A
//! client that leaves therefore frees its slot only once its decode stops,
//! so the limit bounds the decodes in flight as well as the open responses.

use std::sync::Arc;

use axum::body::{Body, HttpBody};
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;
use futures::StreamExt;
use tokio::sync::Semaphore;

use crate::engine::{QueueDepth, WorkHold};

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

/// Middleware: hold one slot for the whole request, a streamed body and
/// the work the request starts included.
pub async fn limit_concurrency(
    State(limit): State<Arc<ConcurrencyLimit>>,
    request: Request,
    next: Next,
) -> Response {
    if EXEMPT_PATHS.contains(&request.uri().path()) {
        return next.run(request).await;
    }
    // The semaphore is never closed, so the acquire always yields a permit.
    let permit = {
        // The guard leaves the queue when the wait ends or the client leaves.
        let _waiting = limit.waiting.enter();
        Arc::clone(&limit.slots).acquire_owned().await.ok()
    };
    let hold = permit.map(WorkHold::new);
    let response = WorkHold::within(hold.clone(), next.run(request)).await;
    match hold {
        Some(hold) => hold_for_body(response, hold),
        None => response,
    }
}

/// A body of unknown length streams after the handler returns. The body
/// keeps a clone of the slot's hold. A body of known length is complete,
/// so its clone drops here. Work the request started keeps its own clones.
fn hold_for_body(response: Response, hold: WorkHold) -> Response {
    if response.body().size_hint().exact().is_some() {
        return response;
    }
    response.map(|body| {
        Body::from_stream(body.into_data_stream().map(move |chunk| {
            let _slot = &hold;
            chunk
        }))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// A streamed body keeps its slot until the body drops. A sized body
    /// gives its slot back when the handler returns.
    #[tokio::test]
    async fn a_streamed_body_holds_its_slot_until_it_drops() -> anyhow::Result<()> {
        let limit = ConcurrencyLimit::new(1);
        let open = || futures::stream::pending::<Result<axum::body::Bytes, std::io::Error>>();
        let app = Router::new()
            .route(
                "/stream",
                get(move || async move { Body::from_stream(open()) }),
            )
            .route("/sized", get(|| async { "done" }))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&limit),
                limit_concurrency,
            ));
        let call = |path: &'static str| {
            let app = app.clone();
            async move {
                let request = Request::builder().uri(path).body(Body::empty())?;
                anyhow::Ok(app.oneshot(request).await?)
            }
        };

        let sized = call("/sized").await?;
        assert_eq!(limit.slots.available_permits(), 1);
        drop(sized);
        let streamed = call("/stream").await?;
        assert_eq!(
            limit.slots.available_permits(),
            0,
            "the open stream holds its slot"
        );
        drop(streamed);
        assert_eq!(limit.slots.available_permits(), 1);
        Ok(())
    }

    /// The client leaves while the request's work still runs, as a decode
    /// runs to its next step. The slot stays taken until that work ends.
    #[tokio::test]
    async fn work_the_request_started_holds_its_slot_past_the_body() -> anyhow::Result<()> {
        let limit = ConcurrencyLimit::new(1);
        let release = Arc::new(Notify::new());
        let gate = Arc::clone(&release);
        let open = || futures::stream::pending::<Result<axum::body::Bytes, std::io::Error>>();
        let app = Router::new()
            .route(
                "/stream",
                get(move || {
                    let gate = Arc::clone(&gate);
                    async move {
                        let hold = WorkHold::current();
                        std::thread::spawn(move || {
                            let _hold = hold;
                            futures::executor::block_on(gate.notified());
                        });
                        Body::from_stream(open())
                    }
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&limit),
                limit_concurrency,
            ));

        let request = Request::builder().uri("/stream").body(Body::empty())?;
        let streamed = app.oneshot(request).await?;
        drop(streamed);
        assert_eq!(
            limit.slots.available_permits(),
            0,
            "the running work holds the slot after the client left"
        );

        release.notify_one();
        for _ in 0..200 {
            if limit.slots.available_permits() == 1 {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        Err(anyhow::anyhow!(
            "the slot stayed taken after the work ended"
        ))
    }

    /// The batch engine thread panics during a streamed request's prefill.
    /// The stream ends with one error event and `[DONE]`, the slot frees,
    /// `/health` answers 503, and the next request is refused.
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_panicked_engine_ends_its_streams_and_frees_their_slots() -> anyhow::Result<()> {
        use crate::engine::batch_test_rig::{rig_over, Rig, POOL};
        use crate::engine::test_support::{seeded_full_vocab_paged, FULL_VOCAB_CTX};
        use crate::engine::{
            EngineThread, FailRequests, GenerationScheduler, Scheduler, ThreadedScheduler,
        };
        use crate::server::handlers::{health, ServeReplica, Serving};
        use crate::server::routes::api_routes;
        use crate::server::streaming::sse_test::parse_events;
        use serde_json::{json, Value};

        const MODEL: &str = "m";
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join(MODEL))?;
        std::fs::write(dir.path().join(MODEL).join("config.json"), "{}")?;
        let models = Scheduler::<crate::runtime::ServerRuntime>::new(
            dir.path().to_path_buf(),
            boostr::CpuDevice::new(),
        )
        .pinned();
        let executor = seeded_full_vocab_paged(0x5eed, POOL)?;
        let served = models
            .register_base(MODEL, dir.path().join(MODEL), executor)
            .await;
        let Rig {
            scheduler, engine, ..
        } = rig_over(Arc::clone(&served), POOL, FULL_VOCAB_CTX)?;
        let mut engine = engine.with_prefill_chunk_size(1);
        engine.after_chunk = Some(Box::new(|_: usize| panic!("injected engine panic")));
        let owner: Arc<dyn FailRequests> = scheduler.clone();
        let thread = EngineThread::spawn("blazr-batch", owner, move |stop| engine.run_until(stop))?;
        let batching: Arc<dyn GenerationScheduler> =
            Arc::new(ThreadedScheduler::new(scheduler, thread));
        let serving = Serving::Single(ServeReplica {
            executor: served,
            scheduler: Some(batching),
            device: "cpu".to_string(),
        });
        let metrics = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let state = crate::server::AppState::new(Arc::new(models), metrics).with_serving(serving);
        let limit = ConcurrencyLimit::new(2);
        let app = api_routes()
            .route("/health", get(health))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&limit),
                limit_concurrency,
            ))
            .with_state(Arc::new(state));
        let call = |method: &str, path: &str, body: Option<Value>| {
            let app = app.clone();
            let request = Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json")
                .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())));
            async move {
                let response = app.oneshot(request?).await?;
                let status = response.status().as_u16();
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
                anyhow::Ok((status, String::from_utf8(bytes.to_vec())?))
            }
        };

        let body =
            json!({"model": MODEL, "prompt": "Hello there", "max_tokens": 4, "stream": true});
        let (status, raw) = call("POST", "/v1/completions", Some(body.clone())).await?;
        assert_eq!(status, 200, "{raw}");
        let events = parse_events(&raw)?;
        let [.., error, done] = events.as_slice() else {
            return Err(anyhow::anyhow!("the stream sent too few events: {raw}"));
        };
        let message = error["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains("injected engine panic"), "{raw}");
        assert_eq!(*done, json!("[DONE]"));
        assert_eq!(
            events.iter().filter(|e| e.get("error").is_some()).count(),
            1
        );
        let mut freed = false;
        for _ in 0..1000 {
            if limit.slots.available_permits() == 2 {
                freed = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        assert!(freed, "the failed request kept its slot");

        let (status, raw) = call("GET", "/health", None).await?;
        assert_eq!(status, 503, "{raw}");
        assert!(
            raw.contains("\"unavailable\"") && raw.contains("panicked"),
            "{raw}"
        );

        let mut plain = body;
        plain["stream"] = json!(false);
        let (status, raw) = call("POST", "/v1/completions", Some(plain)).await?;
        assert_eq!(status, 500, "{raw}");
        assert!(raw.contains("the engine stopped"), "{raw}");
        assert_eq!(limit.slots.available_permits(), 2);
        Ok(())
    }

    /// A finished reply and a finished stream each give their slot back:
    /// the decode thread and the spawned stream task drop their holds.
    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    #[tokio::test]
    async fn a_finished_generation_gives_its_slot_back() -> anyhow::Result<()> {
        use crate::server::ollama_api::e2e_rig::{Rig, MODEL};
        use crate::server::routes::api_routes;
        use serde_json::json;

        {
            let rig = Rig::new().await?;
            let limit = ConcurrencyLimit::new(1);
            let app = api_routes()
                .layer(axum::middleware::from_fn_with_state(
                    Arc::clone(&limit),
                    limit_concurrency,
                ))
                .with_state(Arc::clone(&rig.state));
            let chat = json!({"messages": [{"role": "user", "content": "Hi"}]});
            let text = json!({"prompt": "Hi"});
            for (path, fields) in [("/v1/chat/completions", chat), ("/v1/completions", text)] {
                for stream in [false, true] {
                    let mut body = json!({"model": MODEL, "max_tokens": 3, "stream": stream});
                    if let (Some(body), Some(fields)) = (body.as_object_mut(), fields.as_object()) {
                        body.extend(fields.clone());
                    }
                    let request = Request::post(path)
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))?;
                    let response = app.clone().oneshot(request).await?;
                    assert_eq!(response.status(), axum::http::StatusCode::OK, "{path}");
                    axum::body::to_bytes(response.into_body(), usize::MAX).await?;
                    let mut freed = false;
                    for _ in 0..1000 {
                        if limit.slots.available_permits() == 1 {
                            freed = true;
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    }
                    assert!(freed, "{path} stream={stream} kept its slot");
                }
            }
        }
        Ok(())
    }
}
