//! `POST /v1/systemone` end to end on Kev: the Kev directory served by its
//! name through the model scheduler, which loads the base, attaches the
//! adapter unmerged as a variant and reads the pointer head from
//! `head.pt`; the axum router in-process; every fixture's request posted
//! and its answers compared with Kev's own wire answers.
//!
//! # Fixtures
//!
//! - `BLAZR_DECISION_MODEL_DIR`: the Kev checkpoint directory. Its parent
//!   is the model directory and its name the model name.
//! - `BLAZR_DECISION_BASE`: the Qwen3.5 base checkpoint, passed as
//!   `inference.decision.base`. Loaded on the CPU, every weight F32.
//! - `BLAZR_DECISION_FIXTURE_DIR`: the `kev_*.json` fixtures, schema
//!   `boostr.decision.pointer.v1`.
//!
//! Unset or missing, the test prints one `skip:` line and passes. It runs
//! in a CPU build only: the fixtures are F32 CPU references.
//!
//! # Tolerance
//!
//! Probability band 1e-2 per probability, carried through each formula in
//! `tests/decision_kev/compare.rs`, plus one unit of Kev's 4-digit
//! rounding. Both sides run F32 on the CPU with the adapter unmerged and
//! differ only in the order of float accumulation (matmul blocking, the
//! GDN chunk scan): boostr's `decision_pointer_real_file` bounds the raw
//! pointer logits to 5e-2 of Kev's, and at the fitted temperature (about
//! 2.4) a logit drift `d` moves a probability at most `d / (4T)`, about
//! 5e-3, inside the band. Raw answers (`"calibration": "raw"`, temperature
//! 1) take `5e-2 / 4` against `softmax(raw_logits)`.
//!
//! A merged run must match the same request run alone within one rounding
//! unit: the rows only change lanes.

#![cfg(all(feature = "server", not(any(feature = "cuda", feature = "wgpu"))))]

mod decision_kev;

use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use blazr::boostr::DType;
use blazr::config::{DecisionConfig, DtypeRequest, DtypeSource};
use blazr::engine::decision::{DecisionService, Pending};
use blazr::engine::Scheduler;
use blazr::loader::decision::read_kev_adapter;
use blazr::model::decision_template::{Family, JevRequest};
use blazr::server::handlers::{AppState, ServerRuntime};
use serde_json::{json, Map, Value};
use tower::ServiceExt;

use decision_kev::compare;
use decision_kev::fixture::{env_path, model_and_fixtures, Fixture, ENV_BASE};

const ROUTE: &str = "/v1/systemone";
const MAX_ROWS: usize = 64;
const MAX_QUEUE_ROWS: usize = 8;
const CPU_F32_BAND: f64 = 1e-2;
const RAW_BAND: f64 = 5e-2 / 4.0;
const MERGED_BAND: f64 = 0.0;

struct Served {
    router: Router,
    state: Arc<AppState>,
    name: String,
}

/// The Kev directory `dir` served by name from its parent directory, with
/// `base` as `inference.decision.base`, every weight F32. The prefix-state
/// cache is off, so one-row requests merge as they would without one.
fn serve(dir: &Path, base: &Path) -> Served {
    let parent = dir.parent().expect("the Kev directory has a parent");
    let name = dir
        .file_name()
        .expect("the Kev directory has a name")
        .to_string_lossy()
        .into_owned();
    let device = blazr::runtime::open_device(0).expect("device 0 opens");
    let decision = DecisionConfig {
        base: Some(base.display().to_string()),
        max_rows: MAX_ROWS,
        max_queue_rows: MAX_QUEUE_ROWS,
        prefix_cache_mb: 0,
        start_model: Some(name.clone()),
        ..DecisionConfig::default()
    };
    let scheduler = Scheduler::<ServerRuntime>::new(parent.to_path_buf(), device)
        .with_dtype(Some(DtypeRequest {
            dtype: DType::F32,
            source: DtypeSource::Flag,
        }))
        .with_decision(decision);
    let metrics = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    let state = Arc::new(AppState::new(Arc::new(scheduler), metrics));
    Served {
        router: blazr::server::api_routes().with_state(Arc::clone(&state)),
        state,
        name,
    }
}

async fn send(router: &Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = router.clone().oneshot(request).await.expect("router call");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    let value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!(
            "status {status}: not JSON ({e}): {}",
            String::from_utf8_lossy(&bytes)
        )
    });
    (status, value)
}

async fn post_raw(router: &Router, body: Vec<u8>) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(ROUTE)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .expect("request builds");
    send(router, request).await
}

async fn post(router: &Router, body: &Value) -> (StatusCode, Value) {
    post_raw(router, serde_json::to_vec(body).expect("body encodes")).await
}

async fn get(router: &Router, uri: &str) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .expect("request builds");
    send(router, request).await
}

fn answers_of(body: &Value, what: &str) -> Map<String, Value> {
    body["answers"]
        .as_object()
        .unwrap_or_else(|| panic!("{what}: no answers in {body}"))
        .clone()
}

/// `n` noul questions about `state`: `n` rows.
fn noul_request(n: usize, state: &str) -> Value {
    let questions: Map<String, Value> = (0..n)
        .map(|i| {
            let q = json!({"type": "noul", "instructions": format!("Point {i} needs action."),
                "criteria": {"true": "it does", "false": "it does not"}});
            (format!("q{i}"), q)
        })
        .collect();
    json!({"state": state, "questions": questions})
}

/// A queued request that holds the worker: dropping it cancels it.
fn blocker(service: &DecisionService<ServerRuntime>, body: &Value) -> Pending {
    let request = JevRequest::parse(body, Family::Kev).expect("blocker parses");
    let rendered = service
        .renderer()
        .render(&request)
        .expect("blocker renders");
    let prepared = service.prepare(&rendered, false).expect("blocker prepares");
    assert!(prepared.rows + 2 <= MAX_QUEUE_ROWS && prepared.rows + 5 > MAX_QUEUE_ROWS);
    service.submit(prepared).expect("blocker queues")
}

/// Each fixture's answers, usage and rows.
async fn fixture_checks(served: &Served, fixtures: &[Fixture]) -> Vec<Map<String, Value>> {
    let service = served
        .state
        .scheduler
        .decision_service(&served.name)
        .await
        .unwrap_or_else(|e| panic!("decision service of {}: {e}", served.name));
    assert_eq!(service.family(), Family::Kev);
    let mut solo = Vec::with_capacity(fixtures.len());
    for fx in fixtures {
        let what = fx.label();
        let temps = &service
            .calibration()
            .expect("head.pt sets a temperature")
            .temperatures;
        assert!(
            temps.values().all(|&t| t == fx.temperature),
            "{what}: {temps:?}"
        );
        let request = JevRequest::parse(&fx.request, Family::Kev).expect("parses");
        let rendered = service.renderer().render(&request).expect("renders");
        assert_eq!(
            rendered.rows, fx.rows,
            "{what}: the served tokenizer renders other rows"
        );

        let (status, body) = post(&served.router, &fx.request).await;
        assert_eq!(status, StatusCode::OK, "{what}: {body}");
        let got = answers_of(&body, &what);
        compare::answers(&got, &fx.reference_answers, CPU_F32_BAND, &what);
        assert_eq!(body["model"], json!(served.name), "{what}: model");
        assert_eq!(
            body["usage"]["input_tokens"],
            json!(fx.input_tokens()),
            "{what}"
        );
        assert!(
            body["usage"]["output_tokens"]
                .as_u64()
                .is_some_and(|n| n > 0),
            "{body}"
        );
        assert!(
            body.get("truncated").is_none(),
            "{what}: states are refused, not cut"
        );
        assert!(body["latency_ms"].as_f64().is_some(), "{what}: latency");
        assert_eq!(body["calibration"]["applied"], json!(true), "{what}");
        println!("ok: {what} through {ROUTE}");
        solo.push(got);
    }
    solo
}

/// `/v1/models` reports the Kev decision object.
async fn models_check(served: &Served, dir: &Path, fx: &Fixture) {
    let (status, body) = get(&served.router, "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let data = body["data"].as_array().expect("a data list");
    let listed: Vec<&Value> = data
        .iter()
        .filter(|m| m["id"] == json!(served.name))
        .collect();
    assert_eq!(listed.len(), 1, "listed once: {body}");
    let d = &listed[0]["decision"];
    assert_eq!(d["family"], json!("kev"), "{d}");
    assert_eq!(d["readout"], json!("pointer"), "{d}");
    assert_eq!(d["temperature"], json!(fx.temperature), "{d}");
    let rank = read_kev_adapter(dir).expect("adapter reads").rank;
    assert_eq!(d["adapter_rank"], json!(rank), "{d}");
    assert_eq!(d["max_state_tokens"], json!(65536), "{d}");
    assert_eq!(d["max_prompt_tokens"], json!(65536 + 8192), "{d}");
    assert!(d["base_path"].as_str().is_some(), "{d}");
    assert!(d["context_length"].as_u64().is_some(), "{d}");
    assert!(
        listed[0]["lora"].is_object(),
        "the variant of its base: {}",
        listed[0]
    );
    println!("ok: /v1/models");
}

/// `"calibration": "raw"` answers at temperature 1.
async fn raw_check(served: &Served, fx: &Fixture) {
    let mut raw = fx.request.clone();
    raw["calibration"] = json!("raw");
    let (status, body) = post(&served.router, &raw).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["calibration"]["applied"], json!(false), "{body}");
    for fq in &fx.questions {
        let max = fq
            .raw_logits
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = fq.raw_logits.iter().map(|z| (z - max).exp()).collect();
        let total: f64 = e.iter().sum();
        let a = &body["answers"][&fq.id];
        let what = format!("{} {} raw", fx.label(), fq.id);
        let got: Vec<f64> = match fq.kind.as_str() {
            "noul" => vec![a["noul"].as_f64().expect("noul")],
            _ => fq
                .keys
                .iter()
                .map(|k| a["probabilities"][k].as_f64().expect("probability"))
                .collect(),
        };
        let want: Vec<f64> = match fq.kind.as_str() {
            "noul" => vec![e[1] / total],
            _ => e.iter().map(|x| x / total).collect(),
        };
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= RAW_BAND + 1e-4, "{what}: {g} vs {w}");
        }
    }
    println!("ok: raw answers");
}

/// Two one-row requests queued behind a running blocker run as one merged
/// plan, each answer equal to the request run alone; then a full queue.
async fn merge_and_queue_checks(
    served: &Served,
    fixtures: &[Fixture],
    solo: &[Map<String, Value>],
) {
    let service = served
        .state
        .scheduler
        .decision_service(&served.name)
        .await
        .expect("served above");
    let singles: Vec<usize> = (0..fixtures.len())
        .filter(|&i| fixtures[i].rows.len() == 1)
        .take(2)
        .collect();
    assert_eq!(singles.len(), 2, "two one-row fixtures for the merged pair");
    let state = "word ".repeat(1200);
    let hold = blocker(&service, &noul_request(6, &state));
    let before = service.stats();
    let ((s1, b1), (s2, b2)) = tokio::join!(
        post(&served.router, &fixtures[singles[0]].request),
        post(&served.router, &fixtures[singles[1]].request)
    );
    assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK), "{b1} {b2}");
    drop(hold);
    let merged = service.stats().merged_requests - before.merged_requests;
    assert_eq!(merged, 2, "the pair ran in one merged run");
    for (i, body) in singles.iter().zip([b1, b2]) {
        let what = format!("{} merged", fixtures[*i].label());
        compare::answers(&answers_of(&body, &what), &solo[*i], MERGED_BAND, &what);
    }
    println!("ok: a merged pair equals its solo runs");

    let hold = blocker(&service, &noul_request(6, &state));
    let (status, body) = post(&served.router, &noul_request(5, "busy")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], json!("queue_full"), "{body}");
    drop(hold);
    println!("ok: a full queue answers 503");
}

/// 422 for a bad body, a bad question, too many criteria and an over-long
/// state; 413 over `max_rows`; 404 for an unknown model; 400 for a model
/// that is not a decision model.
async fn error_checks(served: &Served) {
    let router = &served.router;
    let code = |body: &Value| body["error"]["code"].clone();
    let refused = |status: StatusCode, body: &Value, want: StatusCode, want_code: &str| {
        assert_eq!(status, want, "{body}");
        assert_eq!(code(body), json!(want_code), "{body}");
        assert!(body["error"]["message"]
            .as_str()
            .is_some_and(|m| !m.is_empty()));
        assert!(body["error"]["type"].as_str().is_some(), "{body}");
    };

    let (status, body) = post_raw(router, b"{".to_vec()).await;
    refused(
        status,
        &body,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_request",
    );

    let bad = json!({"state": "s", "questions": {"q": {"type": "mystery"}}});
    let (status, body) = post(router, &bad).await;
    refused(
        status,
        &body,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_request",
    );

    let wide: Map<String, Value> = (0..256).map(|i| (format!("o{i}"), Value::Null)).collect();
    let too_wide = json!({"state": "s", "questions": {"q": {"type": "choice", "criteria": wide}}});
    let (status, body) = post(router, &too_wide).await;
    refused(
        status,
        &body,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_request",
    );
    assert!(body["error"]["message"]
        .as_str()
        .is_some_and(|m| m.contains("1..255")));

    let long = noul_request(1, &"word ".repeat(66_000));
    let (status, body) = post(router, &long).await;
    refused(
        status,
        &body,
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_request",
    );
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("65,536-token limit"), "{message}");

    let (status, body) = post(router, &noul_request(MAX_ROWS + 6, "s")).await;
    refused(
        status,
        &body,
        StatusCode::PAYLOAD_TOO_LARGE,
        "request_too_large",
    );

    let mut unknown = noul_request(1, "s");
    unknown["model"] = json!("no-such-decision-model");
    let (status, body) = post(router, &unknown).await;
    refused(status, &body, StatusCode::NOT_FOUND, "model_not_found");

    let plain = tempfile::tempdir().expect("temp dir");
    std::fs::write(plain.path().join("model.safetensors"), b"").expect("write");
    let mut other = noul_request(1, "s");
    other["model"] = json!(plain.path().to_string_lossy());
    let (status, body) = post(router, &other).await;
    refused(
        status,
        &body,
        StatusCode::BAD_REQUEST,
        "not_a_decision_model",
    );
    println!("ok: 422, 413, 404 and 400");
}

/// The base on the CPU, every weight F32, the adapter attached unmerged.
#[tokio::test(flavor = "multi_thread")]
async fn kev_cpu_f32_route_matches_reference() {
    let Some((dir, fixtures)) = model_and_fixtures() else {
        return;
    };
    let Some(base) = env_path(ENV_BASE) else {
        println!("skip: {ENV_BASE} not set");
        return;
    };
    let served = serve(&dir, &base);
    let solo = fixture_checks(&served, &fixtures).await;
    models_check(&served, &dir, &fixtures[0]).await;
    raw_check(&served, &fixtures[0]).await;
    merge_and_queue_checks(&served, &fixtures, &solo).await;
    error_checks(&served).await;
}
