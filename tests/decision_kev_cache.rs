//! `POST /v1/systemone` on Kev with the prefix-state cache on: the Kev
//! directory served by its name through the model scheduler, the axum
//! router in-process.
//!
//! For each fixture: its request twice, then a request about the same
//! state with other questions, twice. The first request about a state runs
//! its prefix, cut at the state's end, and keeps it; every later one reads
//! it back and runs only its branches after it.
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
//! # Checks
//!
//! - The plan's prefix is the state, `<state>` included: the fixture's
//!   `state_len`, the same for other questions about it.
//! - The first send of each fixture matches Kev's answers within the band
//!   of `decision_kev_route.rs`.
//! - Every later send about the state is one hit in the worker's counts.
//!   Its answers equal the first send of the same request within one unit
//!   of Kev's rounding.
//! - `usage` holds Kev's fields only, a hit included.
//! - `/v1/models` reports the hits.

#![cfg(all(feature = "server", not(any(feature = "cuda", feature = "wgpu"))))]

mod decision_kev;
mod decision_route;

use axum::http::StatusCode;
use blazr::boostr::inference::decision::StateCacheStats;
use blazr::config::DecisionConfig;
use blazr::engine::decision::DecisionService;
use blazr::model::decision_template::Family;
use blazr::server::handlers::ServerRuntime;
use serde_json::{json, Map, Value};

use decision_kev::compare;
use decision_kev::fixture::{env_path, model_and_fixtures, Fixture, ENV_BASE};
use decision_route::prefix::{other_questions, prefix_len};
use decision_route::setup::{get, post, serve_config, Served, MAX_QUEUE_ROWS, MAX_ROWS};

const ROUTE: &str = "/v1/systemone";
/// The probability band of `decision_kev_route.rs` against Kev's answers.
const CPU_F32_BAND: f64 = 1e-2;
/// A hit against the miss that kept its state: one rounding unit.
const HIT_BAND: f64 = 0.0;

/// POST `body`, check that `usage` holds Kev's fields only, and return the
/// answers with the cache counts the request moved. The worker publishes
/// its counts before it answers, so they are current when the reply lands.
async fn send(
    served: &Served,
    service: &DecisionService<ServerRuntime>,
    body: &Value,
    input_tokens: Option<usize>,
    what: &str,
) -> (Map<String, Value>, StateCacheStats) {
    let before = service.stats().prefix_cache;
    let (status, reply) = post(&served.router, ROUTE, body).await;
    assert_eq!(status, StatusCode::OK, "{what}: {reply}");
    let now = service.stats().prefix_cache;
    let usage = reply["usage"]
        .as_object()
        .unwrap_or_else(|| panic!("{what}: no usage in {reply}"));
    let keys: Vec<&str> = usage.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        ["input_tokens", "output_tokens"],
        "{what}: Kev's usage"
    );
    if let Some(n) = input_tokens {
        assert_eq!(usage["input_tokens"], json!(n), "{what}: {reply}");
    }
    assert!(reply.get("truncated").is_none(), "{what}: {reply}");
    let answers = reply["answers"]
        .as_object()
        .unwrap_or_else(|| panic!("{what}: no answers in {reply}"))
        .clone();
    let moved = StateCacheStats {
        hits: now.hits - before.hits,
        misses: now.misses - before.misses,
        ..now
    };
    (answers, moved)
}

fn assert_hit(moved: &StateCacheStats, what: &str) {
    assert_eq!((moved.hits, moved.misses), (1, 0), "{what}: one hit");
}

async fn cache_checks(served: &Served, fixtures: &[Fixture]) {
    let service = served
        .state
        .scheduler
        .decision_service(&served.name)
        .await
        .unwrap_or_else(|e| panic!("decision service of {}: {e}", served.name));
    assert_eq!(service.family(), Family::Kev);
    assert!(
        service.calibration().is_some(),
        "head.pt sets a temperature"
    );
    let start = service.stats().prefix_cache;
    let mut hits = 0;
    for fx in fixtures {
        let what = fx.label();
        let saved = prefix_len(&service, &fx.request, &what);
        assert_eq!(saved, fx.state_len, "{what}: the prefix is the state");

        let input = Some(fx.input_tokens());
        let (first, moved) = send(served, &service, &fx.request, input, &what).await;
        assert_eq!(moved.hits + moved.misses, 1, "{what}: one lookup");
        compare::answers(&first, &fx.reference_answers, CPU_F32_BAND, &what);
        let again = format!("{what} sent again");
        let (second, moved) = send(served, &service, &fx.request, input, &again).await;
        assert_hit(&moved, &again);
        compare::answers(&second, &first, HIT_BAND, &again);

        let body = other_questions(&fx.request);
        let other = format!("{what} with other questions");
        assert_eq!(prefix_len(&service, &body, &other), saved, "{other}");
        let (first, moved) = send(served, &service, &body, None, &other).await;
        assert_hit(&moved, &other);
        let other_again = format!("{other} sent again");
        let (second, moved) = send(served, &service, &body, None, &other_again).await;
        assert_hit(&moved, &other_again);
        compare::answers(&second, &first, HIT_BAND, &other_again);
        hits += 3;
        println!("ok: {what} over a kept state ({saved} tokens)");
    }

    let now = service.stats().prefix_cache;
    assert!(now.hits - start.hits >= hits, "{now:?}");
    assert!(now.misses > start.misses, "the first state ran: {now:?}");
    assert!(now.entries >= 1 && now.bytes > 0, "{now:?}");

    let (status, body) = get(&served.router, "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entry = body["data"]
        .as_array()
        .and_then(|d| d.iter().find(|m| m["id"] == json!(served.name)))
        .unwrap_or_else(|| panic!("/v1/models lists no {}: {body}", served.name));
    let report = &entry["decision"]["prefix_cache"];
    assert_eq!(report["enabled"], json!(true), "{report}");
    assert!(report["hits"].as_u64().unwrap_or(0) >= hits, "{report}");
    assert!(report["bytes"].as_u64().unwrap_or(0) > 0, "{report}");
    println!("ok: {} hits reported", now.hits);
}

/// The base on the CPU, every weight F32, the adapter attached unmerged,
/// every state from its first token a kept prefix.
#[tokio::test(flavor = "multi_thread")]
async fn kev_cpu_f32_prefix_cache_hits_match() {
    let Some((dir, fixtures)) = model_and_fixtures() else {
        return;
    };
    let Some(base) = env_path(ENV_BASE) else {
        println!("skip: {ENV_BASE} not set");
        return;
    };
    let decision = DecisionConfig {
        base: Some(base.display().to_string()),
        max_rows: MAX_ROWS,
        max_queue_rows: MAX_QUEUE_ROWS,
        shared_min_tokens: 1,
        prefix_cache_mb: 4096,
        ..DecisionConfig::default()
    };
    let served = serve_config(&dir, decision);
    cache_checks(&served, &fixtures).await;
}
