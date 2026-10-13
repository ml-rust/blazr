//! `POST /v1/systemone` with the prefix-state cache on: a real System One
//! decision model (the Mapika decider, such as `decider-4b`) loaded
//! through the model scheduler, the axum router in-process.
//!
//! For each fixture: its request twice, then a request about the same
//! state with other questions, twice. The first request about a state runs
//! its prefix and keeps it; every later one reads it back.
//!
//! # Fixtures
//!
//! - `BLAZR_DECISION_MODEL_DIR`: the checkpoint directory, sidecars beside
//!   the weights, loaded on the CPU in F32.
//! - `BLAZR_DECISION_FIXTURE_DIR`: the `*.json` fixtures of that model,
//!   schema `boostr.decision.label_token.v2`.
//!
//! Unset or missing, the test prints one `skip:` line and passes.
//!
//! # Checks
//!
//! - The first send of each fixture matches the family's reference answers
//!   within the CPU F32 band of `decision_route_real_file.rs`.
//! - Every later send about the state is a hit: `usage.cached_input_tokens`
//!   and `usage.cached_tokens` equal the plan's prefix length, above 0.
//! - A hit's answers equal the first send of the same request within one
//!   unit of the family's rounding.
//! - The worker's counts and `/v1/models` report the hits.

#![cfg(all(feature = "server", not(any(feature = "cuda", feature = "wgpu"))))]

mod decision_route;

use axum::http::StatusCode;
use blazr::config::DecisionConfig;
use serde_json::{json, Map, Value};

use decision_route::compare::{self, CPU_F32, EXACT};
use decision_route::fixture::{model_and_fixtures, Fixture, ENV_MODEL};
use decision_route::prefix::{other_questions, prefix_len};
use decision_route::setup::{get, post, serve_config, Served};

const ROUTE: &str = "/v1/systemone";

fn answers_of(body: &Value, what: &str) -> Map<String, Value> {
    body["answers"]
        .as_object()
        .unwrap_or_else(|| panic!("{what}: no answers in {body}"))
        .clone()
}

/// POST `body` and return its answers and `usage.cached_input_tokens`,
/// after checking that `cached_tokens` says the same.
async fn send(served: &Served, body: &Value, what: &str) -> (Map<String, Value>, u64) {
    let (status, reply) = post(&served.router, ROUTE, body).await;
    assert_eq!(status, StatusCode::OK, "{what}: {reply}");
    let usage = &reply["usage"];
    let cached = usage["cached_input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("{what}: no usage.cached_input_tokens in {reply}"));
    assert_eq!(usage["cached_tokens"], json!(cached), "{what}: {usage}");
    let input = usage["input_tokens"].as_u64().unwrap_or(0);
    assert!(cached <= input, "{what}: {cached} cached of {input}");
    (answers_of(&reply, what), cached)
}

async fn cache_checks(served: &Served, fixtures: &[Fixture]) {
    let service = served
        .state
        .scheduler
        .decision_service(&served.name)
        .await
        .unwrap_or_else(|e| panic!("decision service of {}: {e}", served.name));
    assert!(
        service.calibration().is_some(),
        "{} serves uncalibrated answers and the fixtures are calibrated (warnings: {:?})",
        served.name,
        service.warnings()
    );
    let family = service.family().name();
    let start = service.stats().prefix_cache;
    let mut hits = 0;
    for fx in fixtures {
        let what = fx.label();
        let saved = prefix_len(&service, &fx.request, &what) as u64;
        assert!(saved > 0, "{what}: the state boundary is a prefix");

        let (first, _) = send(served, &fx.request, &what).await;
        compare::answers(&fx.family, &first, &fx.reference_answers, CPU_F32, &what);
        let again = format!("{what} sent again");
        let (second, cached) = send(served, &fx.request, &again).await;
        assert_eq!(cached, saved, "{again}: a hit saves the prefix");
        compare::answers(&fx.family, &second, &first, EXACT, &again);

        let body = other_questions(&fx.request);
        let other = format!("{what} with other questions");
        assert_eq!(prefix_len(&service, &body, &other) as u64, saved, "{other}");
        let (first, cached) = send(served, &body, &other).await;
        assert_eq!(cached, saved, "{other}: the state is kept");
        let (second, cached) = send(served, &body, &other).await;
        assert_eq!(cached, saved, "{other} sent again");
        compare::answers(family, &second, &first, EXACT, &other);
        hits += 3;
        println!("ok: {what} over a kept prefix ({saved} tokens)");
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

/// The checkpoint on the CPU, every weight F32, every state from its first
/// token a shared prefix.
#[tokio::test(flavor = "multi_thread")]
async fn hf_cpu_f32_prefix_cache_hits_match() {
    let Some((dir, fixtures)) = model_and_fixtures(ENV_MODEL) else {
        return;
    };
    let decision = DecisionConfig {
        shared_min_tokens: 1,
        prefix_cache_mb: 4096,
        ..DecisionConfig::default()
    };
    let served = serve_config(&dir, decision);
    cache_checks(&served, &fixtures).await;
}
