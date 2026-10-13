//! `POST /v1/systemone` end to end: a real System One decision model (a
//! Qwen3.5 text checkpoint such as Jebadiah or the Mapika decider) loaded
//! through the model scheduler, the axum router in-process, every fixture's
//! request posted and its answers compared with the family's own wire
//! answers.
//!
//! # Fixtures
//!
//! - `BLAZR_DECISION_MODEL_DIR`: the checkpoint directory, sidecars beside
//!   the weights. The `hf_` test loads it on the CPU in F32.
//! - `BLAZR_DECISION_GGUF`: a Q8_0 or Q4_K_M `.gguf` of the same model, for
//!   the `gguf_` test, on the build's runtime. Set one of the two.
//! - `BLAZR_DECISION_CONFIG`: the sidecar of a `.gguf` that ships none
//!   beside it, such as the checkpoint's `decider_config.json`. Without one
//!   the model serves uncalibrated answers and the test fails, since the
//!   fixtures are calibrated.
//! - `BLAZR_DECISION_FIXTURE_DIR`: the `*.json` fixtures of that model,
//!   schema `boostr.decision.label_token.v2`.
//!
//! Unset or missing, each test prints one `skip:` line and passes.
//!
//! # Tolerance
//!
//! `formatter_` feeds the reference's own distributions to the response
//! builder: values match to one unit of the family's rounding. The route
//! tests carry the probability band of `tests/decision_loader/compare.rs`
//! (CPU F32 1e-3, Q8_0 0.1, Q4_K_M 0.3) through each formula, plus one
//! rounding unit; a choice must match where the reference's top two are
//! further apart than twice the band. A merged run must match the same
//! request run alone within 2e-5.

#![cfg(feature = "server")]

mod decision_route;

use axum::http::StatusCode;
use blazr::boostr::inference::decision::{answer_from_probs, AnswerOptions, DecisionResult};
use blazr::engine::decision::{prepare, unique_tokens, DecisionService, Pending, PrefixRule};
use blazr::model::decision_template::{DecisionRenderer, JevRequest};
use blazr::server::decision::answers_json;
use blazr::server::handlers::ServerRuntime;
use serde_json::{json, Map, Value};

use decision_route::compare::{self, Tolerance, EXACT, MERGED};
use decision_route::fixture::{env_path, fixtures, model_and_fixtures, Fixture, ENV_CONFIG};
use decision_route::fixture::{ENV_GGUF, ENV_MODEL};
use decision_route::setup::{get, post, post_raw, serve, Served, MAX_QUEUE_ROWS, MAX_ROWS};

const ROUTE: &str = "/v1/systemone";

fn answers_of(body: &Value, what: &str) -> Map<String, Value> {
    body["answers"]
        .as_object()
        .unwrap_or_else(|| panic!("{what}: no answers in {body}"))
        .clone()
}

/// `n` noul questions about `state`: `n` rows in either family.
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

/// The response builder alone: the reference's distributions through
/// boostr's answer stage and the family's formatter give the family's own
/// wire answers. Needs the checkpoint directory for the renderer only.
#[test]
fn formatter_reproduces_reference_answers() {
    let Some(dir) = env_path(ENV_MODEL) else {
        println!("skip: {ENV_MODEL} not set");
        return;
    };
    let Some(fixtures) = fixtures() else {
        return;
    };
    let renderer =
        DecisionRenderer::load(&dir).unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));
    for fx in &fixtures {
        let what = fx.label();
        let request = JevRequest::parse(&fx.request, renderer.family())
            .unwrap_or_else(|e| panic!("{what}: parse: {e}"));
        let rendered = renderer
            .render(&request)
            .unwrap_or_else(|e| panic!("{what}: render: {e}"));
        assert_eq!(rendered.rows, fx.rows, "{what}: rows");
        let answers = rendered
            .questions
            .iter()
            .zip(&fx.questions)
            .map(|(q, fq)| {
                assert_eq!(q.id, fq.id, "{what}: question order");
                answer_from_probs(&q.meta, fx.answer_probs(fq), &AnswerOptions::default())
                    .unwrap_or_else(|e| panic!("{what} {}: answer: {e}", q.id))
            })
            .collect();
        let result = DecisionResult {
            answers,
            raw_logits: Vec::new(),
            temperatures: Vec::new(),
        };
        let got = answers_json(renderer.family(), &rendered.questions, &result)
            .unwrap_or_else(|e| panic!("{what}: format: {e}"));
        compare::answers(&fx.family, &got, &fx.reference_answers, EXACT, &what);
        println!("ok: {what}");
    }
}

/// Every fixture through the route, then `/v1/models`, the shared-prefix
/// path, a merged pair, a full queue and each error status.
async fn route_checks(served: &Served, fixtures: &[Fixture], tol: Tolerance) {
    let router = &served.router;
    let service = served
        .state
        .scheduler
        .decision_service(&served.name)
        .await
        .unwrap_or_else(|e| panic!("decision service of {}: {e}", served.name));
    let family = service.family().name();
    assert!(
        service.calibration().is_some(),
        "{} serves uncalibrated answers and the fixtures are calibrated: set {ENV_CONFIG} to \
         the model's sidecar (warnings: {:?})",
        served.name,
        service.warnings()
    );

    let mut solo = Vec::with_capacity(fixtures.len());
    for fx in fixtures {
        let what = fx.label();
        let request = JevRequest::parse(&fx.request, service.family())
            .unwrap_or_else(|e| panic!("{what}: parse: {e}"));
        let rendered = service
            .renderer()
            .render(&request)
            .unwrap_or_else(|e| panic!("{what}: render: {e}"));
        assert_eq!(
            rendered.rows, fx.rows,
            "{what}: the served tokenizer renders other rows"
        );

        let (status, body) = post(router, ROUTE, &fx.request).await;
        assert_eq!(status, StatusCode::OK, "{what}: {body}");
        let got = answers_of(&body, &what);
        compare::answers(&fx.family, &got, &fx.reference_answers, tol, &what);
        assert_eq!(body["model"], json!(served.name), "{what}: model");
        let usage = json!({"input_tokens": unique_tokens(&fx.rows), "output_tokens": 0,
            "cached_tokens": 0, "cached_input_tokens": 0});
        assert_eq!(body["usage"], usage, "{what}: usage, the prefix cache off");
        assert_eq!(
            body["calibration"]["applied"],
            json!(true),
            "{what}: calibration"
        );
        let temps = body["calibration"]["temperatures"]
            .as_object()
            .unwrap_or_else(|| panic!("{what}: no temperatures"));
        for (kind, t) in &fx.temperatures {
            assert_eq!(temps.get(kind), Some(t), "{what}: temperature of {kind}");
        }
        println!("ok: {what} through {ROUTE}");
        solo.push(got);
    }
    let fx0 = &fixtures[0];

    // /v1/models reports the model's decision object, weights untouched.
    let (status, body) = get(router, "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entry = body["data"]
        .as_array()
        .and_then(|d| d.iter().find(|m| m["id"] == json!(served.name)))
        .unwrap_or_else(|| panic!("/v1/models lists no {}: {body}", served.name));
    let decision = &entry["decision"];
    assert_eq!(decision["family"], json!(family), "{decision}");
    assert_eq!(decision["readout"], json!("label_token"));
    assert!(decision["context_length"].as_u64().is_some(), "{decision}");
    let listed = decision["calibration"]["temperatures"].as_object();
    assert_eq!(listed, body_temps(fx0).as_object(), "{decision}");

    // The shared-prefix path: each multi-row fixture run over its prefix.
    let before = service.stats();
    let mut prefix_runs = 0;
    for fx in fixtures.iter().filter(|fx| fx.rows.len() >= 2) {
        let what = format!("{} over its shared prefix", fx.label());
        let request = JevRequest::parse(&fx.request, service.family()).expect("parsed above");
        let rendered = service.renderer().render(&request).expect("rendered above");
        let rule = PrefixRule {
            min_tokens: 1,
            at_state: false,
        };
        let prepared = prepare(&rendered, service.temperature_policy(false), rule)
            .unwrap_or_else(|e| panic!("{what}: prepare: {e}"));
        if !prepared.has_prefix() {
            continue;
        }
        let result = service
            .submit(prepared)
            .unwrap_or_else(|e| panic!("{what}: submit: {e}"))
            .wait()
            .await
            .unwrap_or_else(|e| panic!("{what}: run: {e}"));
        let got = answers_json(service.family(), &rendered.questions, &result)
            .unwrap_or_else(|e| panic!("{what}: format: {e}"));
        compare::answers(&fx.family, &got, &fx.reference_answers, tol, &what);
        prefix_runs += 1;
        println!("ok: {what}");
    }
    assert_eq!(
        service.stats().prefix_runs - before.prefix_runs,
        prefix_runs
    );

    // A merged pair: two one-row requests queued behind a running blocker
    // run as one plan, and each answer equals the request run alone.
    let singles: Vec<usize> = (0..fixtures.len())
        .filter(|&i| fixtures[i].rows.len() == 1)
        .take(2)
        .collect();
    assert_eq!(singles.len(), 2, "two one-row fixtures for the merged pair");
    let state = "word ".repeat(1200);
    let hold = blocker(&service, &noul_request(6, &state));
    let before = service.stats();
    let ((s1, b1), (s2, b2)) = tokio::join!(
        post(router, ROUTE, &fixtures[singles[0]].request),
        post(router, ROUTE, &fixtures[singles[1]].request)
    );
    assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK), "{b1} {b2}");
    drop(hold);
    let merged = service.stats().merged_requests - before.merged_requests;
    assert_eq!(merged, 2, "the pair ran in one merged run");
    for (i, body) in singles.iter().zip([b1, b2]) {
        let what = format!("{} merged", fixtures[*i].label());
        compare::answers(family, &answers_of(&body, &what), &solo[*i], MERGED, &what);
    }
    println!("ok: a merged pair equals its solo runs");

    // A full queue: a 6-row blocker leaves no room for 5 more rows.
    let hold = blocker(&service, &noul_request(6, &state));
    let (status, body) = post(router, ROUTE, &noul_request(5, "busy")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], json!("queue_full"), "{body}");
    drop(hold);

    error_checks(served).await;

    let mut raw = fx0.request.clone();
    raw["calibration"] = json!("raw");
    let (status, body) = post(router, ROUTE, &raw).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["calibration"]["applied"], json!(false), "{body}");
}

/// The fixture's temperatures as `/v1/models` lists them.
fn body_temps(fx: &Fixture) -> Value {
    Value::Object(fx.temperatures.clone())
}

/// A queued request that holds the worker: dropping it cancels it.
fn blocker(service: &DecisionService<ServerRuntime>, body: &Value) -> Pending {
    let request = JevRequest::parse(body, service.family()).expect("blocker parses");
    let rendered = service
        .renderer()
        .render(&request)
        .expect("blocker renders");
    let prepared = service.prepare(&rendered, false).expect("blocker prepares");
    assert!(prepared.rows + 2 <= MAX_QUEUE_ROWS && prepared.rows + 5 > MAX_QUEUE_ROWS);
    service.submit(prepared).expect("blocker queues")
}

/// 422 for a bad body and a bad question, 413 over `max_rows`, 404 for an
/// unknown model, 400 for a model that is not a decision model.
async fn error_checks(served: &Served) {
    let router = &served.router;
    let code = |body: &Value| body["error"]["code"].clone();

    let (status, body) = post_raw(router, ROUTE, b"{".to_vec()).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(code(&body), json!("invalid_request"));

    let bad = json!({"state": "s", "questions": {"q": {"type": "mystery", "instructions": "x",
        "criteria": {"a": null, "b": null}}}});
    let (status, body) = post(router, ROUTE, &bad).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(body["error"]["message"]
        .as_str()
        .is_some_and(|m| !m.is_empty()));

    let (status, body) = post(router, ROUTE, &noul_request(MAX_ROWS + 6, "s")).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(code(&body), json!("request_too_large"));

    let mut unknown = noul_request(1, "s");
    unknown["model"] = json!("no-such-decision-model");
    let (status, body) = post(router, ROUTE, &unknown).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(code(&body), json!("model_not_found"));

    let plain = tempfile::tempdir().expect("temp dir");
    std::fs::write(plain.path().join("model.safetensors"), b"").expect("write");
    let mut other = noul_request(1, "s");
    other["model"] = json!(plain.path().to_string_lossy());
    let (status, body) = post(router, ROUTE, &other).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(code(&body), json!("not_a_decision_model"));
    println!("ok: 422, 413, 404 and 400");
}

/// The checkpoint on the CPU, every weight F32.
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
#[tokio::test(flavor = "multi_thread")]
async fn hf_cpu_f32_route_matches_reference() {
    let Some((dir, fixtures)) = model_and_fixtures(ENV_MODEL) else {
        return;
    };
    let served = serve(&dir, None, 0);
    route_checks(&served, &fixtures, compare::CPU_F32).await;
}

/// The GGUF on the build's runtime, block-quantized matrices kept
/// quantized; `BLAZR_DECISION_CONFIG` names its sidecar when none sits
/// beside it.
#[tokio::test(flavor = "multi_thread")]
async fn gguf_route_matches_reference() {
    let Some((file, fixtures)) = model_and_fixtures(ENV_GGUF) else {
        return;
    };
    let tol = compare::gguf_tolerance(&file);
    let served = serve(&file, env_path(ENV_CONFIG), 0);
    route_checks(&served, &fixtures, tol).await;
}
