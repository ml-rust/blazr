//! Helpers of the prefix-state cache tests: a request's other questions
//! about the same state, and the prefix length it plans with.

use blazr::engine::decision::DecisionService;
use blazr::model::decision_template::JevRequest;
use blazr::server::handlers::ServerRuntime;
use serde_json::{json, Value};

/// `request` with two noul questions in place of its own.
pub fn other_questions(request: &Value) -> Value {
    let mut body = request.clone();
    body["questions"] = json!({
        "acts": {"type": "noul", "instructions": "The state asks for an action.",
            "criteria": {"true": "it does", "false": "it does not"}},
        "late": {"type": "noul", "instructions": "Something in the state is overdue.",
            "criteria": {"true": "it is", "false": "it is not"}}
    });
    body
}

/// The prefix length `body` plans with on `service`: what a hit saves.
pub fn prefix_len(service: &DecisionService<ServerRuntime>, body: &Value, what: &str) -> usize {
    let request =
        JevRequest::parse(body, service.family()).unwrap_or_else(|e| panic!("{what}: parse: {e}"));
    let rendered = service
        .renderer()
        .render(&request)
        .unwrap_or_else(|e| panic!("{what}: render: {e}"));
    let prepared = service
        .prepare(&rendered, false)
        .unwrap_or_else(|e| panic!("{what}: prepare: {e}"));
    prepared.request.plan.prefix_len()
}
