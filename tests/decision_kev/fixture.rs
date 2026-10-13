//! Kev's reference fixtures (schema `boostr.decision.pointer.v1`, written
//! by `ref_kev.py` through Kev's own code in F32 on the CPU): the wire
//! request, each question's row with its decide and option-end positions,
//! the raw pointer logits, the served probabilities and Kev's own wire
//! answers.

use std::path::PathBuf;

use serde::Deserialize;
use serde_json::{Map, Value};

/// The Kev checkpoint directory: adapter, `head.pt`, tokenizer copy.
pub const ENV_MODEL: &str = "BLAZR_DECISION_MODEL_DIR";
/// The fixtures of that checkpoint.
pub const ENV_FIXTURES: &str = "BLAZR_DECISION_FIXTURE_DIR";
/// The base checkpoint the adapter applies to.
pub const ENV_BASE: &str = "BLAZR_DECISION_BASE";
pub const SCHEMA: &str = "boostr.decision.pointer.v1";

/// The delimiter ids Kev's tokenizer gave.
#[derive(Deserialize)]
pub struct SpecialTokens {
    pub state: u32,
    pub question: u32,
    pub option: u32,
    pub option_end: u32,
    pub decide: u32,
}

/// One question's layout and reference readout.
#[derive(Deserialize)]
pub struct Question {
    pub id: String,
    pub kind: String,
    pub options: usize,
    pub keys: Vec<String>,
    pub row: usize,
    pub decide: usize,
    pub option_ends: Vec<usize>,
    pub raw_logits: Vec<f64>,
    pub probs: Vec<f64>,
}

/// One reference request.
#[derive(Deserialize)]
pub struct Fixture {
    pub schema: String,
    pub name: String,
    pub request: Value,
    pub special_tokens: SpecialTokens,
    pub temperature: f64,
    pub state_len: usize,
    pub rows: Vec<Vec<u32>>,
    pub questions: Vec<Question>,
    pub reference_answers: Map<String, Value>,
}

impl Fixture {
    /// `kev/name`, for messages.
    pub fn label(&self) -> String {
        format!("kev/{}", self.name)
    }

    /// Kev's `usage.input_tokens`: the state once, then every branch.
    pub fn input_tokens(&self) -> usize {
        self.state_len
            + self
                .rows
                .iter()
                .map(|r| r.len() - self.state_len)
                .sum::<usize>()
    }
}

/// The value of `var` as a path. `None` when unset or empty.
pub fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Every fixture of [`ENV_FIXTURES`], sorted by path, or `None` after one
/// `skip:` line.
pub fn fixtures() -> Option<Vec<Fixture>> {
    let Some(dir) = env_path(ENV_FIXTURES) else {
        println!("skip: {ENV_FIXTURES} not set");
        return None;
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        println!("skip: no *.json fixtures in {}", dir.display());
        return None;
    }
    let fixtures = paths
        .iter()
        .map(|p| {
            let text =
                std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let fx: Fixture =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            assert_eq!(fx.schema, SCHEMA, "{}", p.display());
            fx
        })
        .collect();
    Some(fixtures)
}

/// The directory of [`ENV_MODEL`] and the fixtures, or `None` after one
/// `skip:` line.
pub fn model_and_fixtures() -> Option<(PathBuf, Vec<Fixture>)> {
    let Some(dir) = env_path(ENV_MODEL) else {
        println!("skip: {ENV_MODEL} not set");
        return None;
    };
    if !dir.is_dir() {
        println!("skip: {} is not a directory", dir.display());
        return None;
    }
    fixtures().map(|f| (dir, f))
}
