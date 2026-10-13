//! The reference fixtures of `/v1/systemone` (schema
//! `boostr.decision.label_token.v2`, written by `ref_label_token.py`): the
//! request the family's code received, its rows, its final distributions
//! and the family's own wire answers.

use std::path::PathBuf;

use serde::Deserialize;
use serde_json::{Map, Value};

pub const ENV_MODEL: &str = "BLAZR_DECISION_MODEL_DIR";
pub const ENV_GGUF: &str = "BLAZR_DECISION_GGUF";
pub const ENV_FIXTURES: &str = "BLAZR_DECISION_FIXTURE_DIR";
pub const ENV_CONFIG: &str = "BLAZR_DECISION_CONFIG";
pub const SCHEMA: &str = "boostr.decision.label_token.v2";

/// One scoring row's reference probabilities.
#[derive(Deserialize)]
pub struct ScoringRow {
    pub probs: Vec<f64>,
}

/// One question's reference distribution.
#[derive(Deserialize)]
pub struct Question {
    pub id: String,
    pub isolated: bool,
    pub scoring_rows: [usize; 2],
    pub probs: Vec<f64>,
}

/// One reference request.
#[derive(Deserialize)]
pub struct Fixture {
    pub schema: String,
    pub family: String,
    pub name: String,
    pub request: Value,
    pub rows: Vec<Vec<u32>>,
    pub temperatures: Map<String, Value>,
    pub scoring_rows: Vec<ScoringRow>,
    pub questions: Vec<Question>,
    pub reference_answers: Map<String, Value>,
}

impl Fixture {
    /// `family/name`, for messages.
    pub fn label(&self) -> String {
        format!("{}/{}", self.family, self.name)
    }

    /// The per-level yes probabilities of an isolated Score question, else
    /// the question's distribution: what boostr's `answer_from_probs` takes.
    pub fn answer_probs(&self, q: &Question) -> Vec<f64> {
        let [first, count] = q.scoring_rows;
        if q.isolated {
            self.scoring_rows[first..first + count]
                .iter()
                .map(|row| row.probs[1])
                .collect()
        } else {
            q.probs.clone()
        }
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
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .map(|e| {
            e.unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
                .path()
        })
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        println!("skip: {} holds no *.json fixture", dir.display());
        return None;
    }
    let parsed = paths
        .iter()
        .map(|p| {
            let text =
                std::fs::read_to_string(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
            let fx: Fixture = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("parse {}: {e}", p.display()));
            assert_eq!(fx.schema, SCHEMA, "{}: schema", p.display());
            fx
        })
        .collect();
    Some(parsed)
}

/// The model of [`ENV_MODEL`] (a checkpoint directory) or [`ENV_GGUF`] (a
/// `.gguf`), and the fixtures, or `None` after one `skip:` line.
pub fn model_and_fixtures(var: &str) -> Option<(PathBuf, Vec<Fixture>)> {
    let set = |v: &str| env_path(v).is_some();
    assert!(
        !(set(ENV_MODEL) && set(ENV_GGUF)),
        "{ENV_MODEL} and {ENV_GGUF} are both set; unset one: the fixtures belong to one model"
    );
    let Some(path) = env_path(var) else {
        println!("skip: {var} not set");
        return None;
    };
    if !path.exists() {
        println!("skip: {} does not exist", path.display());
        return None;
    }
    Some((path, fixtures()?))
}
