//! Label-token decision fixtures (schema `boostr.decision.label_token.v2`,
//! written by `ref_label_token.py`) and the `DecisionPlan`, items,
//! question metadata and temperature policy each one describes. The plan
//! is built the way boostr's `decision_label_token_real_file` test builds
//! it.

use std::path::PathBuf;

use boostr::inference::decision::{
    DecisionPlan, Gather, PrefixInput, QuestionKind, QuestionMeta, TemperaturePolicy,
};
use boostr::model::decision::LabelTokenItem;
use serde::Deserialize;

pub const ENV_MODEL: &str = "BLAZR_DECISION_MODEL_DIR";
pub const ENV_GGUF: &str = "BLAZR_DECISION_GGUF";
pub const ENV_FIXTURES: &str = "BLAZR_DECISION_FIXTURE_DIR";
pub const SCHEMA: &str = "boostr.decision.label_token.v2";

/// The temperature the reference applies to each answer type.
#[derive(Deserialize)]
pub struct Temperatures {
    pub choice: f64,
    pub noul: f64,
    pub score: f64,
}

/// One scoring row: where the label logits are read and what they were.
#[derive(Deserialize)]
pub struct ScoringRow {
    pub row: usize,
    pub slot: usize,
    pub labels: Vec<u32>,
    pub question: usize,
    pub temperature: f64,
    pub raw_logits: Vec<f32>,
    pub probs: Vec<f64>,
}

/// One question and its final distribution.
#[derive(Deserialize)]
pub struct Question {
    pub id: String,
    pub kind: String,
    pub isolated: bool,
    pub options: usize,
    pub yes_index: Option<usize>,
    pub temperature: f64,
    pub scoring_rows: [usize; 2],
    pub probs: Vec<f64>,
    pub p_yes: Option<f64>,
    pub fit_mass: Option<f64>,
}

/// One reference request.
#[derive(Deserialize)]
pub struct Fixture {
    pub schema: String,
    pub family: String,
    pub name: String,
    pub shared_prefix_len: usize,
    pub rows: Vec<Vec<u32>>,
    pub temperatures: Temperatures,
    pub scoring_rows: Vec<ScoringRow>,
    pub questions: Vec<Question>,
}

impl Fixture {
    /// `family/name`, for messages.
    pub fn label(&self) -> String {
        format!("{}/{}", self.family, self.name)
    }
}

/// The value of `var` as a path, or `None` after one `skip:` line.
fn env_path(var: &str) -> Option<PathBuf> {
    let path = std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    if path.is_none() {
        println!("skip: {var} not set");
    }
    path
}

/// Panics when both [`ENV_MODEL`] and [`ENV_GGUF`] are set: the fixtures
/// belong to one model, so the run takes one source.
fn one_model_source() {
    let set = |var: &str| std::env::var(var).is_ok_and(|v| !v.is_empty());
    assert!(
        !(set(ENV_MODEL) && set(ENV_GGUF)),
        "{ENV_MODEL} and {ENV_GGUF} are both set; unset one: {ENV_MODEL} names a Hugging \
         Face directory, {ENV_GGUF} a .gguf file"
    );
}

/// The model directory of [`ENV_MODEL`] and every fixture of
/// [`ENV_FIXTURES`], or `None` after one `skip:` line.
pub fn inputs() -> Option<(PathBuf, Vec<Fixture>)> {
    one_model_source();
    let dir = env_path(ENV_MODEL)?;
    if !dir.join("config.json").is_file() {
        println!("skip: {} has no config.json", dir.display());
        return None;
    }
    Some((dir, fixtures()?))
}

/// The `.gguf` file of [`ENV_GGUF`] and every fixture of
/// [`ENV_FIXTURES`], or `None` after one `skip:` line.
pub fn gguf_inputs() -> Option<(PathBuf, Vec<Fixture>)> {
    one_model_source();
    let file = env_path(ENV_GGUF)?;
    if !file.is_file() {
        println!("skip: {} is not a file", file.display());
        return None;
    }
    Some((file, fixtures()?))
}

/// Every fixture of [`ENV_FIXTURES`], sorted by path, or `None` after one
/// `skip:` line.
fn fixtures() -> Option<Vec<Fixture>> {
    let fixtures = env_path(ENV_FIXTURES)?;
    let entries = std::fs::read_dir(&fixtures)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", fixtures.display()));
    let mut paths: Vec<PathBuf> = entries
        .map(|e| {
            e.unwrap_or_else(|e| panic!("{}: {e}", fixtures.display()))
                .path()
        })
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        println!("skip: {} holds no *.json fixture", fixtures.display());
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

/// The question metadata of `fx`, after checking that each question's
/// scoring rows follow the previous question's.
fn questions(fx: &Fixture) -> Vec<QuestionMeta> {
    let what = fx.label();
    let mut next = 0;
    let meta = fx
        .questions
        .iter()
        .enumerate()
        .map(|(q, question)| {
            let [start, count] = question.scoring_rows;
            assert_eq!(start, next, "{what}: question {q} rows start at {start}");
            next += count;
            for sr in &fx.scoring_rows[start..start + count] {
                assert_eq!(sr.question, q, "{what}: scoring row of question {q}");
            }
            let kind = match question.kind.as_str() {
                "choice" => QuestionKind::Choice,
                "noul" => QuestionKind::Noul {
                    yes: question
                        .yes_index
                        .unwrap_or_else(|| panic!("{what}: noul {} has no yes_index", question.id)),
                },
                "score" => QuestionKind::Score {
                    values: None,
                    isolated: question.isolated,
                },
                other => panic!("{what}: question kind {other:?}"),
            };
            QuestionMeta {
                kind,
                options: question.options,
            }
        })
        .collect();
    assert_eq!(
        next,
        fx.scoring_rows.len(),
        "{what}: rows past the questions"
    );
    meta
}

/// The plan, items, question metadata and temperature policy of `fx`.
///
/// The plan's prefix is the fixture's shared prefix, each row its tokens
/// after it, and each scoring row one gather at its answer slot and one
/// item over its labels. One temperature for every question gives
/// `Fixed`; mixed answer types give `ByKind` over the fixture's map.
pub fn request(
    fx: &Fixture,
) -> (
    DecisionPlan,
    Vec<LabelTokenItem>,
    Vec<QuestionMeta>,
    TemperaturePolicy,
) {
    let what = fx.label();
    let p = fx.shared_prefix_len;
    let first = fx.rows.first().unwrap_or_else(|| panic!("{what}: no rows"));
    let prefix = first[..p].to_vec();
    for (r, row) in fx.rows.iter().enumerate() {
        assert!(
            row.len() > p && row[..p] == prefix[..],
            "{what}: row {r} does not continue the shared prefix of {p} tokens"
        );
    }
    let rows = fx.rows.iter().map(|row| row[p..].to_vec()).collect();
    let mut gathers = Vec::with_capacity(fx.scoring_rows.len());
    let mut items = Vec::with_capacity(fx.scoring_rows.len());
    for (s, sr) in fx.scoring_rows.iter().enumerate() {
        assert!(
            sr.slot >= p && sr.slot < fx.rows[sr.row].len(),
            "{what}: scoring row {s} reads position {} of row {}",
            sr.slot,
            sr.row
        );
        gathers.push(Gather {
            row: Some(sr.row),
            pos: sr.slot,
        });
        items.push(LabelTokenItem {
            slot: s,
            labels: sr.labels.clone(),
        });
    }
    let meta = questions(fx);
    let t0 = fx.questions[0].temperature;
    let temp = if fx.questions.iter().all(|q| q.temperature == t0) {
        TemperaturePolicy::Fixed(t0)
    } else {
        let t = &fx.temperatures;
        TemperaturePolicy::ByKind {
            choice: t.choice,
            noul: t.noul,
            score: t.score,
        }
    };
    let plan = DecisionPlan {
        prefix: PrefixInput::Tokens(prefix),
        rows,
        gathers,
    };
    (plan, items, meta, temp)
}
