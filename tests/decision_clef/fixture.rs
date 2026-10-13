//! Clef's reference fixtures (schema `boostr.decision.joint.v1`, written
//! by `clef_ref.py` through the release's own `encode_record`,
//! `ClefModel` and `systemone` on the CPU, backbone in BF16): the wire
//! request, every prompt piece with its ids, the record's ids and spans,
//! the backbone's final-normed hidden rows and the `lm_head` rows of every
//! option token as BF16 `.npy` arrays, each field's logits and
//! probabilities from the release BF16 head and from an F32 head, and the
//! wire answers of both.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Map, Value};

use super::npy::read_bf16;

/// The Clef release directory: backbone, head and tokenizer.
pub const ENV_MODEL: &str = "BLAZR_DECISION_MODEL_DIR";
/// The fixtures of that release.
pub const ENV_FIXTURES: &str = "BLAZR_DECISION_FIXTURE_DIR";
pub const SCHEMA: &str = "boostr.decision.joint.v1";

/// One prompt piece, encoded on its own.
#[derive(Deserialize)]
pub struct Segment {
    pub text: String,
    pub ids: Vec<u32>,
}

/// One field's layout and reference readout.
#[derive(Deserialize)]
pub struct Question {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub type_id: usize,
    pub question_span: [usize; 2],
    pub option_ids: Vec<String>,
    pub option_spans: Vec<[usize; 2]>,
    pub option_token_ids: Vec<Vec<u32>>,
    pub logits_bf16: Vec<f64>,
    pub probs_bf16: Vec<f64>,
    pub logits_f32: Vec<f64>,
    pub probs_f32: Vec<f64>,
}

/// A `.npy` file beside the fixture.
#[derive(Deserialize)]
pub struct Hidden {
    pub file: String,
}

/// The `lm_head` rows the fixture holds.
#[derive(Deserialize)]
pub struct LmHead {
    pub row_ids: Vec<u32>,
    pub rows_file: String,
}

/// One reference request.
#[derive(Deserialize)]
pub struct Fixture {
    pub schema: String,
    pub name: String,
    pub request: Value,
    pub segments: Vec<Segment>,
    pub input_ids: Vec<u32>,
    pub global_row: usize,
    pub hidden: Hidden,
    pub lm_head: LmHead,
    pub questions: Vec<Question>,
    /// The release's whole response: `model`, `answers`, `usage`.
    pub answers: Value,
    /// The answers of the F32 head on the same BF16 hidden rows.
    pub answers_f32: Map<String, Value>,
    #[serde(skip)]
    pub dir: PathBuf,
}

impl Fixture {
    /// `clef/name`, for messages.
    pub fn label(&self) -> String {
        format!("clef/{}", self.name)
    }

    /// The release's wire answers.
    pub fn release_answers(&self) -> Map<String, Value> {
        self.answers["answers"]
            .as_object()
            .unwrap_or_else(|| panic!("{}: no answers.answers", self.label()))
            .clone()
    }

    /// `[seq_len, H]` hidden rows, BF16 values as F32, and `H`.
    pub fn hidden_rows(&self) -> (Vec<f32>, usize) {
        let npy = read_bf16(&self.dir.join(&self.hidden.file));
        assert_eq!(
            npy.shape.first(),
            Some(&self.input_ids.len()),
            "{}: hidden rows",
            self.label()
        );
        (npy.data, npy.shape[1])
    }

    /// The `lm_head` row of each id in `row_ids`, BF16 values as F32.
    pub fn lm_rows(&self) -> Vec<f32> {
        let npy = read_bf16(&self.dir.join(&self.lm_head.rows_file));
        assert_eq!(
            npy.shape.first(),
            Some(&self.lm_head.row_ids.len()),
            "{}: lm rows",
            self.label()
        );
        npy.data
    }
}

/// The value of `var` as a path. `None` when unset or empty.
pub fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn load(path: &Path) -> Fixture {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut fx: Fixture =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(fx.schema, SCHEMA, "{}", path.display());
    fx.dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    fx
}

/// The Clef directory and every fixture of [`ENV_FIXTURES`], sorted by
/// path, or `None` after one `skip:` line.
pub fn model_and_fixtures() -> Option<(PathBuf, Vec<Fixture>)> {
    let Some(dir) = env_path(ENV_MODEL) else {
        println!("skip: {ENV_MODEL} not set");
        return None;
    };
    if !dir.join("joint_head.safetensors").is_file() {
        println!("skip: {} has no joint_head.safetensors", dir.display());
        return None;
    }
    let Some(fixtures) = env_path(ENV_FIXTURES) else {
        println!("skip: {ENV_FIXTURES} not set");
        return None;
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&fixtures)
        .unwrap_or_else(|e| panic!("{}: {e}", fixtures.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    if paths.is_empty() {
        println!("skip: no *.json fixtures in {}", fixtures.display());
        return None;
    }
    Some((dir, paths.iter().map(|p| load(p)).collect()))
}
