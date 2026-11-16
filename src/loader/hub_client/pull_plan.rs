//! The files `blazr pull` and `POST /api/pull` fetch from a model repo,
//! picked from the repo's file list.

use crate::engine::model_dir::stays_in_dir;

/// The configs and tokenizer files a text model's loader reads. Each one
/// the repo holds is fetched.
pub const SIDE_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
];

/// The single-file SafeTensors weights.
pub const SAFETENSORS: &str = "model.safetensors";

/// The index that names the shards of sharded SafeTensors weights.
pub const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";

/// The files to fetch from a repo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullPlan {
    /// Repo paths to fetch, in order.
    pub files: Vec<String>,
    /// Whether [`SAFETENSORS_INDEX`] is among `files`. Its shards are
    /// fetched after it, from [`shards_of`].
    pub sharded: bool,
}

/// Why a repo's file list gives no plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// The repo holds no weights blazr loads.
    NoWeights,
    /// The repo holds several GGUF files, and nothing picks one.
    SeveralGguf(Vec<String>),
    /// A repo path leaves the model directory.
    Unsafe(String),
}

impl PlanError {
    /// The message for `repo`, ending in `pick`: how the caller names one
    /// file, such as `Pass it with --file <name>`.
    pub fn message(&self, repo: &str, pick: &str) -> String {
        match self {
            Self::NoWeights => {
                format!("{repo} holds no {SAFETENSORS}, {SAFETENSORS_INDEX} or .gguf file. {pick}")
            }
            Self::SeveralGguf(names) => format!(
                "{repo} holds {} GGUF files: {}. {pick}",
                names.len(),
                names.join(", ")
            ),
            Self::Unsafe(name) => format!(
                "{repo} names the file '{name}', which leaves the model directory. Pull a repo \
                 with plain file names"
            ),
        }
    }
}

/// The plan for a repo whose files are `files`: every [`SIDE_FILES`] entry
/// it holds, then its weights. The weights are [`SAFETENSORS`], else
/// [`SAFETENSORS_INDEX`] and its shards, else its one `.gguf` file.
///
/// # Errors
///
/// [`PlanError`] when the repo holds no such weights, holds several GGUF
/// files, or names a path that leaves the model directory.
pub fn plan(files: &[String]) -> Result<PullPlan, PlanError> {
    let holds = |name: &str| files.iter().any(|f| f == name);
    let mut out: Vec<String> = SIDE_FILES
        .iter()
        .filter(|f| holds(f))
        .map(|f| (*f).to_string())
        .collect();
    let mut sharded = false;
    if holds(SAFETENSORS) {
        out.push(SAFETENSORS.to_string());
    } else if holds(SAFETENSORS_INDEX) {
        out.push(SAFETENSORS_INDEX.to_string());
        sharded = true;
    } else {
        let gguf: Vec<String> = files
            .iter()
            .filter(|f| f.to_ascii_lowercase().ends_with(".gguf"))
            .cloned()
            .collect();
        match gguf.len() {
            0 => return Err(PlanError::NoWeights),
            1 => out.extend(gguf),
            _ => return Err(PlanError::SeveralGguf(gguf)),
        }
    }
    if let Some(bad) = out.iter().find(|f| !stays_in_dir(f)) {
        return Err(PlanError::Unsafe(bad.clone()));
    }
    Ok(PullPlan {
        files: out,
        sharded,
    })
}

/// The shard files the SafeTensors index `json` names, sorted and once each.
///
/// # Errors
///
/// A message when the index does not parse, names no shard, or names a
/// shard that leaves the model directory.
pub fn shards_of(json: &str) -> Result<Vec<String>, String> {
    let index: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| format!("{SAFETENSORS_INDEX} does not parse: {e}"))?;
    let mut shards: Vec<String> = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .map(|map| {
            map.values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err(format!("{SAFETENSORS_INDEX} names no shard in weight_map"));
    }
    if let Some(bad) = shards.iter().find(|s| !stays_in_dir(s)) {
        return Err(format!(
            "{SAFETENSORS_INDEX} names the shard '{bad}', which leaves the model directory"
        ));
    }
    Ok(shards)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn a_safetensors_repo_fetches_its_side_files_and_weights() {
        let files = names(&[
            "README.md",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "model.safetensors",
        ]);
        let got = plan(&files).expect("plan");
        assert_eq!(
            got.files,
            names(&[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "model.safetensors"
            ])
        );
        assert!(!got.sharded);
    }

    #[test]
    fn a_sharded_repo_fetches_its_index() {
        let files = names(&[
            "config.json",
            SAFETENSORS_INDEX,
            "model-00001-of-00002.safetensors",
        ]);
        let got = plan(&files).expect("plan");
        assert_eq!(got.files, names(&["config.json", SAFETENSORS_INDEX]));
        assert!(got.sharded);
    }

    /// The old guess built `<repo name>.Q4_K_M.gguf` from the lowercased
    /// repo name, `-gguf` suffix included, so it named no real file.
    #[test]
    fn a_gguf_repo_fetches_the_file_it_holds() {
        let files = names(&["README.md", "qwen2.5-0.5b-instruct-q4_k_m.gguf"]);
        let got = plan(&files).expect("plan");
        assert_eq!(got.files, names(&["qwen2.5-0.5b-instruct-q4_k_m.gguf"]));
    }

    #[test]
    fn several_gguf_files_need_a_pick() {
        let files = names(&["m-Q4_K_M.gguf", "m-Q8_0.gguf"]);
        let err = plan(&files).expect_err("no pick");
        assert_eq!(err, PlanError::SeveralGguf(files.clone()));
        let text = err.message("org/m-GGUF", "Pass the one to fetch with --file <name>");
        assert_eq!(
            text,
            "org/m-GGUF holds 2 GGUF files: m-Q4_K_M.gguf, m-Q8_0.gguf. Pass the one to fetch \
             with --file <name>"
        );
    }

    #[test]
    fn a_repo_without_weights_is_an_error() {
        let err = plan(&names(&["config.json", "README.md"])).expect_err("no weights");
        assert_eq!(err, PlanError::NoWeights);
    }

    #[test]
    fn a_path_that_leaves_the_directory_is_refused() {
        let err = plan(&names(&["../evil.gguf"])).expect_err("unsafe");
        assert_eq!(err, PlanError::Unsafe("../evil.gguf".into()));
    }

    #[test]
    fn shards_come_from_the_index_once_each() {
        let json = r#"{"weight_map": {"a": "m-2.safetensors", "b": "m-1.safetensors", "c": "m-1.safetensors"}}"#;
        assert_eq!(
            shards_of(json).expect("shards"),
            names(&["m-1.safetensors", "m-2.safetensors"])
        );
        assert!(shards_of(r#"{"weight_map": {}}"#).is_err());
        assert!(shards_of("not json").is_err());
        assert!(shards_of(r#"{"weight_map": {"a": "/etc/passwd"}}"#).is_err());
    }
}
