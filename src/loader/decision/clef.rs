//! A Clef directory: a Hugging Face Qwen3.5 checkpoint with the
//! joint-schema head beside its weights. Its detection and the head's
//! config.

use std::path::Path;

use boostr::model::decision::joint_schema::config::{CONFIG_FILE, WEIGHTS_FILE};
use boostr::model::decision::JointSchemaConfig;

/// The head's sizes.
pub const HEAD_CONFIG: &str = CONFIG_FILE;
/// The head's weights.
pub const HEAD_WEIGHTS: &str = WEIGHTS_FILE;
/// The dtype the head computes in: every head weight is cast to F32 once
/// at load, whatever the backbone dtype. The release runs the head in the
/// backbone's BF16.
pub const HEAD_DTYPE: &str = "f32";

/// Whether `dir` holds the joint-schema head: [`HEAD_CONFIG`] and
/// [`HEAD_WEIGHTS`].
pub fn has_joint_head(dir: &Path) -> bool {
    dir.join(HEAD_CONFIG).is_file() && dir.join(HEAD_WEIGHTS).is_file()
}

/// The head config of the Clef directory `dir`.
///
/// # Errors
///
/// The file and the reason when [`HEAD_CONFIG`] does not read, parse or
/// check.
pub fn read_head_config(dir: &Path) -> Result<JointSchemaConfig, String> {
    JointSchemaConfig::from_file(dir.join(HEAD_CONFIG)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASE: &str = r#"{"hidden_size": 4096, "width": 1024, "routing_layers": 2,
        "layers": 4, "heads": 16, "feedforward": 4096}"#;

    #[test]
    fn the_head_needs_both_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        assert!(!has_joint_head(dir.path()));
        std::fs::write(dir.path().join(HEAD_CONFIG), RELEASE).expect("write");
        assert!(!has_joint_head(dir.path()), "the config alone");
        std::fs::write(dir.path().join(HEAD_WEIGHTS), b"").expect("write");
        assert!(has_joint_head(dir.path()));
        let config = read_head_config(dir.path()).expect("reads");
        assert_eq!((config.hidden_size, config.width), (4096, 1024));

        std::fs::write(dir.path().join(HEAD_CONFIG), r#"{"width": 8}"#).expect("write");
        let e = read_head_config(dir.path()).expect_err("incomplete");
        assert!(e.contains(HEAD_CONFIG), "{e}");
    }
}
