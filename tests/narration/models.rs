//! The models `blazr speak` refuses: a `--model` that is no VITS
//! checkpoint or holds no `model.safetensors`, exit 3, and a missing model
//! offline, with the `hf download` next action.

use super::sandbox::{whisper, Sandbox};

const SCRIPT: &str = "The river runs north past the old mill.";

#[test]
fn a_model_that_is_not_vits_exits_three() {
    let Some((base, small)) = whisper() else {
        return;
    };
    let sb = Sandbox::new(&base, &small, &[]);
    // A Whisper checkpoint: a Hugging Face model with a config.json that
    // is no VITS one. A path naming Kokoro speaks with Kokoro instead.
    let model = base.to_string_lossy();
    let run = sb.run(&["speak", SCRIPT, "--model", &model]);
    assert_eq!(run.code, 3, "{}", run.stderr);
    assert_eq!(
        run.stderr.trim(),
        format!(
            "error: speak --model {model}: only VITS (MMS-TTS) checkpoints run natively, and {model} is not one. Run it with the skill's own runner, or pass --command."
        )
    );
    sb.assert_untouched();
}

#[test]
fn a_vits_model_with_onnx_weights_only_exits_three() {
    let Some((base, small)) = whisper() else {
        return;
    };
    let sb = Sandbox::new(&base, &small, &[]);
    let onnx = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
    std::fs::write(onnx.path().join("config.json"), r#"{"model_type": "vits"}"#)
        .unwrap_or_else(|e| panic!("{e}"));
    std::fs::create_dir(onnx.path().join("onnx")).unwrap_or_else(|e| panic!("{e}"));
    std::fs::write(onnx.path().join("onnx/model.onnx"), b"onnx").unwrap_or_else(|e| panic!("{e}"));
    let model = onnx.path().to_string_lossy();
    // A script with review notes: the refusal comes before the review
    // prints, so a caller that speaks it another way prints it once.
    let run = sb.run_with(
        &["speak", "Use the --raw flag", "--model", &model],
        &[("HF_HUB_OFFLINE", "1")],
    );
    assert_eq!(run.code, 3, "{}", run.stderr);
    assert_eq!(run.stdout, "", "{}", run.stderr);
    assert_eq!(
        run.stderr.trim(),
        format!(
            "error: speak --model {model}: only VITS (MMS-TTS) checkpoints with model.safetensors run natively, and {model} holds other weights. Run it with the skill's own runner, or pass --command."
        )
    );
    sb.assert_untouched();
}

#[test]
fn a_missing_model_offline_names_the_download() {
    let Some((base, small)) = whisper() else {
        return;
    };
    let sb = Sandbox::new(&base, &small, &[]);
    let run = sb.run_with(
        &["speak", SCRIPT, "--engine", "kokoro"],
        &[("HF_HUB_OFFLINE", "1")],
    );
    assert_eq!(run.code, 1, "{}", run.stderr);
    let line = run.stderr.trim();
    assert!(
        line.starts_with("Error: speak: the Kokoro model "),
        "{line}"
    );
    assert!(line.contains(" is not on disk. Searched: "), "{line}");
    let searched = line
        .split(" is not on disk. Searched: ")
        .nth(1)
        .and_then(|rest| rest.split(". Download").next())
        .unwrap_or_default();
    let dirs: Vec<&str> = searched.split(", ").collect();
    let once: std::collections::BTreeSet<&str> = dirs.iter().copied().collect();
    assert_eq!(dirs.len(), once.len(), "a directory is named twice: {line}");
    assert!(line.contains("Download it with `hf download "), "{line}");
    assert!(!line.contains("Downloading it into"), "{line}");
    sb.assert_untouched();
}
