//! `blazr sound-review` on a small rendered catalog: the table, the CLAP
//! ranks, `--no-model`, the exit codes, and a missing model offline.

use std::path::Path;

use super::sandbox::Sandbox;
use crate::common::{model_fixture, skip_notice};

/// A catalog of three sounds under `dir`: two base variants and a second
/// variant, each a 0.3 s fading tone at 48 kHz.
fn catalog(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{e}"));
    for (label, freq) in [("pop_v0", 440.0), ("pop_v1", 520.0), ("boom_v0", 70.0)] {
        let n = 14_400usize;
        let pcm: Vec<i16> = (0..n)
            .map(|i| {
                let fade = 1.0 - i as f64 / n as f64;
                let t = i as f64 / 48_000.0;
                (12_000.0 * fade * (2.0 * std::f64::consts::PI * freq * t).sin()) as i16
            })
            .collect();
        let wav = voxr::encode_wav_pcm16_values(&pcm, 48_000).unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(dir.join(format!("{label}.wav")), wav).unwrap_or_else(|e| panic!("{e}"));
    }
    let index = r#"{"pop_v0": {"says": "a short soft pop bubble sound", "endsOnCue": false},
        "pop_v1": {"says": "a short soft pop bubble sound", "endsOnCue": false},
        "boom_v0": {"says": "a deep cinematic boom", "endsOnCue": false}}"#;
    std::fs::write(dir.join("sounds.json"), index).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn the_review_prints_the_table_and_ranks() {
    let Some(clap) = model_fixture("CLAP_DIR", "larger_clap_general") else {
        skip_notice("laion/larger_clap_general checkpoint", "CLAP_DIR");
        return;
    };
    let sb = Sandbox::with_models(&[("larger_clap_general", &clap)]);
    catalog(&sb.project().join("sounds"));
    let run = sb.run(&["sound-review", "sounds"]);
    assert_eq!(run.code, 0, "{}", run.stderr);
    let lines: Vec<&str> = run.stdout.lines().collect();
    assert_eq!(
        lines[0],
        "sound                  sharp   hiss   tail    tp  rank    verdict"
    );
    assert_eq!(lines.len(), 6, "{}", run.stdout);
    for (line, label) in lines[1..4].iter().zip(["pop_v0", "pop_v1", "boom_v0"]) {
        assert!(line.starts_with(&format!("{label:<22}")), "{line}");
        assert!(line.ends_with("ok"), "{line}");
    }
    assert!(lines[1].contains("/2  "), "{}", lines[1]);
    assert!(!lines[2].contains('/'), "{}", lines[2]);
    assert_eq!(lines[4], "");
    assert_eq!(lines[5], "3 sounds, 0 failing a numeric limit.");

    let skipped = sb.run(&["sound-review", "sounds", "--no-model"]);
    assert_eq!(skipped.code, 0, "{}", skipped.stderr);
    assert!(!skipped.stdout.contains("/2"), "{}", skipped.stdout);
}

#[test]
fn usage_and_a_missing_catalog_exit_as_the_script_does() {
    let sb = Sandbox::empty();
    let usage = sb.run(&["sound-review"]);
    assert_eq!(usage.code, 2);
    assert!(usage
        .stdout
        .contains("\n  blazr sound-review <dir> [--no-model]\n"));
    let missing = sb.run(&["sound-review", "nowhere", "--no-model"]);
    assert_eq!(missing.code, 1);
    assert_eq!(
        missing.stderr.trim(),
        "FileNotFoundError: [Errno 2] No such file or directory: 'nowhere/sounds.json'"
    );
}

#[test]
fn a_missing_model_offline_names_the_download() {
    let sb = Sandbox::empty();
    catalog(&sb.project().join("sounds"));
    let run = sb.run_with(&["sound-review", "sounds"], &[("HF_HUB_OFFLINE", "1")]);
    assert_eq!(run.code, 1, "{}", run.stdout);
    let err = run.stderr.trim();
    assert!(
        err.starts_with(
            "sound-review: the CLAP model laion/larger_clap_general is not on disk. Searched: "
        ),
        "{err}"
    );
    assert!(
        err.ends_with("Download it with `hf download laion/larger_clap_general`."),
        "{err}"
    );
    assert_eq!(run.stdout, "");
}
