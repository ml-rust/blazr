//! `blazr speak` with Kokoro: the review, a bad input, and a script
//! spoken, heard back and polished, with its files and `--json`.

use super::sandbox::{wav_format, whisper, Sandbox};
use crate::common::{model_fixture, skip_notice};

const SCRIPT: &str = "The river runs north past the old mill, and the town wakes slowly.";

fn sandbox() -> Option<Sandbox> {
    let (base, small) = whisper()?;
    let Some(kokoro) = model_fixture("KOKORO_DIR", "Kokoro-82M") else {
        skip_notice("Kokoro-82M checkpoint", "KOKORO_DIR");
        return None;
    };
    Some(Sandbox::new(&base, &small, &[("Kokoro-82M", &kokoro)]))
}

#[test]
fn review_and_bad_input_stop_before_any_audio() {
    let Some(sb) = sandbox() else { return };
    let review = sb.run(&["speak", "Use the --raw flag", "--review"]);
    assert_eq!(review.code, 0, "{}", review.stderr);
    assert!(review
        .stdout
        .starts_with("Script review, 2 note(s) (narration.md, Prepare the script for speech):\n"));
    let bad = sb.run(&["speak", SCRIPT, "--pace", "1"]);
    assert_eq!(bad.code, 2);
    assert_eq!(
        bad.stderr.trim(),
        "error: --pace needs a number of at least 2, got \"1\"."
    );
    let empty = sb.run(&["speak"]);
    assert_eq!(empty.code, 2);
    assert_eq!(
        empty.stderr.trim(),
        "error: speak needs the script: a text file, or the words in quotes."
    );
    assert!(!sb.project().join("assets").exists());
}

#[test]
fn kokoro_speaks_a_script_and_hears_it_back() {
    let Some(sb) = sandbox() else { return };
    sb.write("script.txt", &format!("{SCRIPT}\n"));
    let run = sb.run(&["speak", "script.txt", "--engine", "kokoro"]);
    assert_eq!(
        run.code, 0,
        "stdout:\n{}\nstderr:\n{}",
        run.stdout, run.stderr
    );
    let lines: Vec<&str> = run.stdout.lines().collect();
    assert!(
        lines[0].starts_with("Voice polish: loudness evened in "),
        "{lines:?}"
    );
    assert!(lines[1].starts_with("Pace: "), "{lines:?}");
    assert!(lines[2].starts_with("assets/voiceover.wav ("), "{lines:?}");
    assert!(lines[2].ends_with("s, Kokoro (af_heart)) and its script assets/voiceover.txt"));
    assert!(lines.contains(&"Heard back, phrase by phrase:"));
    assert_eq!(
        lines.last(),
        Some(&"Speech check: every phrase was heard back as written.")
    );
    let wav = std::fs::read(sb.project().join("assets/voiceover.wav")).unwrap();
    assert_eq!(wav_format(&wav), (1, 48_000, 16));
    assert_eq!(sb.read("assets/voiceover.txt"), format!("{SCRIPT}\n"));
    let phrases: serde_json::Value =
        serde_json::from_str(&sb.read("assets/voiceover.phrases.json")).unwrap();
    assert_eq!(phrases["script"], SCRIPT);
    assert!(!phrases["phrases"].as_array().unwrap().is_empty());

    let json = sb.run(&[
        "speak",
        SCRIPT,
        "--engine",
        "kokoro",
        "--out",
        "vo/take.wav",
        "--json",
    ]);
    assert_eq!(json.code, 0, "{}", json.stderr);
    let v = json.json();
    assert_eq!(v["engine"], "Kokoro (af_heart)");
    assert_eq!(v["exitCode"], 0);
    assert_eq!(v["problems"], serde_json::json!([]));
    assert_eq!(v["checks"][0]["ok"], true);
    assert!(v["duration"].as_f64().unwrap() > 1.0);
    let out = sb.project().join("vo/take.wav");
    assert_eq!(v["files"]["out"], out.to_string_lossy().as_ref());
    assert_eq!(
        v["files"]["phrases"],
        sb.project()
            .join("vo/take.phrases.json")
            .to_string_lossy()
            .as_ref()
    );
    assert!(v["pace"]["rate"].as_f64().unwrap() > 0.0);
    assert!(v["polish"]["leveled"].is_u64());
    assert!(json
        .stderr
        .contains("Speech check: every phrase was heard back as written."));
}
