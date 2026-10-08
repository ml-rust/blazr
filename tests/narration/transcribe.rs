//! `blazr transcribe` on a Kokoro voiceover: a loose file with its script
//! and the spans speak recorded, then the manifest's narration on the
//! video timeline.

use super::sandbox::{whisper, Sandbox};
use crate::common::{model_fixture, skip_notice};

const SCRIPT: &str = "The river runs north past the old mill, and the town wakes slowly.";

#[test]
fn transcribe_reads_a_file_and_a_manifest() {
    let Some((base, small)) = whisper() else {
        return;
    };
    let Some(kokoro) = model_fixture("KOKORO_DIR", "Kokoro-82M") else {
        skip_notice("Kokoro-82M checkpoint", "KOKORO_DIR");
        return;
    };
    let sb = Sandbox::new(&base, &small, &[("Kokoro-82M", &kokoro)]);
    let spoken = sb.run(&["speak", SCRIPT, "--engine", "kokoro"]);
    assert_eq!(spoken.code, 0, "{}", spoken.stderr);

    let run = sb.run(&[
        "transcribe",
        "assets/voiceover.wav",
        "--script",
        "assets/voiceover.txt",
        "--json",
    ]);
    assert_eq!(run.code, 0, "{}", run.stderr);
    let v = run.json();
    let n = &v["narrations"][0];
    assert_eq!(n["where"], "file");
    assert_eq!(n["model"], "onnx-community/whisper-base_timestamped");
    let words: Vec<&str> = n["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["text"].as_str().unwrap())
        .collect();
    assert_eq!(words.join(" "), SCRIPT);
    let project = sb.project();
    assert_eq!(
        n["phrases"],
        project
            .join("assets/voiceover.phrases.json")
            .to_string_lossy()
            .as_ref()
    );
    let file: serde_json::Value =
        serde_json::from_str(&sb.read("out/voice/voiceover.words.json")).unwrap();
    assert_eq!(file["words"], n["words"]);
    assert!(v.get("timeline").is_none());
    assert!(run.stderr.contains("  file seconds, start-end, word:"));

    sb.write(
        "video.json",
        r#"{"scenes": [{"file": "scenes/01-a.html", "duration": 9}], "voiceover": {"file": "assets/voiceover.wav", "start": 0.5, "script": "assets/voiceover.txt"}}"#,
    );
    let manifest = sb.run(&["transcribe"]);
    assert_eq!(manifest.code, 0, "{}", manifest.stderr);
    let lines: Vec<&str> = manifest.stdout.lines().collect();
    assert!(lines[0].contains("(voiceover, "), "{lines:?}");
    assert_eq!(lines[1], "  video seconds, start-end, word:");
    assert!(lines
        .last()
        .is_some_and(|l| l.starts_with("video timeline -> ")));
    let timeline: serde_json::Value =
        serde_json::from_str(&sb.read("out/voice/words.json")).unwrap();
    let first = timeline["words"][0]["start"].as_f64().unwrap();
    assert!(
        first >= 0.5,
        "the voiceover starts at 0.5 s, its first word at {first}"
    );

    let missing = sb.run(&["transcribe", "gone.wav"]);
    assert_eq!(missing.code, 2);
    assert!(
        missing.stderr.trim().ends_with("gone.wav does not exist."),
        "{}",
        missing.stderr
    );
}
