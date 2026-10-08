//! `blazr speak` with VoxCPM2 from a GGUF checkpoint: a designed voice
//! speaks a script, each part heard back, with the take record and the
//! designed voice beside the output.

use super::sandbox::{wav_format, whisper, Sandbox};
use crate::common::{model_fixture, skip_notice};

const SCRIPT: &str = "The river runs north past the old mill, and the town wakes slowly.";

#[test]
fn voxcpm_designs_a_voice_and_speaks() {
    let Some((base, small)) = whisper() else {
        return;
    };
    let Some(gguf) = model_fixture("VOXCPM2_GGUF", "") else {
        skip_notice("VoxCPM2 GGUF", "VOXCPM2_GGUF");
        return;
    };
    let sb = Sandbox::new(&base, &small, &[]);
    sb.config(&serde_json::json!({ "tts": { "checkpoint": gguf } }));
    let run = sb.run(&[
        "speak",
        SCRIPT,
        "--engine",
        "voxcpm",
        "--voice",
        "(a calm man in his forties, low, warm voice)",
        "--seed",
        "7",
        "--json",
    ]);
    let v = run.json();
    assert_eq!(v["exitCode"], run.code, "{}", run.stderr);
    assert_eq!(
        run.code, 0,
        "problems: {}\nstderr:\n{}",
        v["problems"], run.stderr
    );
    assert!(v["engine"]
        .as_str()
        .unwrap()
        .starts_with("VoxCPM2 (a calm man"));
    assert!(v["checks"][0]["seed"].as_f64().is_some());
    let wav = std::fs::read(sb.project().join("assets/voiceover.wav")).unwrap();
    assert_eq!(wav_format(&wav), (1, 48_000, 16));
    let takes: serde_json::Value =
        serde_json::from_str(&sb.read("assets/voiceover.takes.json")).unwrap();
    assert_eq!(takes["seed"], 7);
    assert!(sb.project().join("assets/voiceover.voice.wav").is_file());
    assert_eq!(
        v["voiceFile"],
        sb.project()
            .join("assets/voiceover.voice.wav")
            .to_string_lossy()
            .as_ref()
    );
    // A rerun reuses every take from the shared cache.
    let again = sb.run(&[
        "speak",
        SCRIPT,
        "--engine",
        "voxcpm",
        "--voice",
        "(a calm man in his forties, low, warm voice)",
        "--seed",
        "7",
    ]);
    assert_eq!(again.code, 0, "{}", again.stderr);
    assert!(!again.stderr.contains("is speaking"), "{}", again.stderr);
}
