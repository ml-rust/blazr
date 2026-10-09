//! `blazr speak --model` with a VITS (MMS-TTS) checkpoint: a Malay script
//! spoken and heard back. A raw take is 48 kHz, as the skill's Python
//! runner hands it on.

use super::sandbox::{wav_format, whisper, Sandbox};
use crate::common::{model_fixture, skip_notice};

const MODEL: &str = "facebook/mms-tts-zlm";
const SCRIPT: &str = "Selamat pagi semua. Sungai itu mengalir ke utara.";

#[test]
fn a_vits_model_speaks_at_the_skill_rate() {
    let Some((base, small)) = whisper() else {
        return;
    };
    let Some(mms) = model_fixture("MMS_TTS_ZLM_DIR", MODEL) else {
        skip_notice("facebook/mms-tts-zlm checkpoint", "MMS_TTS_ZLM_DIR");
        return;
    };
    let sb = Sandbox::new(&base, &small, &[("mms-tts-zlm", &mms)]);
    let run = sb.run(&[
        "speak",
        SCRIPT,
        "--model",
        MODEL,
        "--language",
        "ms",
        "--raw",
        "--json",
    ]);
    let v = run.json();
    assert_eq!(v["engine"], MODEL, "{}", run.stderr);
    assert_eq!(v["exitCode"], run.code, "{}", run.stderr);
    let wav = std::fs::read(sb.project().join("assets/voiceover.wav")).unwrap();
    assert_eq!(wav_format(&wav), (1, 48_000, 16));
    assert_eq!(v["phrases"].as_array().map(Vec::len), Some(2));
}

/// A VITS checkpoint whose `tokenizer_config.json` sets `is_uroman: true`.
const UROMAN_MODEL: &str = "facebook/mms-tts-kor";
const UROMAN_SCRIPT: &str = "안녕하세요. 만나서 반갑습니다.";

#[test]
fn a_uroman_vits_model_speaks() {
    let Some((base, small)) = whisper() else {
        return;
    };
    let Some(mms) = model_fixture("MMS_TTS_KOR_DIR", UROMAN_MODEL) else {
        skip_notice("facebook/mms-tts-kor checkpoint", "MMS_TTS_KOR_DIR");
        return;
    };
    let sb = Sandbox::new(&base, &small, &[("mms-tts-kor", &mms)]);
    let run = sb.run(&[
        "speak",
        UROMAN_SCRIPT,
        "--model",
        UROMAN_MODEL,
        "--language",
        "ko",
        "--raw",
        "--json",
    ]);
    assert_eq!(run.code, 0, "{}", run.stderr);
    let v = run.json();
    assert_eq!(v["engine"], UROMAN_MODEL, "{}", run.stderr);
    assert_eq!(v["exitCode"], 0, "{}", run.stderr);
    let wav = std::fs::read(sb.project().join("assets/voiceover.wav")).unwrap();
    assert_eq!(wav_format(&wav), (1, 48_000, 16));
    assert_eq!(v["phrases"].as_array().map(Vec::len), Some(2));
}
