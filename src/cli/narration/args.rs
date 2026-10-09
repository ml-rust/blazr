//! `blazr speak` and `blazr transcribe` arguments: read as the narration
//! skill's `parseArgs` reads them ([`parse_skill_args`]), so the skill
//! hands its own arguments on unchanged. Each command takes every flag of
//! the skill's CLI, and leaves the ones it does not read.

use clap::Args;
use voxr_narrate::{FlagKind, SKILL_FLAGS};

/// `blazr speak` arguments, as given.
#[derive(Args, Debug, Clone)]
#[command(disable_help_flag = true)]
pub struct SpeakArgs {
    /// The script and the flags, read as the narration skill reads them.
    /// `blazr speak --help` lists the flags.
    #[arg(
        value_name = "ARGS",
        num_args = 0..,
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub argv: Vec<String>,
}

/// `blazr transcribe` arguments, as given.
#[derive(Args, Debug, Clone)]
#[command(disable_help_flag = true)]
pub struct TranscribeArgs {
    /// The files and the flags, read as the narration skill reads them.
    /// `blazr transcribe --help` lists the flags.
    #[arg(
        value_name = "ARGS",
        num_args = 0..,
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub argv: Vec<String>,
}

/// `blazr sound-review` arguments, as given: the rendered catalog's
/// directory and `--no-model`, read as the skill's script reads them.
#[derive(Args, Debug, Clone)]
#[command(disable_help_flag = true)]
pub struct SoundReviewArgs {
    /// The directory `render-sounds.mjs` wrote, then `--no-model` to skip
    /// the CLAP ranking.
    #[arg(
        value_name = "ARGS",
        num_args = 0..,
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub argv: Vec<String>,
}

/// The flags `speak` reads, with what each does.
const SPEAK_FLAGS: &[(&str, &str)] = &[
    ("--engine kokoro|voxcpm", "The engine. Default: VoxCPM2 with its GPU and checkpoint, else Kokoro."),
    ("--voice V", "A Kokoro voice (af_heart), or a VoxCPM2 voice description such as \"(young woman, warm, low)\"."),
    ("--style S", "VoxCPM2's delivery for every part, such as \"(calm)\"."),
    ("--reference WAV", "A recording to clone (VoxCPM2), and the command's {reference}."),
    ("--speed S", "The speed, 0.5 to 2. VoxCPM2 is never time-stretched."),
    ("--pace P", "The target pace in syllables per second, at least 2. Default 4.1."),
    ("--seed N", "The seed of the voice and the takes, at least 0."),
    ("--reroll N,M", "VoxCPM2 parts to speak again, from 1, such as 3,7."),
    ("--cfg G", "VoxCPM2's guidance, at least 0.5. Default 1.6."),
    ("--steps N", "VoxCPM2's diffusion steps, at least 1. Default 16."),
    ("--hifi", "VoxCPM2 clones with the reference's transcript."),
    ("--written FILE", "The written script: the same words, punctuated for captions."),
    ("--review", "Print the script review and stop, before any audio."),
    ("--raw", "Skip the voice polish."),
    ("--device D", "The device, such as cpu. Set, VoxCPM2 runs where it is not ready."),
    ("--language L", "The script's language, which picks the recognizer."),
    ("--model HF_ID", "A Hugging Face TTS model: a VITS (MMS-TTS) id or directory. Any other exits 3."),
    ("--command TEMPLATE", "Any TTS, run once per phrase: {text}, {text_file}, {out}, {voice} and {reference} are filled in."),
    ("--one-call", "Run --command once for the whole script, pause marks dropped."),
    ("--out WAV", "The voiceover. Default assets/voiceover.wav."),
    ("--json", "Print one JSON object on standard output: the run's result. Every other line goes to standard error."),
];

/// The flags `transcribe` reads, with what each does.
const TRANSCRIBE_FLAGS: &[(&str, &str)] = &[
    ("--model M", "The recognizer: a Whisper id or directory. Default: whisper-base for English, whisper-small for other languages, or the config's \"asr.model\"."),
    ("--language L", "The language spoken. Default en."),
    ("--script TXT", "The one narration's script: its words replace the recognized ones, on the recognized timings."),
    ("--manifest PATH", "The video manifest. Default ./video.json."),
    ("--json", "Print one JSON object on standard output: every words file. Every other line goes to standard error."),
];

/// A command's help: its usage, the flags it reads, and the skill's other
/// flags it takes and leaves.
fn help(usage: &str, about: &str, flags: &[(&str, &str)]) -> String {
    let mut out = format!("{about}\n\nUsage: {usage}\n\nOptions:\n");
    for (name, what) in flags {
        out.push_str(&format!("  {name:<20} {what}\n"));
    }
    out.push_str("  -h, --help           Print this help.\n");
    let read: Vec<&str> = flags
        .iter()
        .filter_map(|(f, _)| f.split(' ').next())
        .collect();
    let left: Vec<String> = SKILL_FLAGS
        .iter()
        .filter(|f| f.skill && !read.contains(&f.name) && f.name != "-h" && f.name != "--help")
        .map(|f| match f.kind {
            FlagKind::Switch => f.name.to_string(),
            _ => format!("{} V", f.name),
        })
        .collect();
    out.push_str(&format!(
        "\nTaken and left, as the narration skill takes them for its other commands:\n  {}",
        left.join(", ")
    ));
    out
}

/// `blazr speak --help`.
pub fn speak_help() -> String {
    help(
        "blazr speak [TEXT|FILE]... [OPTIONS]",
        "Narration from a script: the voiceover WAV, its script, its phrase spans, and every phrase heard back with Whisper. The script is a text file, or the words. A script with times per line places each line at its time. Exits 1 when a problem remains, 2 on a bad input, 3 on a model that runs elsewhere.",
        SPEAK_FLAGS,
    )
}

/// `blazr transcribe --help`.
pub fn transcribe_help() -> String {
    help(
        "blazr transcribe [FILE]... [OPTIONS]",
        "Word timings for narration from a local Whisper model: one words file per narration, and the video timeline from a manifest. With no FILE: the manifest's voiceover and every scene audio, on the video clock. Exits 2 on a bad input.",
        TRANSCRIBE_FLAGS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_flag_a_command_reads_is_a_skill_flag() {
        for (name, _) in SPEAK_FLAGS.iter().chain(TRANSCRIBE_FLAGS) {
            let flag = name.split(' ').next().unwrap_or_default();
            assert!(
                SKILL_FLAGS.iter().any(|f| f.name == flag),
                "{flag} is in the help, not in the parser"
            );
        }
        let text = speak_help();
        assert!(text.contains("  --voice V            A Kokoro voice"));
        assert!(text.contains("--manifest V, --scene V, --draft,"));
        assert!(!transcribe_help().contains(" --manifest V,"));
    }
}
