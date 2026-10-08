//! `blazr speak` and `blazr transcribe` flags, one for one with the
//! narration skill's own `speak` and `transcribe`.

use clap::Args;
use voxr_narrate::{flag_number, reroll_parts, CommandError, SpeakFlags, TranscribeFlags};

/// `blazr speak` flags.
#[derive(Args, Debug, Clone)]
pub struct SpeakArgs {
    /// The script: a text file, or the words. Several words join with
    /// spaces. A script with times per line places each line at its time.
    #[arg(value_name = "TEXT|FILE", required = false)]
    pub script: Vec<String>,

    /// The engine: kokoro or voxcpm. Default: VoxCPM2 with its GPU and
    /// checkpoint, else Kokoro.
    #[arg(long, value_name = "kokoro|voxcpm")]
    pub engine: Option<String>,

    /// A Kokoro voice (af_heart), or a VoxCPM2 voice description such as
    /// "(young woman, warm, low)".
    #[arg(long, value_name = "V")]
    pub voice: Option<String>,

    /// VoxCPM2's delivery for every part, such as "(calm)".
    #[arg(long, value_name = "S")]
    pub style: Option<String>,

    /// A recording to clone (VoxCPM2), and the command's {reference}.
    #[arg(long, value_name = "WAV")]
    pub reference: Option<String>,

    /// The speed, 0.5 to 2. VoxCPM2 is never time-stretched.
    #[arg(long, value_name = "S")]
    pub speed: Option<String>,

    /// The target pace in syllables per second, at least 2. Default 4.1.
    #[arg(long, value_name = "P")]
    pub pace: Option<String>,

    /// The seed of the voice and the takes, at least 0.
    #[arg(long, value_name = "N")]
    pub seed: Option<String>,

    /// VoxCPM2 parts to speak again, from 1, such as 3,7.
    #[arg(long, value_name = "N,M")]
    pub reroll: Option<String>,

    /// VoxCPM2's guidance, at least 0.5. Default 1.6.
    #[arg(long, value_name = "G")]
    pub cfg: Option<String>,

    /// VoxCPM2's diffusion steps, at least 1. Default 16.
    #[arg(long, value_name = "N")]
    pub steps: Option<String>,

    /// VoxCPM2 clones with the reference's transcript.
    #[arg(long)]
    pub hifi: bool,

    /// The written script: the same words, punctuated for captions.
    #[arg(long, value_name = "FILE")]
    pub written: Option<String>,

    /// Print the script review and stop, before any audio.
    #[arg(long)]
    pub review: bool,

    /// Skip the voice polish.
    #[arg(long)]
    pub raw: bool,

    /// The device, such as cpu. Set, VoxCPM2 runs where it is not ready.
    #[arg(long, value_name = "D")]
    pub device: Option<String>,

    /// The script's language, which picks the recognizer.
    #[arg(long, value_name = "L")]
    pub language: Option<String>,

    /// A Hugging Face TTS model: a VITS (MMS-TTS) id or directory.
    #[arg(long, value_name = "HF_ID")]
    pub model: Option<String>,

    /// Any TTS, run once per phrase: {text}, {text_file}, {out}, {voice}
    /// and {reference} are filled in.
    #[arg(long, value_name = "TEMPLATE")]
    pub command: Option<String>,

    /// Run --command once for the whole script, pause marks dropped.
    #[arg(long = "one-call")]
    pub one_call: bool,

    /// The voiceover. Default assets/voiceover.wav.
    #[arg(long, value_name = "WAV")]
    pub out: Option<String>,

    /// Print one JSON object on standard output: the run's result. Every
    /// other line goes to standard error.
    #[arg(long)]
    pub json: bool,
}

/// A flag's number, checked against its lower bound.
fn number(flag: &str, value: &Option<String>, min: f64) -> Result<Option<f64>, CommandError> {
    value
        .as_deref()
        .map(|v| flag_number(flag, v, min))
        .transpose()
}

impl SpeakArgs {
    /// The flags as the skill's `parseArgs` leaves them. A number below
    /// its bound is a bad input.
    pub fn flags(&self) -> Result<SpeakFlags, CommandError> {
        let steps = number("--steps", &self.steps, 1.0)?.map(|n| n.round() as u32);
        Ok(SpeakFlags {
            script: self.script.clone(),
            engine: self.engine.clone(),
            voice: self.voice.clone(),
            style: self.style.clone(),
            reference: self.reference.clone(),
            speed: number("--speed", &self.speed, 0.5)?,
            pace: number("--pace", &self.pace, 2.0)?,
            seed: number("--seed", &self.seed, 0.0)?,
            reroll: self.reroll.as_deref().map(reroll_parts).transpose()?,
            cfg: number("--cfg", &self.cfg, 0.5)?,
            steps,
            hifi: self.hifi,
            written: self.written.clone(),
            review: self.review,
            raw: self.raw,
            device: self.device.clone(),
            language: self.language.clone(),
            model: self.model.clone(),
            command: self.command.clone(),
            one_call: self.one_call,
            out: self.out.clone(),
        })
    }
}

/// `blazr transcribe` flags.
#[derive(Args, Debug, Clone)]
pub struct TranscribeArgs {
    /// Audio files to read. None: the manifest's voiceover and every scene
    /// audio, on the video clock.
    #[arg(value_name = "FILE")]
    pub files: Vec<String>,

    /// The recognizer: a Whisper id or directory. Default: whisper-base
    /// for English, whisper-small for other languages, or the config's
    /// "asr.model".
    #[arg(long, value_name = "M")]
    pub model: Option<String>,

    /// The language spoken. Default en.
    #[arg(long, value_name = "L")]
    pub language: Option<String>,

    /// The one narration's script: its words replace the recognized ones,
    /// on the recognized timings.
    #[arg(long, value_name = "TXT")]
    pub script: Option<String>,

    /// The video manifest. Default ./video.json.
    #[arg(long, value_name = "PATH")]
    pub manifest: Option<String>,

    /// Print one JSON object on standard output: every words file. Every
    /// other line goes to standard error.
    #[arg(long)]
    pub json: bool,
}

impl TranscribeArgs {
    /// The flags as the skill's `parseArgs` leaves them.
    pub fn flags(&self) -> TranscribeFlags {
        TranscribeFlags {
            files: self.files.clone(),
            model: self.model.clone(),
            language: self.language.clone(),
            script: self.script.clone(),
            manifest: self.manifest.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Speak {
        #[command(flatten)]
        args: SpeakArgs,
    }

    fn parse(argv: &[&str]) -> SpeakArgs {
        Speak::parse_from(std::iter::once("speak").chain(argv.iter().copied())).args
    }

    #[test]
    fn numbers_are_checked_against_their_bounds() {
        let ok = parse(&[
            "One", "two.", "--speed", "0.9", "--steps", "9.6", "--reroll", "2,3",
        ]);
        let f = ok.flags().unwrap();
        assert_eq!(f.script, ["One", "two."]);
        assert_eq!(f.speed, Some(0.9));
        assert_eq!(f.steps, Some(10));
        assert_eq!(f.reroll, Some(vec![2, 3]));
        let low = parse(&["Hi.", "--pace", "1"]);
        assert_eq!(
            low.flags().err(),
            Some(CommandError::Input(
                "--pace needs a number of at least 2, got \"1\".".to_string()
            ))
        );
    }
}
