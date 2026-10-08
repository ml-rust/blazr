//! `blazr transcribe`: the narration skill's `transcribe` command. blazr
//! reads the config, finds and loads the recognizer, decodes each file,
//! and prints. voxr-narrate does the rest.

use std::path::Path;

use voxr_narrate::{
    check_narration, decode_16k, narration_lines, narration_words, overrun_line, parse_skill_args,
    plan_transcribe, probe_duration, transcribe_json, transcribe_model, transcribe_narration,
    write_narration, write_video_timeline, CommandError, ReadNarration, SkillArgs,
    TranscribeOptions,
};

use super::args::transcribe_help;
use super::config::read_config;
use super::hearing::{load_whisper, Loaded, Placement};
use super::paths::{model_search, working_dir};
use super::print::Printer;
use crate::loader::hub::ModelKind;

/// Runs `blazr transcribe` on `argv`, the arguments after the command,
/// and returns its exit code: 0 when every narration was read, 1 when a
/// run failed, 2 on a bad input.
pub fn transcribe(argv: &[String]) -> i32 {
    let args = match parse_skill_args(argv, &transcribe_help()) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("{}", e.line());
            return e.exit_code();
        }
    };
    if args.help() {
        println!("{}", transcribe_help());
        return 0;
    }
    let printer = Printer {
        json: args.on("--json"),
    };
    match run(&args, printer) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{}", e.line());
            e.exit_code()
        }
    }
}

fn failed(e: &voxr_narrate::Error) -> CommandError {
    CommandError::Failed(format!("transcribe: {}", e.reason()))
}

fn run(args: &SkillArgs, printer: Printer) -> Result<i32, CommandError> {
    let flags = args.transcribe_flags();
    let cwd = working_dir()?;
    let config = read_config()?;
    let mut probe = |p: &Path| probe_duration(p);
    let plan = plan_transcribe(&flags, &cwd, &mut probe)?;
    let model = transcribe_model(
        flags.model.as_deref(),
        flags.language.as_deref(),
        config.asr_model.as_deref(),
    );
    let search = model_search(&config, &cwd);
    let options = TranscribeOptions {
        language: flags.language.clone(),
    };
    let mut recognizer: Option<Loaded> = None;
    let mut read = Vec::with_capacity(plan.narrations.len());
    for n in &plan.narrations {
        check_narration(n)?;
        let transcriber = match &mut recognizer {
            Some(t) => t,
            None => {
                let dir = search
                    .obtain("transcribe", &model, ModelKind::Whisper, None)
                    .map_err(CommandError::Failed)?;
                recognizer
                    .insert(load_whisper(&dir, Placement::pick(None)).map_err(|e| failed(&e))?)
            }
        };
        let pcm = decode_16k(&n.file).map_err(|e| failed(&e))?;
        let heard = transcribe_narration(&pcm, transcriber, &options).map_err(|e| failed(&e))?;
        let words = narration_words(n, &model, heard, &pcm)?;
        let out = write_narration(&plan.out_dir, n, &words)?;
        for line in narration_lines(n, &words, &out) {
            printer.out(&line);
        }
        read.push(ReadNarration {
            narration: n.clone(),
            out,
            words,
        });
    }
    let timeline = match (&plan.manifest, plan.timeline) {
        (Some(m), true) => {
            let (path, line) = write_video_timeline(&m.narration)?;
            printer.out(&line);
            Some(path)
        }
        _ => None,
    };
    let overrun = plan.manifest.as_ref().map_or(0.0, |m| m.voiceover_overrun);
    if let Some(line) = overrun_line(overrun) {
        printer.out(&line);
    }
    if printer.json {
        let value = transcribe_json(&read, timeline.as_deref(), overrun).map_err(|e| failed(&e))?;
        printer.json(&value);
    }
    Ok(0)
}
