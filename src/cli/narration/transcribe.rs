//! `blazr transcribe`: the narration skill's `transcribe` command. blazr
//! reads the config, finds and loads the recognizer, decodes each file,
//! and prints. voxr-narrate does the rest.

use std::path::Path;

use voxr_narrate::{
    check_narration, decode_16k, narration_lines, narration_words, overrun_line, plan_transcribe,
    probe_duration, transcribe_json, transcribe_model, transcribe_narration, write_narration,
    write_video_timeline, CommandError, ReadNarration, TranscribeOptions,
};

use super::args::TranscribeArgs;
use super::config::read_config;
use super::hearing::{load_whisper, Loaded, Placement};
use super::kinds::ModelKind;
use super::models::ModelSearch;
use super::paths::working_dir;
use super::print::Printer;

/// Runs `blazr transcribe` and returns its exit code: 0 when every
/// narration was read, 1 when a run failed, 2 on a bad input.
pub fn transcribe(args: &TranscribeArgs) -> i32 {
    let printer = Printer { json: args.json };
    match run(args, printer) {
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

fn run(args: &TranscribeArgs, printer: Printer) -> Result<i32, CommandError> {
    let flags = args.flags();
    let cwd = working_dir()?;
    let config = read_config()?;
    let mut probe = |p: &Path| probe_duration(p);
    let plan = plan_transcribe(&flags, &cwd, &mut probe)?;
    let model = transcribe_model(
        flags.model.as_deref(),
        flags.language.as_deref(),
        config.asr_model.as_deref(),
    );
    let search = ModelSearch::new(&config, &cwd);
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
                let dir = search.obtain("transcribe", &model, ModelKind::Whisper, None)?;
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
