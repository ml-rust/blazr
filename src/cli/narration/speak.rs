//! `blazr speak`: the narration skill's `speak` command. blazr reads the
//! config, finds and loads the models, checks VoxCPM2's readiness, runs
//! the user's TTS command, and prints. voxr-narrate does the rest.

use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::json;
use voxr_narrate::{
    asr_model, decode_48k, finish_speak, prepare_speak, resolve_speak, speak_failure, CommandError,
    Error, FinishInputs, PreparedSpeak, Result, SpeakFlags, SpeakTarget, TtsConfig,
};

use super::args::SpeakArgs;
use super::config::{read_config, MotionConfig};
use super::engines::{load_engines, run_speak, voxcpm_checkpoint, RunInput};
use super::gpu::{gpu, voxcpm_readiness};
use super::hearing::{hearing, Placement};
use super::models::ModelSearch;
use super::paths::{take_cache, working_dir};
use super::print::Printer;

/// Runs `line` in the shell, as Node's `spawnSync(cmd, { shell: true })`
/// does: no input, its output on this process's own output, or on
/// standard error with `--json`. The exit code, `None` when a signal ends
/// it.
fn spawn(line: &str, json: bool) -> Result<Option<i32>> {
    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", line]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", line]);
        c
    };
    cmd.stdin(Stdio::null());
    if json {
        cmd.stdout(Stdio::from(std::io::stderr()));
    }
    let status = cmd.status().map_err(|e| Error::Engine {
        reason: format!(
            "running the TTS command {line}: {e}. Check that a shell runs it from here"
        ),
    })?;
    Ok(status.code())
}

/// Runs `blazr speak` and returns its exit code: 0 when every phrase was
/// heard back as written, 1 when a problem remains or the run failed, 2
/// on a bad input.
pub fn speak(args: &SpeakArgs) -> i32 {
    let printer = Printer { json: args.json };
    match run(args, printer) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{}", e.line());
            e.exit_code()
        }
    }
}

fn run(args: &SpeakArgs, printer: Printer) -> std::result::Result<i32, CommandError> {
    let flags = args.flags()?;
    let cwd = working_dir()?;
    let config = read_config()?;
    let tts = config.tts.config();
    let search = ModelSearch::new(&config, &cwd);
    let checkpoint = voxcpm_checkpoint(&config, &search, &cwd);
    let ready = voxcpm_readiness(gpu());
    let mut lines = Vec::new();
    let prepared = prepare_speak(&flags, &tts, ready.ok, &ready.why, &cwd, &mut lines);
    printer.lines(&lines);
    let prepared = prepared?;
    if prepared.review_only {
        if printer.json {
            printer.json(&json!({ "review": prepared.review, "exitCode": 0 }));
        }
        return Ok(0);
    }
    let spoken = speak_run(
        &flags,
        &prepared,
        &config,
        &search,
        &checkpoint,
        &ready,
        printer,
    )?;
    let inputs = FinishInputs {
        flags: &flags,
        prepared: &prepared,
        tts: &tts,
        tts_pace: config.tts.pace,
        cwd: &cwd,
    };
    let mut lines = Vec::new();
    let mut decode = |p: &Path| decode_48k(p);
    let outcome = finish_speak(&spoken, &inputs, &mut decode, &mut lines);
    printer.lines(&lines);
    let outcome = outcome?;
    if printer.json {
        let value = outcome
            .json(&spoken, &prepared.review)
            .map_err(|e| speak_failure(&e, &prepared.opts(&flags)))?;
        printer.json(&value);
    }
    Ok(outcome.code)
}

/// Loads what the run speaks and hears with, and speaks the script.
fn speak_run(
    flags: &SpeakFlags,
    prepared: &PreparedSpeak,
    config: &MotionConfig,
    search: &ModelSearch,
    checkpoint: &std::result::Result<std::path::PathBuf, String>,
    ready: &super::gpu::Readiness,
    printer: Printer,
) -> std::result::Result<voxr_narrate::SpokenScript, CommandError> {
    let tts: TtsConfig<'_> = config.tts.config();
    let opts = prepared.opts(flags);
    let failed = |e: Error| speak_failure(&e, &opts);
    let resolution =
        resolve_speak(&prepared.script, &opts, &tts, ready.ok, &ready.why).map_err(failed)?;
    let at = Placement::pick(opts.device);
    let engines = load_engines(&resolution.engine, search, config, checkpoint, at)?;
    let configured = config.asr_model.as_deref();
    let main = asr_model(opts.language, configured);
    let hearing = hearing(search, configured, main, at)?;
    // The output directory is made only once every refusal has passed: a
    // refused run leaves the project as it was.
    if let Some(dir) = prepared.out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| {
            CommandError::Failed(format!(
                "creating {}: {e}. Check the directory is writable, or pass another --out",
                dir.display()
            ))
        })?;
    }
    let cache = take_cache();
    std::fs::create_dir_all(&cache).map_err(|e| {
        CommandError::Failed(format!(
            "creating the take cache {}: {e}. Check the cache directory is writable",
            cache.display()
        ))
    })?;
    // A run that downloaded its checkpoint names none, as the skill's does.
    let checkpoint_text = checkpoint
        .as_ref()
        .ok()
        .filter(|_| !engines.downloaded)
        .map(|p| p.to_string_lossy().into_owned());
    let target = SpeakTarget {
        out: &prepared.out,
        cache_dir: &cache,
        checkpoint: checkpoint_text.as_deref(),
        voxcpm_ready: ready.ok,
        voxcpm_why: &ready.why,
        asr_model: configured,
    };
    let input = RunInput {
        script: &prepared.script,
        opts: &opts,
        tts: &tts,
        target: &target,
    };
    let json = printer.json;
    run_speak(
        &engines,
        hearing,
        move |line: &str| spawn(line, json),
        &input,
    )
    .map_err(failed)
}
