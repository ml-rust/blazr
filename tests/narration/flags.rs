//! The narration skill's flags, as `blazr speak` and `blazr transcribe`
//! take them: every flag of the skill's `parseArgs`, read from the skill's
//! own source, is taken by both commands, and a value flag given last
//! reads as the skill reads it.
//!
//! The skill's source is `$NARRATION_JS_DIR/../video.mjs`, else
//! `~/.claude/skills/motion-video/scripts/video.mjs`. Without it the
//! source check skips with a notice.

use std::path::PathBuf;

use voxr_narrate::{FlagKind, SKILL_FLAGS};

use super::sandbox::Sandbox;

/// The skill's `video.mjs`, when it is on disk.
fn skill_source() -> Option<String> {
    let from_env = std::env::var_os("NARRATION_JS_DIR")
        .filter(|d| !d.is_empty())
        .map(|d| PathBuf::from(d).join("..").join("video.mjs"));
    let from_home = std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join(".claude/skills/motion-video/scripts/video.mjs"));
    from_env
        .into_iter()
        .chain(from_home)
        .find(|p| p.is_file())
        .and_then(|p| std::fs::read_to_string(p).ok())
}

/// Every flag `parseArgs` names, with how it reads its value, from the
/// skill's source.
fn parse_args_flags(source: &str) -> Vec<(String, FlagKind)> {
    let start = source
        .find("function parseArgs(")
        .unwrap_or_else(|| panic!("video.mjs holds no parseArgs"));
    let body = &source[start..];
    let body = &body[..body.find("\n}\n").unwrap_or(body.len())];
    let mut out = Vec::new();
    for line in body.lines() {
        let kind = if let Some(at) = line.find("num(a, argv[++i], ") {
            let rest = &line[at + "num(a, argv[++i], ".len()..];
            let min: f64 = rest[..rest.find(')').unwrap_or(rest.len())]
                .trim()
                .parse()
                .unwrap_or_else(|e| panic!("the bound in {line:?}: {e}"));
            FlagKind::Number { min }
        } else if line.contains(".push(argv[++i])") {
            FlagKind::List
        } else if line.contains("argv[++i].split(") {
            FlagKind::Parts
        } else if line.contains("argv[++i]") {
            FlagKind::Text
        } else {
            FlagKind::Switch
        };
        let mut rest = line;
        while let Some(at) = rest.find("a === '") {
            let tail = &rest[at + "a === '".len()..];
            let end = tail.find('\'').unwrap_or(tail.len());
            out.push((tail[..end].to_string(), kind));
            rest = &tail[end..];
        }
    }
    out
}

/// A value the flag takes: its lower bound for a number.
fn value_for(name: &str, kind: FlagKind) -> Option<String> {
    match kind {
        FlagKind::Switch => None,
        FlagKind::Number { min } => Some(min.to_string()),
        FlagKind::Parts => Some("1".to_string()),
        FlagKind::Text if name == "--written" => Some("w.txt".to_string()),
        FlagKind::Text | FlagKind::List => Some("x".to_string()),
    }
}

#[test]
fn every_skill_flag_is_taken_by_both_commands() {
    let Some(source) = skill_source() else {
        eprintln!("skipping: the skill's video.mjs unavailable; set NARRATION_JS_DIR");
        return;
    };
    let flags = parse_args_flags(&source);
    assert!(flags.len() > 30, "parseArgs read as {flags:?}");
    let ours: Vec<(String, FlagKind)> = SKILL_FLAGS
        .iter()
        .filter(|f| f.skill)
        .map(|f| (f.name.to_string(), f.kind))
        .collect();
    assert_eq!(
        ours, flags,
        "the parser's flags differ from the skill's parseArgs"
    );

    let sb = Sandbox::empty();
    sb.write("w.txt", "Hello there.\n");
    for (name, kind) in &flags {
        let mut speak = vec!["speak", "Hello there.", "--review", name.as_str()];
        let mut transcribe = vec!["transcribe", "gone.wav", name.as_str()];
        let value = value_for(name, *kind);
        if let Some(v) = &value {
            speak.push(v);
            transcribe.push(v);
        }
        let s = sb.run(&speak);
        let t = sb.run(&transcribe);
        if name == "-h" || name == "--help" {
            assert_eq!(s.code, 0, "{name}: {}", s.stderr);
            assert!(s.stdout.starts_with("Narration from a script"), "{name}");
            assert_eq!(t.code, 0, "{name}: {}", t.stderr);
            assert!(t.stdout.starts_with("Word timings"), "{name}");
            continue;
        }
        assert_eq!(s.code, 0, "speak {name}: {}", s.stderr);
        assert!(
            s.stdout.starts_with("Script review"),
            "speak {name}: {}",
            s.stdout
        );
        assert_eq!(t.code, 2, "transcribe {name}: {}", t.stderr);
        assert!(
            t.stderr.trim().ends_with(" does not exist."),
            "transcribe {name}: {}",
            t.stderr
        );
    }
    assert!(sb.project().join("assets").read_dir().is_err());
}

#[test]
fn a_value_flag_given_last_reads_as_the_skill_reads_it() {
    let sb = Sandbox::empty();
    // Text: left unset, the default holds.
    let voice = sb.run(&["speak", "Hello there.", "--review", "--voice"]);
    assert_eq!(voice.code, 0, "{}", voice.stderr);
    // A value flag takes the next argument whatever it is.
    let taken = sb.run(&["speak", "Hello there.", "--voice", "--review", "--review"]);
    assert_eq!(taken.code, 0, "{}", taken.stderr);
    // Number: `Number(undefined)` fails its bound.
    let pace = sb.run(&["speak", "Hello there.", "--pace"]);
    assert_eq!(pace.code, 2);
    assert_eq!(
        pace.stderr.trim(),
        "error: --pace needs a number of at least 2, got \"undefined\"."
    );
    // --reroll: the skill's split of an undefined value fails the run.
    let reroll = sb.run(&["speak", "Hello there.", "--reroll"]);
    assert_eq!(reroll.code, 1, "{}", reroll.stderr);
    // An unknown option, `--` included, is a bad input.
    let dashes = sb.run(&["speak", "--", "Hello there."]);
    assert_eq!(dashes.code, 2);
    assert_eq!(
        dashes.stderr.lines().next(),
        Some("error: Unknown option --.")
    );
    // --manifest: files are read alone, whatever video.json holds.
    sb.write("video.json", "not a manifest");
    let loose = sb.run(&["transcribe", "gone.wav", "--manifest"]);
    assert_eq!(loose.code, 2);
    assert!(
        loose.stderr.trim().ends_with("gone.wav does not exist."),
        "{}",
        loose.stderr
    );
    // With no file, the run fails.
    let none = sb.run(&["transcribe", "--manifest"]);
    assert_eq!(none.code, 1);
    assert!(
        none.stderr
            .starts_with("Error: transcribe --manifest: no path follows it."),
        "{}",
        none.stderr
    );
}
