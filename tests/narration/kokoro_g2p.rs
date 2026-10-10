//! `blazr speak` with a Kokoro checkpoint that holds no `g2p/`: its G2P
//! models come from the Hub repo `fs90/kokoro-g2p`, downloaded on first
//! use and found after. Offline, the run names the `hf download` next
//! action.
//!
//! The download test needs the Hub: it skips with a notice when
//! `HF_HUB_OFFLINE` is set or huggingface.co is out of reach.

use std::path::Path;

use super::sandbox::{whisper, Sandbox};
use crate::common::{hub_online, model_fixture, skip_notice, symlink};

const SCRIPT: &str = "The river runs north past the old mill.";
const REPO: &str = "fs90/kokoro-g2p";

/// A Kokoro checkpoint without `g2p/`: links to `kokoro`'s config, weights
/// and voices in a fresh directory.
fn bare_kokoro(kokoro: &Path) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
    for name in ["config.json", "kokoro-v1_0.pth", "voices"] {
        symlink(&kokoro.join(name), &dir.path().join(name));
    }
    dir
}

/// The sandbox and the bare checkpoint, linked into its model folder as
/// `Kokoro-82M`.
fn setup() -> Option<(Sandbox, tempfile::TempDir)> {
    let (base, small) = whisper()?;
    let Some(kokoro) = model_fixture("KOKORO_DIR", "Kokoro-82M") else {
        skip_notice("Kokoro-82M checkpoint", "KOKORO_DIR");
        return None;
    };
    let bare = bare_kokoro(&kokoro);
    let sb = Sandbox::new(&base, &small, &[("Kokoro-82M", bare.path())]);
    Some((sb, bare))
}

#[test]
fn a_kokoro_checkpoint_without_g2p_downloads_it_then_finds_it() {
    let Some((sb, _bare)) = setup() else { return };
    if !hub_online() {
        eprintln!("skipping: the Hub is out of reach; {REPO} cannot download");
        return;
    }
    let first = sb.run(&["speak", SCRIPT, "--engine", "kokoro", "--out", "a.wav"]);
    assert_eq!(first.code, 0, "{}", first.stderr);
    assert!(
        first.stderr.contains(&format!(
            "speak: no Kokoro G2P model {REPO} on disk. Downloading it into "
        )),
        "{}",
        first.stderr
    );
    let again = sb.run(&["speak", SCRIPT, "--engine", "kokoro", "--out", "b.wav"]);
    assert_eq!(again.code, 0, "{}", again.stderr);
    assert!(!again.stderr.contains("Downloading"), "{}", again.stderr);
    assert!(again
        .stdout
        .ends_with("Speech check: every phrase was heard back as written.\n"));
}

#[test]
fn a_kokoro_checkpoint_without_g2p_offline_names_the_download() {
    let Some((sb, _bare)) = setup() else { return };
    let run = sb.run_with(
        &["speak", SCRIPT, "--engine", "kokoro"],
        &[("HF_HUB_OFFLINE", "1")],
    );
    assert_eq!(run.code, 1, "{}", run.stderr);
    let line = run.stderr.trim();
    assert!(
        line.starts_with(&format!(
            "Error: speak: the Kokoro G2P model {REPO} is not on disk. Searched: "
        )),
        "{line}"
    );
    assert!(
        line.ends_with(&format!("Download it with `hf download {REPO}`.")),
        "{line}"
    );
    assert!(!line.contains("convert_tagger"), "{line}");
    sb.assert_untouched();
}
