//! `blazr serve`'s TTS loader on a Kokoro checkpoint that holds no `g2p/`:
//! its G2P models come from the Hub repo `fs90/kokoro-g2p` through the
//! same resolution `blazr speak` runs, downloaded on first use and found
//! after, offline.
//!
//! Needs `KOKORO_DIR` (or `Kokoro-82M` under `BOOSTR_MODELS_DIR`) and the
//! Hub: it skips with a notice when either is missing. The download goes
//! into a fresh hub cache, `HF_HOME` set for this test's own process.
#![cfg(feature = "audio")]

mod common;

use blazr::loader::load_tts_from_dir;
use common::{hub_online, model_fixture, skip_notice, symlink};
use voxr::voxcpm::VoxCpm2SynthOptions;
use voxr::SynthesizeOptions;

#[test]
fn the_serve_loader_resolves_kokoro_g2p_without_a_g2p_folder() {
    let Some(kokoro) = model_fixture("KOKORO_DIR", "Kokoro-82M") else {
        skip_notice("Kokoro-82M checkpoint", "KOKORO_DIR");
        return;
    };
    if !hub_online() {
        eprintln!("skipping: the Hub is out of reach; fs90/kokoro-g2p cannot download");
        return;
    }
    let bare = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
    for name in ["config.json", "kokoro-v1_0.pth", "voices"] {
        symlink(&kokoro.join(name), &bare.path().join(name));
    }
    // This file holds one test, so it owns its process's environment: the
    // hub cache is fresh, and no model folder holds the repo.
    let hub = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
    let empty = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
    std::env::set_var("HF_HOME", hub.path());
    std::env::set_var("BOOSTR_MODELS_DIR", empty.path());
    std::env::remove_var("HF_HUB_CACHE");
    std::env::remove_var("HF_HUB_OFFLINE");

    let first = load_tts_from_dir(bare.path(), VoxCpm2SynthOptions::default())
        .unwrap_or_else(|e| panic!("loading Kokoro with the repo downloaded: {e:#}"));
    assert!(first.has_engine());
    let snapshots = hub.path().join("hub/models--fs90--kokoro-g2p/snapshots");
    assert!(
        snapshots.is_dir(),
        "no download under {}",
        hub.path().display()
    );
    assert!(
        !bare.path().join("g2p").exists(),
        "the checkpoint was written into"
    );

    // Offline, the repo is found where it was downloaded.
    std::env::set_var("HF_HUB_OFFLINE", "1");
    let again = load_tts_from_dir(bare.path(), VoxCpm2SynthOptions::default())
        .unwrap_or_else(|e| panic!("loading Kokoro with the repo on disk: {e:#}"));
    let audio = again
        .synthesize(
            "The river runs north.",
            "af_heart",
            &SynthesizeOptions::default(),
        )
        .unwrap_or_else(|e| panic!("synthesizing: {e}"));
    assert!(!audio.is_empty());
}
