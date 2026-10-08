//! Kokoro's G2P models: the spaCy tagger and the out-of-vocabulary model
//! under `g2p/` beside a Kokoro checkpoint, else the Hugging Face repo
//! [`KOKORO_G2P_ID`], found on disk or downloaded on first use. The repo is
//! read where it sits: nothing is written into the Kokoro checkpoint.

use std::path::Path;

use voxr::kokoro::KokoroG2pModels;

use crate::loader::hub::{ModelKind, ModelSearch, KOKORO_G2P_ID, OOV_FILES, TAGGER_FILES};

/// Whether `dir` holds every one of `files`.
fn holds(dir: &Path, files: &[&str]) -> bool {
    files.iter().all(|f| dir.join(f).is_file())
}

/// The G2P models for the Kokoro checkpoint at `kokoro`: its own `g2p/`
/// when it holds both models, else the repo [`KOKORO_G2P_ID`] for each
/// model it lacks, found by `search` or downloaded. The error is the
/// reason with the next action, led by `what`, such as `speak`.
pub fn kokoro_g2p(
    kokoro: &Path,
    search: &ModelSearch,
    what: &str,
) -> Result<KokoroG2pModels, String> {
    let tagger = kokoro.join("g2p/tagger");
    let oov = kokoro.join("g2p/oov");
    let (has_tagger, has_oov) = (holds(&tagger, &TAGGER_FILES), holds(&oov, &OOV_FILES));
    if has_tagger && has_oov {
        return Ok(KokoroG2pModels {
            tagger_dir: tagger,
            oov_dir: Some(oov),
        });
    }
    let repo = search.obtain(what, KOKORO_G2P_ID, ModelKind::KokoroG2p, None)?;
    Ok(KokoroG2pModels {
        tagger_dir: if has_tagger {
            tagger
        } else {
            repo.join("tagger")
        },
        oov_dir: Some(if has_oov { oov } else { repo.join("oov") }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, files: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        for f in files {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
    }

    #[test]
    fn a_checkpoint_with_its_own_g2p_reads_it() {
        let tmp = tempfile::tempdir().unwrap();
        let k = tmp.path();
        touch(&k.join("g2p/tagger"), &TAGGER_FILES);
        touch(&k.join("g2p/oov"), &OOV_FILES);
        let search = ModelSearch::new(Vec::new(), Vec::new(), k);
        let g2p = kokoro_g2p(k, &search, "speak").unwrap();
        assert_eq!(g2p.tagger_dir, k.join("g2p/tagger"));
        assert_eq!(g2p.oov_dir, Some(k.join("g2p/oov")));
    }

    #[test]
    fn a_missing_repo_offline_names_the_download() {
        let tmp = tempfile::tempdir().unwrap();
        let search = ModelSearch::new(Vec::new(), Vec::new(), tmp.path());
        // No hub cache: nothing is found, and nothing can download into it
        // offline. The reason names the repo's download.
        let err = search
            .find(KOKORO_G2P_ID, ModelKind::KokoroG2p)
            .err()
            .unwrap_or_default();
        assert!(
            err.ends_with(&format!("Download it with `hf download {KOKORO_G2P_ID}`.")),
            "{err}"
        );
    }
}
