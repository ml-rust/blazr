//! Model-fixture resolution, the hub reachability check, and skip notices.

#[cfg(feature = "audio")]
use std::path::Path;
use std::path::PathBuf;

/// Whether the Hugging Face hub is in reach: `HF_HUB_OFFLINE` unset, and
/// huggingface.co answers on port 443 within 5 s. A test that downloads
/// skips with a notice otherwise.
pub fn hub_online() -> bool {
    use std::net::{TcpStream, ToSocketAddrs};
    if std::env::var_os("HF_HUB_OFFLINE").is_some_and(|v| !v.is_empty() && v != "0") {
        return false;
    }
    ("huggingface.co", 443)
        .to_socket_addrs()
        .ok()
        .and_then(|mut a| a.next())
        .is_some_and(|addr| {
            TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(5)).is_ok()
        })
}

/// Resolve a model fixture: `$<specific_var>`, else `$BOOSTR_MODELS_DIR/<relative>`.
/// Pass `relative = ""` for a fixture with no fallback under the models root,
/// such as a single GGUF file. Returns `None` when neither is set or the
/// resolved path is absent on disk, so callers skip.
pub fn model_fixture(specific_var: &str, relative: &str) -> Option<PathBuf> {
    let path = match std::env::var(specific_var) {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ if relative.is_empty() => return None,
        _ => {
            let root = std::env::var("BOOSTR_MODELS_DIR")
                .ok()
                .filter(|r| !r.is_empty())?;
            PathBuf::from(root).join(relative)
        }
    };
    path.exists().then_some(path)
}

/// Symlink `link` to `target`, a file or a directory. Windows needs
/// Developer Mode or an elevated shell to create a symlink.
#[cfg(feature = "audio")]
pub fn symlink(target: &Path, link: &Path) {
    #[cfg(unix)]
    let made = std::os::unix::fs::symlink(target, link);
    #[cfg(windows)]
    let made = if target.is_dir() {
        std::os::windows::fs::symlink_dir(target, link)
    } else {
        std::os::windows::fs::symlink_file(target, link)
    };
    made.unwrap_or_else(|e| panic!("linking {} to {}: {e}", link.display(), target.display()));
}

/// Print the standard skip notice naming both the specific env var and
/// `BOOSTR_MODELS_DIR`, so the reader knows how to enable the test.
pub fn skip_notice(what: &str, specific_var: &str) {
    eprintln!("skipping: {what} unavailable; set {specific_var} or BOOSTR_MODELS_DIR");
}

/// `whisper-tiny` checkpoint directory: `$WHISPER_TINY_DIR`, else
/// `$BOOSTR_MODELS_DIR/whisper-tiny`, else the HF hub cache layout
/// `huggingface_hub.snapshot_download` produces (`~/.cache/huggingface/hub/
/// models--openai--whisper-tiny/snapshots/<rev>`) — the same fallback
/// voxr's `whisper_word_oracle` test uses for `whisper-small` when no
/// local copy is staged under `$BOOSTR_MODELS_DIR`.
pub fn whisper_tiny_dir() -> Option<PathBuf> {
    if let Some(dir) = model_fixture("WHISPER_TINY_DIR", "whisper-tiny") {
        return Some(dir);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--openai--whisper-tiny/snapshots");
    let mut entries = std::fs::read_dir(&snapshots).ok()?;
    entries.find_map(|e| e.ok()).map(|e| e.path())
}

/// A short real speech clip for the ASR route tests:
/// `$BLAZR_ASR_TEST_WAV`, else `$BOOSTR_MODELS_DIR/librispeech_en_sample.wav`.
pub fn asr_test_wav() -> Option<PathBuf> {
    model_fixture("BLAZR_ASR_TEST_WAV", "librispeech_en_sample.wav")
}
