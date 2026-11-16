//! API keys: the ones `blazr serve` checks, `--api-key` or
//! `$BLAZR_API_KEY` and the lines of `--api-key-file`, and the one key
//! `blazr ps` sends.

use std::path::Path;

use anyhow::{bail, Result};

/// The keys of `flag_key` and of the file at `key_file`, the flag key first.
/// An empty list runs the server without auth.
///
/// # Errors
///
/// An error when `flag_key` is empty or blank, when the file does not
/// read, and when the file holds no key: auth the operator asked for never
/// turns off in silence.
pub fn collect_api_keys(flag_key: Option<String>, key_file: Option<&Path>) -> Result<Vec<String>> {
    let mut keys = Vec::new();
    if let Some(key) = flag_key {
        keys.push(checked_flag_key(key)?);
    }
    if let Some(path) = key_file {
        let from_file = read_key_file(path)?;
        tracing::info!(
            "Loaded {} API key(s) from {}",
            from_file.len(),
            path.display()
        );
        keys.extend(from_file);
    }
    Ok(keys)
}

/// The one key a client sends: `flag_key`, else the first key of the file
/// at `key_file`, else `env_key` (`$BLAZR_API_KEY`). `None` sends no key.
///
/// # Errors
///
/// The errors of [`collect_api_keys`]: an empty `flag_key`, a file that
/// does not read, or a file with no key.
pub fn client_api_key(
    flag_key: Option<String>,
    key_file: Option<&Path>,
    env_key: Option<String>,
) -> Result<Option<String>> {
    if let Some(key) = flag_key {
        return checked_flag_key(key).map(Some);
    }
    if let Some(path) = key_file {
        return Ok(read_key_file(path)?.into_iter().next());
    }
    Ok(env_key.filter(|k| !k.trim().is_empty()))
}

/// `key`, or the error for an empty or blank `--api-key`.
fn checked_flag_key(key: String) -> Result<String> {
    if key.trim().is_empty() {
        bail!("--api-key is empty. Pass a key, or omit --api-key to run without auth");
    }
    Ok(key)
}

/// The keys of the key file at `path`.
///
/// # Errors
///
/// An error naming `path` when it does not read or holds no key.
fn read_key_file(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to read API key file {}: {e}. Check the path and its read permission",
            path.display()
        )
    })?;
    let keys = keys_in(&content);
    if keys.is_empty() {
        bail!(
            "API key file {} holds no key: every line is empty or a # comment. Add one key \
             per line, or omit --api-key-file",
            path.display()
        );
    }
    Ok(keys)
}

/// The keys in a key file's `content`: each line, trimmed, that is not
/// empty and does not start with `#`.
fn keys_in(content: &str) -> Vec<String> {
    content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(content: &str) -> tempfile::NamedTempFile {
        let f = tempfile::NamedTempFile::new().expect("temp file");
        std::fs::write(f.path(), content).expect("write");
        f
    }

    /// An empty or comment-only file used to leave the key list empty, so
    /// the server ran without auth.
    #[test]
    fn a_key_file_with_no_key_is_an_error() {
        for content in ["", "\n\n", "# keys\n  # none yet\n"] {
            let f = file(content);
            let err = collect_api_keys(None, Some(f.path()))
                .expect_err("no key")
                .to_string();
            assert!(err.contains(&f.path().display().to_string()), "{err}");
            assert!(err.contains("holds no key"), "{err}");
            // A flag key does not hide the empty file.
            assert!(collect_api_keys(Some("k".into()), Some(f.path())).is_err());
        }
    }

    /// `--api-key ""` used to be a key that no client could send.
    #[test]
    fn an_empty_flag_key_is_an_error() {
        for key in ["", "   "] {
            let err = collect_api_keys(Some(key.into()), None).expect_err("empty key");
            assert!(err.to_string().starts_with("--api-key is empty"), "{err}");
        }
    }

    #[test]
    fn keys_come_from_the_flag_then_the_file() {
        let f = file("# team keys\n  alpha \n\nbeta\n");
        let keys = collect_api_keys(Some("flag".into()), Some(f.path())).expect("keys");
        assert_eq!(keys, ["flag", "alpha", "beta"]);
        assert!(collect_api_keys(None, None).expect("no auth").is_empty());
    }

    #[test]
    fn an_unreadable_file_is_an_error() {
        let err = collect_api_keys(None, Some(Path::new("/no/such/blazr/keys")))
            .expect_err("missing file");
        assert!(err.to_string().contains("/no/such/blazr/keys"), "{err}");
    }

    #[test]
    fn a_client_sends_the_flag_then_the_file_then_the_variable() {
        let f = file("# keys\nfirst\nsecond\n");
        let pick = |flag: Option<&str>, path: Option<&Path>, env: Option<&str>| {
            client_api_key(flag.map(String::from), path, env.map(String::from))
        };
        assert_eq!(
            pick(Some("flag"), Some(f.path()), Some("env")).expect("key"),
            Some("flag".into())
        );
        assert_eq!(
            pick(None, Some(f.path()), Some("env")).expect("key"),
            Some("first".into())
        );
        assert_eq!(
            pick(None, None, Some("env")).expect("key"),
            Some("env".into())
        );
        assert_eq!(pick(None, None, Some("  ")).expect("no key"), None);
        assert_eq!(pick(None, None, None).expect("no key"), None);
    }

    #[test]
    fn a_client_key_is_never_empty() {
        let err =
            client_api_key(Some(String::new()), None, Some("env".into())).expect_err("empty flag");
        assert!(err.to_string().starts_with("--api-key is empty"), "{err}");
        let f = file("# none\n\n");
        let err = client_api_key(None, Some(f.path()), None).expect_err("empty file");
        assert!(err.to_string().contains("holds no key"), "{err}");
    }
}
