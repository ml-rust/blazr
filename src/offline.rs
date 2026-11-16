//! Offline mode: `HF_HUB_OFFLINE` blocks every outbound request blazr makes.
//!
//! The guarded requests are the Hugging Face hub fetches (`blazr pull`,
//! `POST /api/pull`, the speech models' first-use download, the Kokoro G2P
//! files) and the remote image URLs a chat request can name. A blocked
//! request fails with an [`OfflineError`] that names what it needed and
//! the offline fix.

use std::fmt;

/// The variable that turns offline mode on.
pub const OFFLINE_VAR: &str = "HF_HUB_OFFLINE";

/// Whether offline mode is on: `$HF_HUB_OFFLINE` set to any non-empty
/// value but `0` or `false` (any case).
pub fn is_offline() -> bool {
    offline_value(std::env::var(OFFLINE_VAR).ok().as_deref())
}

/// Whether the `HF_HUB_OFFLINE` value `value` turns offline mode on.
pub fn offline_value(value: Option<&str>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
}

/// What a blocked request needed, and how to work without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Need {
    /// A file or the file list of a Hugging Face model repo.
    HubModel,
    /// An image a request names by `http://` or `https://` URL.
    RemoteImage,
}

/// A request that offline mode blocked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineError {
    /// The model, file or URL the request needed.
    pub target: String,
    /// The kind of request.
    pub need: Need,
}

impl fmt::Display for OfflineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fix = match self.need {
            Need::HubModel => {
                "Copy the model to this host and pass its local path, or unset HF_HUB_OFFLINE \
                 on a host with network access"
            }
            Need::RemoteImage => {
                "Send the image inline as a data: URL or base64, or unset HF_HUB_OFFLINE"
            }
        };
        write!(
            f,
            "{} needs the network, and HF_HUB_OFFLINE is set. {fix}",
            self.target
        )
    }
}

impl std::error::Error for OfflineError {}

/// `Err` when offline mode is on: the request for `target` must not run.
pub fn guard(target: &str, need: Need) -> Result<(), OfflineError> {
    guard_when(is_offline(), target, need)
}

/// [`guard`] with offline mode given as `offline`.
pub fn guard_when(offline: bool, target: &str, need: Need) -> Result<(), OfflineError> {
    if offline {
        return Err(OfflineError {
            target: target.to_string(),
            need,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_value_but_zero_or_false_turns_offline_on() {
        for on in ["1", "true", "TRUE", "yes", "anything"] {
            assert!(offline_value(Some(on)), "{on}");
        }
        for off in ["", "0", "false", "False", "FALSE"] {
            assert!(!offline_value(Some(off)), "{off}");
        }
        assert!(!offline_value(None));
    }

    #[test]
    fn the_guard_blocks_only_when_offline() {
        assert_eq!(guard_when(false, "org/name", Need::HubModel), Ok(()));
        let err =
            guard_when(true, "org/name: config.json", Need::HubModel).expect_err("offline blocks");
        let text = err.to_string();
        assert!(
            text.starts_with("org/name: config.json needs the network"),
            "{text}"
        );
        assert!(text.contains("HF_HUB_OFFLINE is set"), "{text}");
        assert!(text.contains("pass its local path"), "{text}");
    }

    #[test]
    fn a_blocked_image_names_the_url_and_the_inline_fix() {
        let err = guard_when(true, "https://example.com/a.png", Need::RemoteImage)
            .expect_err("offline blocks");
        let text = err.to_string();
        assert!(
            text.starts_with("https://example.com/a.png needs the network"),
            "{text}"
        );
        assert!(text.contains("data: URL"), "{text}");
    }
}
