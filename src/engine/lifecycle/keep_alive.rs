//! Ollama-style `keep_alive` values.

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Deserializer};

/// A `keep_alive` value that does not parse, with the forms that do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeepAliveError {
    /// The value as sent.
    pub value: String,
}

impl fmt::Display for KeepAliveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "keep_alive {:?} is not a duration. Send whole seconds such as 300, a whole number \
             with s, m or h such as 5m, 0 to unload, or a negative value to keep the model loaded",
            self.value
        )
    }
}

impl std::error::Error for KeepAliveError {}

/// Parse an Ollama-style keep_alive string into a Duration.
///
/// Supported formats:
/// - `"0"`, `"0s"`, `"0m"` → `Ok(Some(Duration::ZERO))` (unload immediately)
/// - Any negative value, such as `"-1"` or `"-5m"` → `Ok(None)` (keep forever)
/// - `"5m"` → 5 minutes
/// - `"1h"` → 1 hour
/// - `"30s"` → 30 seconds
/// - `"300"` → 300 seconds (bare number = seconds)
///
/// # Errors
///
/// [`KeepAliveError`] for any other value, such as `"5min"`, `"1.5h"` or
/// `"30S"`, and for a duration too long to hold.
pub fn parse_keep_alive(s: &str) -> Result<Option<Duration>, KeepAliveError> {
    let s = s.trim();
    if s.starts_with('-') {
        return Ok(None); // forever
    }
    let (digits, unit) = match s.char_indices().last() {
        Some((i, 'm')) => (&s[..i], 60),
        Some((i, 'h')) => (&s[..i], 3600),
        Some((i, 's')) => (&s[..i], 1),
        _ => (s, 1),
    };
    let err = || KeepAliveError {
        value: s.to_string(),
    };
    // `u64::from_str` takes a leading `+`. Only digits are a duration.
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(err());
    }
    let n: u64 = digits.parse().map_err(|_| err())?;
    let secs = n.checked_mul(unit).ok_or_else(err)?;
    Ok(Some(Duration::from_secs(secs)))
}

/// A request's `keep_alive`, checked when the request parses: how long the
/// model stays loaded after it. `None` keeps it loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeepAlive(pub Option<Duration>);

impl std::str::FromStr for KeepAlive {
    type Err = KeepAliveError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_keep_alive(s).map(Self)
    }
}

impl<'de> Deserialize<'de> for KeepAlive {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_and_bare_seconds_parse() {
        assert_eq!(parse_keep_alive("5m"), Ok(Some(Duration::from_secs(300))));
        assert_eq!(parse_keep_alive("1h"), Ok(Some(Duration::from_secs(3600))));
        assert_eq!(parse_keep_alive("30s"), Ok(Some(Duration::from_secs(30))));
        assert_eq!(
            parse_keep_alive(" 300 "),
            Ok(Some(Duration::from_secs(300)))
        );
    }

    #[test]
    fn zero_in_any_unit_unloads() {
        for zero in ["0", "0s", "0m", "0h"] {
            assert_eq!(parse_keep_alive(zero), Ok(Some(Duration::ZERO)), "{zero}");
        }
    }

    #[test]
    fn every_negative_form_keeps_forever() {
        for negative in ["-1", "-5", "-5m", "-1s", "-2h"] {
            assert_eq!(parse_keep_alive(negative), Ok(None), "{negative}");
        }
    }

    /// An unparseable value used to mean "keep forever".
    #[test]
    fn an_unparseable_value_is_an_error() {
        for bad in ["5min", "1.5h", "30S", "abc", "", "m", "+5m", "5 m", "1d"] {
            let err = parse_keep_alive(bad).expect_err(bad);
            assert!(
                err.to_string().contains("is not a duration"),
                "{bad}: {err}"
            );
        }
    }

    /// `n * 60` and `n * 3600` used to overflow on a huge `n`.
    #[test]
    fn a_duration_too_long_to_hold_is_an_error() {
        let max = u64::MAX;
        assert_eq!(
            parse_keep_alive(&format!("{max}s")),
            Ok(Some(Duration::from_secs(max)))
        );
        assert!(parse_keep_alive(&format!("{max}m")).is_err());
        assert!(parse_keep_alive(&format!("{}h", max / 3600 + 1)).is_err());
    }

    #[test]
    fn a_request_field_checks_its_value() {
        let ok: KeepAlive = serde_json::from_str("\"5m\"").expect("parses");
        assert_eq!(ok, KeepAlive(Some(Duration::from_secs(300))));
        let err = serde_json::from_str::<KeepAlive>("\"5min\"").expect_err("rejects");
        assert!(err.to_string().contains("is not a duration"), "{err}");
    }
}
