//! Ollama-style `keep_alive` values.

use std::time::Duration;

/// Parse an Ollama-style keep_alive string into a Duration.
///
/// Supported formats:
/// - `"0"`, `"0s"`, `"0m"` → `Some(Duration::ZERO)` (unload immediately)
/// - Any negative value, such as `"-1"` or `"-5m"` → `None` (keep forever)
/// - `"5m"` → 5 minutes
/// - `"1h"` → 1 hour
/// - `"30s"` → 30 seconds
/// - `"300"` → 300 seconds (bare number = seconds)
///
/// An unparseable value is `None`: keep forever.
pub fn parse_keep_alive(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.starts_with('-') {
        return None; // forever
    }
    if let Some(mins) = s.strip_suffix('m') {
        if let Ok(n) = mins.parse::<u64>() {
            return Some(Duration::from_secs(n * 60));
        }
    }
    if let Some(hours) = s.strip_suffix('h') {
        if let Ok(n) = hours.parse::<u64>() {
            return Some(Duration::from_secs(n * 3600));
        }
    }
    if let Some(secs) = s.strip_suffix('s') {
        if let Ok(n) = secs.parse::<u64>() {
            return Some(Duration::from_secs(n));
        }
    }
    // Bare number = seconds
    if let Ok(n) = s.parse::<u64>() {
        return Some(Duration::from_secs(n));
    }
    None // unparseable → default (forever)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_and_bare_seconds_parse() {
        assert_eq!(parse_keep_alive("5m"), Some(Duration::from_secs(300)));
        assert_eq!(parse_keep_alive("1h"), Some(Duration::from_secs(3600)));
        assert_eq!(parse_keep_alive("30s"), Some(Duration::from_secs(30)));
        assert_eq!(parse_keep_alive(" 300 "), Some(Duration::from_secs(300)));
    }

    #[test]
    fn zero_in_any_unit_unloads() {
        for zero in ["0", "0s", "0m", "0h"] {
            assert_eq!(parse_keep_alive(zero), Some(Duration::ZERO), "{zero}");
        }
    }

    #[test]
    fn every_negative_form_keeps_forever() {
        for negative in ["-1", "-5", "-5m", "-1s", "-2h"] {
            assert_eq!(parse_keep_alive(negative), None, "{negative}");
        }
    }
}
