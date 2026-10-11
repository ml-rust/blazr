//! Ollama-style `keep_alive` values.

use std::time::Duration;

/// Parse an Ollama-style keep_alive string into a Duration.
///
/// Supported formats:
/// - `"0"` → `Some(Duration::ZERO)` (unload immediately)
/// - `"-1"` → `None` (keep forever)
/// - `"5m"` → 5 minutes
/// - `"1h"` → 1 hour
/// - `"30s"` → 30 seconds
/// - `"300"` → 300 seconds (bare number = seconds)
pub fn parse_keep_alive(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s == "-1" {
        return None; // forever
    }
    if s == "0" {
        return Some(Duration::ZERO);
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
