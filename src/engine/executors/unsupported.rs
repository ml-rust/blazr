//! Shared error text for decode paths that don't yet carry qwen35's Gated
//! DeltaNet (GDN) state alongside the KV cache.

use anyhow::anyhow;

/// Build the standard "qwen35 not supported" error for `path`.
///
/// `context` is an optional parenthesized qualifier right after `path`
/// (e.g. a model name); `detail` is an optional explanation appended after
/// a colon. Both are `None` for paths that need no further explanation.
pub(crate) fn qwen35_unsupported(
    path: &str,
    context: Option<&str>,
    detail: Option<&str>,
) -> anyhow::Error {
    let mut msg = format!("qwen35 not supported on the {path} path yet");
    if let Some(c) = context {
        msg.push_str(&format!(" ({c})"));
    }
    if let Some(d) = detail {
        msg.push_str(&format!(": {d}"));
    }
    anyhow!(msg)
}
