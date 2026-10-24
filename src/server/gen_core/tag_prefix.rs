//! The tail of streamed text that can still grow into a tag.

/// Length of the longest suffix of `text` that is a strict prefix of `tag`.
///
/// A stream parser holds this many bytes back until the next piece shows
/// whether they open `tag`. The suffix always starts on a char boundary.
pub(in crate::server) fn partial_prefix_len(text: &str, tag: &str) -> usize {
    let max = text.len().min(tag.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&k| text.is_char_boundary(text.len() - k) && text.ends_with(&tag[..k]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_strict_prefix_at_the_end_is_held() {
        assert_eq!(partial_prefix_len("abc <tool_c", "<tool_call>"), 7);
        assert_eq!(partial_prefix_len("abc <", "<tool_call>"), 1);
    }

    #[test]
    fn a_whole_tag_or_no_prefix_holds_nothing() {
        assert_eq!(partial_prefix_len("<tool_call>", "<tool_call>"), 0);
        assert_eq!(partial_prefix_len("<b>", "<tool_call>"), 0);
        assert_eq!(partial_prefix_len("", "<tool_call>"), 0);
    }

    #[test]
    fn a_multibyte_tail_is_not_split() {
        assert_eq!(partial_prefix_len("caf\u{e9}", "<tool_call>"), 0);
        assert_eq!(partial_prefix_len("\u{e9}<t", "<tool_call>"), 2);
    }
}
