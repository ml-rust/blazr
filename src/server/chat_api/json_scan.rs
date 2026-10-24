//! Scans over a JSON object that is still arriving.

/// The keys a tool-call object can open with.
const CALL_KEYS: [&str; 2] = ["name", "arguments"];

/// What the first key of a JSON object says about a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FirstKey {
    /// The key has not fully arrived, and it can still be a call key.
    Pending,
    /// The key is `name` or `arguments`.
    CallKey,
    /// The object opens with anything else, so it is no tool call.
    Other,
}

/// Classify the first key of `text`, which starts with `{`.
///
/// A key with an escape counts as [`FirstKey::Other`].
pub(super) fn first_key(text: &str) -> FirstKey {
    let rest = text.strip_prefix('{').unwrap_or(text).trim_start();
    let Some(key) = rest.strip_prefix('"') else {
        return if rest.is_empty() {
            FirstKey::Pending
        } else {
            FirstKey::Other
        };
    };
    match key.find(['"', '\\']) {
        Some(end) if key.as_bytes()[end] == b'"' => {
            if CALL_KEYS.contains(&&key[..end]) {
                FirstKey::CallKey
            } else {
                FirstKey::Other
            }
        }
        Some(_) => FirstKey::Other,
        None if CALL_KEYS.iter().any(|k| k.starts_with(key)) => FirstKey::Pending,
        None => FirstKey::Other,
    }
}

/// A resumable scan for the end of the JSON object a text starts with.
///
/// Braces inside strings do not count, and escapes inside strings are
/// honoured. Each call to [`ObjectScan::advance`] reads only the bytes the
/// last call did not see.
#[derive(Debug, Clone, Default)]
pub(super) struct ObjectScan {
    pos: usize,
    depth: usize,
    in_string: bool,
    escaped: bool,
    /// The object's length, once its braces balance.
    end: Option<usize>,
}

impl ObjectScan {
    /// The byte length of the object at the start of `text`, or `None`
    /// while it is still open.
    ///
    /// `text` must extend the text of the previous call on this scan.
    pub(super) fn advance(&mut self, text: &str) -> Option<usize> {
        if self.end.is_some() {
            return self.end;
        }
        let bytes = text.as_bytes();
        while self.pos < bytes.len() {
            let byte = bytes[self.pos];
            self.pos += 1;
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == b'"' {
                    self.in_string = false;
                }
                continue;
            }
            match byte {
                b'"' => self.in_string = true,
                b'{' => self.depth += 1,
                b'}' => {
                    self.depth = self.depth.saturating_sub(1);
                    if self.depth == 0 {
                        self.end = Some(self.pos);
                        return self.end;
                    }
                }
                _ => {}
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_key_classifies_each_prefix() {
        assert_eq!(first_key("{"), FirstKey::Pending);
        assert_eq!(first_key("{ \"na"), FirstKey::Pending);
        assert_eq!(first_key("{\"name\""), FirstKey::CallKey);
        assert_eq!(first_key("{\"arguments\": {}"), FirstKey::CallKey);
        assert_eq!(first_key("{\"nam\""), FirstKey::Other);
        assert_eq!(first_key("{\"id\""), FirstKey::Other);
        assert_eq!(first_key("{\"x"), FirstKey::Other);
        assert_eq!(first_key("{}"), FirstKey::Other);
        assert_eq!(first_key("{\"na\\u006de\""), FirstKey::Other);
    }

    #[test]
    fn a_scan_skips_braces_inside_strings() {
        let text = r#"{"a":"} { \" }","b":{"c":1}} tail"#;
        let mut scan = ObjectScan::default();
        assert_eq!(scan.advance(text), Some(text.len() - " tail".len()));
    }

    #[test]
    fn a_scan_resumes_across_pieces() {
        let text = r#"{"a":"x\"}","b":{}}"#;
        let mut scan = ObjectScan::default();
        for end in 1..text.len() {
            assert_eq!(scan.advance(&text[..end]), None, "{}", &text[..end]);
        }
        assert_eq!(scan.advance(text), Some(text.len()));
        let longer = format!("{text} and {{more}}");
        assert_eq!(scan.advance(&longer), Some(text.len()), "the end stays put");
    }
}
