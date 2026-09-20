//! JSON text in the shape Python's `json.dumps` writes.
//!
//! Chat templates embed tool schemas with `tojson`, and the model saw the
//! Python spelling in training: `", "` between items, `": "` after a key,
//! non-ASCII text left as is unless `ensure_ascii` is set.

use std::fmt::Write;

/// Serialize `value` like `json.dumps(value, indent, ensure_ascii, sort_keys)`.
pub fn python_json(
    value: &serde_json::Value,
    indent: Option<usize>,
    ensure_ascii: bool,
    sort_keys: bool,
) -> String {
    let mut out = String::new();
    let writer = Writer {
        indent,
        ensure_ascii,
        sort_keys,
    };
    writer.value(&mut out, value, 0);
    out
}

struct Writer {
    indent: Option<usize>,
    ensure_ascii: bool,
    sort_keys: bool,
}

impl Writer {
    fn value(&self, out: &mut String, value: &serde_json::Value, depth: usize) {
        match value {
            serde_json::Value::Null => out.push_str("null"),
            serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            serde_json::Value::Number(n) => self.number(out, n),
            serde_json::Value::String(s) => self.string(out, s),
            serde_json::Value::Array(items) => self.array(out, items, depth),
            serde_json::Value::Object(map) => self.object(out, map, depth),
        }
    }

    /// Python prints a float that holds a whole number with a trailing `.0`.
    /// `serde_json::Number` does the same, so the two agree.
    fn number(&self, out: &mut String, n: &serde_json::Number) {
        // Writing a number into a `String` cannot fail.
        let _ = write!(out, "{n}");
    }

    fn string(&self, out: &mut String, s: &str) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(out, "\\u{:04x}", c as u32);
                }
                c if self.ensure_ascii && !c.is_ascii() => {
                    let mut units = [0u16; 2];
                    for unit in c.encode_utf16(&mut units) {
                        let _ = write!(out, "\\u{:04x}", unit);
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }

    fn array(&self, out: &mut String, items: &[serde_json::Value], depth: usize) {
        if items.is_empty() {
            out.push_str("[]");
            return;
        }
        out.push('[');
        for (i, item) in items.iter().enumerate() {
            self.separator(out, i, depth + 1);
            self.value(out, item, depth + 1);
        }
        self.close(out, depth);
        out.push(']');
    }

    fn object(
        &self,
        out: &mut String,
        map: &serde_json::Map<String, serde_json::Value>,
        depth: usize,
    ) {
        if map.is_empty() {
            out.push_str("{}");
            return;
        }
        let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
        if self.sort_keys {
            entries.sort_by(|a, b| a.0.cmp(b.0));
        }
        out.push('{');
        for (i, (key, value)) in entries.iter().enumerate() {
            self.separator(out, i, depth + 1);
            self.string(out, key);
            out.push_str(": ");
            self.value(out, value, depth + 1);
        }
        self.close(out, depth);
        out.push('}');
    }

    /// Text before item `i` of a container whose items sit at `depth`.
    fn separator(&self, out: &mut String, i: usize, depth: usize) {
        match self.indent {
            Some(width) => {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                out.extend(std::iter::repeat_n(' ', width * depth));
            }
            None => {
                if i > 0 {
                    out.push_str(", ");
                }
            }
        }
    }

    /// Text before the closing bracket of a container at `depth`.
    fn close(&self, out: &mut String, depth: usize) {
        if let Some(width) = self.indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * depth));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_matches_python_separators() {
        let v =
            serde_json::json!({"type": "function", "n": [1, 2.5, true, null], "s": "a\"b\n<é>"});
        assert_eq!(
            python_json(&v, None, false, false),
            r#"{"type": "function", "n": [1, 2.5, true, null], "s": "a\"b\n<é>"}"#
        );
    }

    #[test]
    fn ensure_ascii_escapes_non_ascii() {
        let v = serde_json::json!("é😀");
        assert_eq!(
            python_json(&v, None, true, false),
            "\"\\u00e9\\ud83d\\ude00\""
        );
    }

    #[test]
    fn sort_keys_orders_recursively() {
        let v = serde_json::json!({"b": {"d": 1, "c": 2}, "a": 0});
        assert_eq!(
            python_json(&v, None, false, true),
            r#"{"a": 0, "b": {"c": 2, "d": 1}}"#
        );
    }

    #[test]
    fn indent_matches_python_layout() {
        let v = serde_json::json!({"a": [1, {"b": 2}], "c": {}, "d": []});
        assert_eq!(
            python_json(&v, Some(2), false, false),
            "{\n  \"a\": [\n    1,\n    {\n      \"b\": 2\n    }\n  ],\n  \"c\": {},\n  \"d\": []\n}"
        );
    }

    #[test]
    fn control_characters_use_unicode_escapes() {
        let v = serde_json::json!("x\u{1}y\u{8}z\u{c}");
        assert_eq!(python_json(&v, None, false, false), "\"x\\u0001y\\bz\\f\"");
    }
}
