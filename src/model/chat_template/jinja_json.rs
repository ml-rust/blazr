//! JSON text in the shape Python's `json.dumps` writes.
//!
//! Chat templates embed tool schemas with `tojson`, and the decision
//! templates embed a request's state, so both need the Python spelling the
//! model saw in training:
//!
//! - separators: `", "` and `": "` by default, `","` and `": "` with an
//!   indent, or the pair the caller passes
//! - keys in insertion order, or sorted by code point with `sort_keys`
//! - non-ASCII text as is, or every character outside `' '..='~'` escaped
//!   with `ensure_ascii`
//! - floats as Python's `repr`: `1e-05`, `1e+16`, `1.0`, `-0.0`

use std::fmt::Write;

/// The item and key separators Python uses when the caller passes none.
fn default_separators(indent: Option<usize>) -> (&'static str, &'static str) {
    match indent {
        Some(_) => (",", ": "),
        None => (", ", ": "),
    }
}

/// Serialize `value` like
/// `json.dumps(value, indent=indent, separators=separators,
/// ensure_ascii=ensure_ascii, sort_keys=sort_keys)`.
///
/// `separators` is `(item_separator, key_separator)`. `None` takes Python's
/// default for the indent.
pub fn python_json(
    value: &serde_json::Value,
    indent: Option<usize>,
    separators: Option<(&str, &str)>,
    ensure_ascii: bool,
    sort_keys: bool,
) -> String {
    let (item, key) = separators.unwrap_or_else(|| default_separators(indent));
    let mut out = String::new();
    let writer = Writer {
        indent,
        item,
        key,
        ensure_ascii,
        sort_keys,
    };
    writer.value(&mut out, value, 0);
    out
}

/// Python's `repr` of a float: the shortest digits that read back to the
/// same value, in exponent form below `1e-4` and from `1e16` up.
///
/// `nan`, `inf` and `-inf` are spelled as Python spells them. JSON text
/// uses [`python_json_float`] instead.
pub fn python_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    // `{:e}` writes the shortest round-trip digits as `d.ddde<exp>`.
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    // Python's decimal-point position: the value is 0.<digits> * 10^decpt.
    let decpt = exp + 1;
    if decpt <= -4 || decpt > 16 {
        let (first, rest) = digits.split_at(1);
        let dot = if rest.is_empty() { "" } else { "." };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        return format!(
            "{sign}{first}{dot}{rest}e{exp_sign}{:02}",
            exp.unsigned_abs()
        );
    }
    if decpt <= 0 {
        let zeros = "0".repeat(decpt.unsigned_abs() as usize);
        return format!("{sign}0.{zeros}{digits}");
    }
    let point = decpt as usize;
    if point >= digits.len() {
        let zeros = "0".repeat(point - digits.len());
        format!("{sign}{digits}{zeros}.0")
    } else {
        let (whole, frac) = digits.split_at(point);
        format!("{sign}{whole}.{frac}")
    }
}

/// A float as `json.dumps` writes it: its `repr`, with `NaN`, `Infinity`
/// and `-Infinity` for the values JSON has no spelling for.
pub fn python_json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
    } else {
        python_float_repr(x)
    }
}

struct Writer<'a> {
    indent: Option<usize>,
    item: &'a str,
    key: &'a str,
    ensure_ascii: bool,
    sort_keys: bool,
}

impl Writer<'_> {
    fn value(&self, out: &mut String, value: &serde_json::Value, depth: usize) {
        match value {
            serde_json::Value::Null => out.push_str("null"),
            serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            serde_json::Value::Number(n) => number(out, n),
            serde_json::Value::String(s) => self.string(out, s),
            serde_json::Value::Array(items) => self.array(out, items, depth),
            serde_json::Value::Object(map) => self.object(out, map, depth),
        }
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
                // Python's `ensure_ascii` keeps `' '..='~'` and escapes the
                // rest, DEL included.
                c if self.ensure_ascii && !(' '..='~').contains(&c) => {
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
            // UTF-8 byte order is code point order, which is Python's.
            entries.sort_by(|a, b| a.0.cmp(b.0));
        }
        out.push('{');
        for (i, (key, value)) in entries.iter().enumerate() {
            self.separator(out, i, depth + 1);
            self.string(out, key);
            out.push_str(self.key);
            self.value(out, value, depth + 1);
        }
        self.close(out, depth);
        out.push('}');
    }

    /// Text before item `i` of a container whose items sit at `depth`.
    fn separator(&self, out: &mut String, i: usize, depth: usize) {
        if i > 0 {
            out.push_str(self.item);
        }
        if let Some(width) = self.indent {
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * depth));
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

/// An integer as written; a float as Python's `repr`.
fn number(out: &mut String, n: &serde_json::Number) {
    if let Some(i) = n.as_i64() {
        let _ = write!(out, "{i}");
    } else if let Some(u) = n.as_u64() {
        let _ = write!(out, "{u}");
    } else if let Some(f) = n.as_f64() {
        out.push_str(&python_json_float(f));
    } else {
        let _ = write!(out, "{n}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected string below is the output of CPython 3.12's
    // `json.dumps` or `repr` for the same input.

    #[test]
    fn compact_matches_python_separators() {
        let v =
            serde_json::json!({"type": "function", "n": [1, 2.5, true, null], "s": "a\"b\n<é>"});
        assert_eq!(
            python_json(&v, None, None, false, false),
            r#"{"type": "function", "n": [1, 2.5, true, null], "s": "a\"b\n<é>"}"#
        );
    }

    #[test]
    fn ensure_ascii_escapes_non_ascii() {
        let v = serde_json::json!("é😀");
        assert_eq!(
            python_json(&v, None, None, true, false),
            "\"\\u00e9\\ud83d\\ude00\""
        );
    }

    /// `ensure_ascii` escapes DEL; without it DEL is written raw.
    #[test]
    fn ensure_ascii_escapes_del() {
        let v = serde_json::json!("\u{7f}\u{1f}");
        assert_eq!(
            python_json(&v, None, None, true, false),
            r#""\u007f\u001f""#
        );
        assert_eq!(
            python_json(&v, None, None, false, false),
            "\"\u{7f}\\u001f\""
        );
    }

    #[test]
    fn sort_keys_orders_recursively() {
        let v = serde_json::json!({"b": {"d": 1, "c": 2}, "a": 0});
        assert_eq!(
            python_json(&v, None, None, false, true),
            r#"{"a": 0, "b": {"c": 2, "d": 1}}"#
        );
    }

    #[test]
    fn indent_matches_python_layout() {
        let v = serde_json::json!({"a": [1, {"b": 2}], "c": {}, "d": []});
        assert_eq!(
            python_json(&v, Some(2), None, false, false),
            "{\n  \"a\": [\n    1,\n    {\n      \"b\": 2\n    }\n  ],\n  \"c\": {},\n  \"d\": []\n}"
        );
    }

    #[test]
    fn control_characters_use_unicode_escapes() {
        let v = serde_json::json!("x\u{1}y\u{8}z\u{c}");
        assert_eq!(
            python_json(&v, None, None, false, false),
            "\"x\\u0001y\\bz\\f\""
        );
    }

    fn sample() -> serde_json::Value {
        serde_json::json!({"b": [1, 2.5, true, null, {"z": "é\u{7f}\u{2028}"}], "a": "x\ty", "c": 1e-05})
    }

    /// `json.dumps(v, separators=(",", ":"), sort_keys=True, ensure_ascii=False)`.
    #[test]
    fn compact_sorted_separators() {
        assert_eq!(
            python_json(&sample(), None, Some((",", ":")), false, true),
            "{\"a\":\"x\\ty\",\"b\":[1,2.5,true,null,{\"z\":\"é\u{7f}\u{2028}\"}],\"c\":1e-05}"
        );
    }

    /// `json.dumps(v, ensure_ascii=False)` and `json.dumps(v)`.
    #[test]
    fn default_separators_keep_insertion_order() {
        assert_eq!(
            python_json(&sample(), None, None, false, false),
            "{\"b\": [1, 2.5, true, null, {\"z\": \"é\u{7f}\u{2028}\"}], \"a\": \"x\\ty\", \"c\": 1e-05}"
        );
        assert_eq!(
            python_json(&sample(), None, None, true, false),
            r#"{"b": [1, 2.5, true, null, {"z": "\u00e9\u007f\u2028"}], "a": "x\ty", "c": 1e-05}"#
        );
    }

    /// `json.dumps(v, indent=2, separators=(", ", ": "))`: the item separator
    /// keeps its space before each newline.
    #[test]
    fn explicit_separators_with_indent() {
        assert_eq!(
            python_json(&sample(), Some(2), Some((", ", ": ")), true, false),
            "{\n  \"b\": [\n    1, \n    2.5, \n    true, \n    null, \n    {\n      \"z\": \"\\u00e9\\u007f\\u2028\"\n    }\n  ], \n  \"a\": \"x\\ty\", \n  \"c\": 1e-05\n}"
        );
    }

    #[test]
    fn float_repr_matches_python() {
        let cases: [(f64, &str); 21] = [
            (1e-05, "1e-05"),
            (1e16, "1e+16"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (-0.0, "-0.0"),
            (1e-4, "0.0001"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (9999999999999998.0, "9999999999999998.0"),
            (1e22, "1e+22"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (0.30000000000000004, "0.30000000000000004"),
            (2.5, "2.5"),
            (100.0, "100.0"),
            (1e15, "1000000000000000.0"),
            (1.5e-7, "1.5e-07"),
            (12345.678, "12345.678"),
            (-1.5e16, "-1.5e+16"),
            (3.14e-10, "3.14e-10"),
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "-inf"),
        ];
        for (x, want) in cases {
            assert_eq!(python_float_repr(x), want, "repr of {x:e}");
        }
        assert_eq!(python_float_repr(f64::INFINITY), "inf");
    }

    #[test]
    fn json_float_spells_non_finite_like_python() {
        assert_eq!(python_json_float(f64::NAN), "NaN");
        assert_eq!(python_json_float(f64::INFINITY), "Infinity");
        assert_eq!(python_json_float(f64::NEG_INFINITY), "-Infinity");
        assert_eq!(python_json_float(1e16), "1e+16");
        let v = serde_json::json!([1e-05, 1e16, 1.0, -0.0, 7]);
        assert_eq!(
            python_json(&v, None, None, false, false),
            "[1e-05, 1e+16, 1.0, -0.0, 7]"
        );
    }
}
