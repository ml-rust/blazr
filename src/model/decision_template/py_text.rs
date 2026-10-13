//! Python text semantics the reference renderers rely on.
//!
//! - whitespace: `str.isspace`, `str.strip()`, `str.split()` and the regex
//!   `\s`. Python counts U+001C..U+001F as whitespace, `char::is_whitespace`
//!   does not.
//! - decimal digits: the regex `\d` and `float()` take every character of
//!   Unicode category `Nd`, not ASCII alone.
//! - `str()` of a JSON scalar: `True`, `False`, `None`, an integer, or a
//!   float's `repr`.
//! - `float()` of a string, with its whitespace, underscore and
//!   `inf`/`nan` rules.
//! - truthiness of a JSON value.
//! - `repr()` for the error messages that quote a value.
//!
//! The tables are those of CPython 3.12 (Unicode 15.0), the interpreter
//! the reference renderers run on.
//!
//! `str()` of a list or an object is Python's `repr` of a dict or list.
//! Prompt text never takes that path here: [`py_str`] refuses it.

use serde_json::Value;

use crate::model::chat_template::jinja_json::python_float_repr;

/// Every character `str.isspace` accepts.
const PY_SPACE: [char; 29] = [
    '\t', '\n', '\u{b}', '\u{c}', '\r', '\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', ' ', '\u{85}',
    '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}',
    '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}',
    '\u{205f}', '\u{3000}',
];

/// The digit zero of every Unicode `Nd` block, in code point order. Each
/// block holds the digits 0..=9 in order. Hand-listed from the `Nd` ranges
/// of Unicode 15.0 (`UnicodeData.txt`), the version of CPython 3.12.
const ND_ZEROS: [u32; 68] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];

/// Whether Python's `str.isspace` (and the regex `\s`) accepts `c`.
pub fn is_py_space(c: char) -> bool {
    PY_SPACE.contains(&c)
}

/// `s.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `s.split()`: the runs between whitespace, no empty pieces.
pub fn py_split(s: &str) -> Vec<&str> {
    s.split(is_py_space).filter(|p| !p.is_empty()).collect()
}

/// The value of `c` when it is a Unicode decimal digit (category `Nd`).
pub fn nd_digit(c: char) -> Option<u32> {
    let cp = c as u32;
    let i = ND_ZEROS.partition_point(|&zero| zero <= cp);
    let zero = *ND_ZEROS.get(i.checked_sub(1)?)?;
    (cp - zero < 10).then_some(cp - zero)
}

/// Python's `str(value)` of a JSON scalar.
///
/// # Errors
///
/// A list or an object, whose `str()` is Python's `repr` of a list or a
/// dict. The message names `what`.
pub fn py_str(value: &Value, what: &str) -> Result<String, String> {
    match value {
        Value::Null => Ok("None".to_string()),
        Value::Bool(true) => Ok("True".to_string()),
        Value::Bool(false) => Ok("False".to_string()),
        Value::Number(n) => Ok(py_number(n)),
        Value::String(s) => Ok(s.clone()),
        Value::Array(_) | Value::Object(_) => Err(format!(
            "{what} is a JSON {}: the reference renderer writes Python's repr of it, which is \
             not reproduced here. Send a string",
            json_kind(value)
        )),
    }
}

/// `list`, `object`, `string`, ... for a message about `value`.
pub fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "list",
        Value::Object(_) => "object",
    }
}

fn py_number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        n.as_f64()
            .map(python_float_repr)
            .unwrap_or_else(|| n.to_string())
    }
}

/// Python's `bool(value)`.
pub fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python's `repr(value)`, for an error message that quotes it.
///
/// Printability of a non-ASCII character follows Python for controls,
/// separators and the common format characters; any other character is
/// written as is.
pub fn py_repr(value: &Value) -> String {
    match value {
        Value::String(s) => py_repr_str(s),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr_str(k), py_repr(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        scalar => py_str(scalar, "a scalar").unwrap_or_default(),
    }
}

/// Python's `repr(s)` of a string.
pub fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !is_printable(c) => {
                let cp = c as u32;
                if cp < 0x100 {
                    out.push_str(&format!("\\x{cp:02x}"));
                } else if cp < 0x10000 {
                    out.push_str(&format!("\\u{cp:04x}"));
                } else {
                    out.push_str(&format!("\\U{cp:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `str.isprintable` for one character: controls, separators other than
/// the space, format characters and private-use characters are not.
fn is_printable(c: char) -> bool {
    let cp = c as u32;
    if c == ' ' {
        return true;
    }
    if cp < 0x20 || (0x7f..=0xa0).contains(&cp) || is_py_space(c) {
        return false;
    }
    !matches!(cp,
        0xad
        | 0x600..=0x605
        | 0x61c
        | 0x6dd
        | 0x70f
        | 0x180e
        | 0x200b..=0x200f
        | 0x202a..=0x202e
        | 0x2060..=0x2064
        | 0x2066..=0x206f
        | 0xe000..=0xf8ff
        | 0xfeff
        | 0xfff9..=0xfffb
        | 0xf0000..)
}

/// Python's `float(s)`.
///
/// Leading and trailing whitespace goes, non-ASCII decimal digits count,
/// `_` may sit between two digits, and `inf`, `infinity` and `nan` take any
/// case and an optional sign.
///
/// # Errors
///
/// Python's own message: `could not convert string to float: '<s>'`.
pub fn py_float(s: &str) -> Result<f64, String> {
    let err = || format!("could not convert string to float: {}", py_repr_str(s));
    // CPython keeps ASCII as is, turns other whitespace into ' ' and other
    // decimal digits into ASCII ones, and fails on anything else.
    let mut ascii = String::with_capacity(s.len());
    for c in s.chars() {
        if (c as u32) < 127 {
            ascii.push(c);
        } else if is_py_space(c) {
            ascii.push(' ');
        } else if let Some(d) = nd_digit(c).and_then(|d| char::from_digit(d, 10)) {
            ascii.push(d);
        } else {
            return Err(err());
        }
    }
    let text = ascii.trim_matches(|c| matches!(c, ' ' | '\t' | '\n' | '\u{b}' | '\u{c}' | '\r'));
    let (negative, body) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let lower = body.to_ascii_lowercase();
    let magnitude = if lower == "inf" || lower == "infinity" {
        f64::INFINITY
    } else if lower == "nan" {
        f64::NAN
    } else {
        let cleaned = without_underscores(body).ok_or_else(err)?;
        if !is_decimal_literal(&cleaned) {
            return Err(err());
        }
        cleaned.parse::<f64>().map_err(|_| err())?
    };
    Ok(if negative { -magnitude } else { magnitude })
}

/// `s` without its `_`, when each one sits between two ASCII digits.
fn without_underscores(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'_' {
            let before = i.checked_sub(1).and_then(|j| bytes.get(j));
            let after = bytes.get(i + 1);
            if !(before.is_some_and(u8::is_ascii_digit) && after.is_some_and(u8::is_ascii_digit)) {
                return None;
            }
        } else {
            out.push(char::from(b));
        }
    }
    Some(out)
}

/// `digits [. digits] [e [sign] digits]` or `. digits [...]`, nothing else.
fn is_decimal_literal(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    let digits = |i: &mut usize| {
        let start = *i;
        while bytes.get(*i).is_some_and(u8::is_ascii_digit) {
            *i += 1;
        }
        *i - start
    };
    let mut mantissa = digits(&mut i);
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        mantissa += digits(&mut i);
    }
    if mantissa == 0 {
        return false;
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if digits(&mut i) == 0 {
            return false;
        }
    }
    i == bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Expected values are CPython 3.12's output for the same input.

    #[test]
    fn strip_and_split_use_python_whitespace() {
        // " a\x1cb ".strip() == "a\x1cb"; " a\x1cb c ".split() == ["a", "b", "c"]
        assert_eq!(py_strip(" a\u{1c}b "), "a\u{1c}b");
        assert_eq!(py_split(" a\u{1c}b c "), vec!["a", "b", "c"]);
        assert_eq!(py_strip("\u{1f}x\u{3000}"), "x");
        assert!(!is_py_space('\u{200b}'));
        assert!(!'\u{1c}'.is_whitespace(), "Rust disagrees, hence the table");
    }

    #[test]
    fn nd_digits_cover_every_block() {
        assert_eq!(nd_digit('7'), Some(7));
        assert_eq!(nd_digit('\u{661}'), Some(1));
        assert_eq!(nd_digit('\u{ff12}'), Some(2));
        assert_eq!(nd_digit('\u{1d7d7}'), Some(9));
        assert_eq!(nd_digit('\u{1d7d8}'), Some(0));
        assert_eq!(nd_digit('\u{b2}'), None, "superscript two is No, not Nd");
        assert_eq!(nd_digit('a'), None);
        assert_eq!(nd_digit('\u{2f}'), None);
    }

    #[test]
    fn str_of_scalars() {
        assert_eq!(py_str(&json!(true), "x").as_deref(), Ok("True"));
        assert_eq!(py_str(&json!(null), "x").as_deref(), Ok("None"));
        assert_eq!(py_str(&json!(1000), "x").as_deref(), Ok("1000"));
        assert_eq!(py_str(&json!(1.0), "x").as_deref(), Ok("1.0"));
        assert_eq!(py_str(&json!(1e16), "x").as_deref(), Ok("1e+16"));
        assert!(py_str(&json!({"a": 1}), "x").is_err());
        assert!(py_str(&json!([1]), "x").is_err());
    }

    #[test]
    fn repr_picks_quotes_like_python() {
        assert_eq!(py_repr_str("it's"), "\"it's\"");
        assert_eq!(py_repr_str("say \"hi\""), "'say \"hi\"'");
        assert_eq!(py_repr_str("both ' \""), "'both \\' \"'");
        assert_eq!(py_repr_str("tab\t\u{1}é\u{2028}"), "'tab\\t\\x01é\\u2028'");
        assert_eq!(
            py_repr(&json!({"a": [true, null, 1.5, "x"]})),
            "{'a': [True, None, 1.5, 'x']}"
        );
    }

    #[test]
    fn truthiness() {
        for v in [
            json!(null),
            json!(false),
            json!(0),
            json!(0.0),
            json!(""),
            json!([]),
            json!({}),
        ] {
            assert!(!py_truthy(&v), "{v}");
        }
        for v in [
            json!(true),
            json!(2),
            json!("0"),
            json!([0]),
            json!({"a": null}),
        ] {
            assert!(py_truthy(&v), "{v}");
        }
    }

    #[test]
    fn float_parses_like_python() {
        for (s, want) in [
            (" 1 ", 1.0),
            ("1_000", 1000.0),
            ("1e3", 1000.0),
            ("Infinity", f64::INFINITY),
            ("-inf", f64::NEG_INFINITY),
            ("\u{661}\u{662}", 12.0),
            (".5", 0.5),
            ("5.", 5.0),
            ("\u{3000}2\u{a0}", 2.0),
        ] {
            assert_eq!(py_float(s), Ok(want), "{s:?}");
        }
        assert!(py_float("nan").is_ok_and(f64::is_nan));
        for (s, msg) in [
            ("1__0", "could not convert string to float: '1__0'"),
            ("_1", "could not convert string to float: '_1'"),
            ("1e", "could not convert string to float: '1e'"),
            ("0x1", "could not convert string to float: '0x1'"),
            (
                " \u{1c}2\u{1f}",
                "could not convert string to float: ' \\x1c2\\x1f'",
            ),
        ] {
            assert_eq!(py_float(s), Err(msg.to_string()), "{s:?}");
        }
    }
}
